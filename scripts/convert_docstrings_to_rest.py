#!/usr/bin/env python3
"""Convert Markdown-ish docstrings in the localuf package to reStructuredText.

This script is intentionally conservative:
- Only rewrites actual docstrings (module/class/function first-statement strings).
- Does NOT touch other string literals.
- Converts inline code from `code` -> ``code``.
- Converts sections labeled "Input:"/"Output:" into :param/:returns field lists when possible.

Run from the repo root:
python scripts/convert_docstrings_to_rest.py
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path


_RE_SINGLE_TICK_CODE = re.compile(r"(?<!`)`([^`\n]+?)`(?!`)")


def _convert_inline_code(text: str) -> str:
    # `foo` -> ``foo`` (leave existing ``foo`` alone)
    return _RE_SINGLE_TICK_CODE.sub(r"``\1``", text)


_HEADER_INPUT = re.compile(r"^(additional\s+)?inputs?:\s*$", re.IGNORECASE)
_HEADER_OUTPUT = re.compile(r"^(additional\s+)?outputs?:\s*$", re.IGNORECASE)
_HEADER_SIDE_EFFECT = re.compile(r"^side\s+effect\s*:\s*$", re.IGNORECASE)


def _strip_bullet_prefix(s: str) -> str:
    return re.sub(r"^\s*([*\-]\s+)", "", s)


def _looks_like_identifier(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", s))


def _parse_param_item(line: str) -> tuple[str, str] | None:
    """Parse an item line into (name, desc) if possible."""
    s = _strip_bullet_prefix(line).strip()

    # e.g. "positive integer `n` is ..."
    m = re.match(r"^positive\s+integer\s+(``|`)([^`]+?)\1\s*(.*)$", s, flags=re.IGNORECASE)
    if m:
        return m.group(2).strip(), m.group(3).strip()

    # e.g. "`decoder` the decoder" or "``decoder`` the decoder"
    m = re.match(r"^``([^`]+?)``\s*(.*)$", s)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = re.match(r"^`([^`]+?)`\s*(.*)$", s)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    # e.g. "decoder the decoder" (fallback)
    parts = s.split(None, 1)
    if not parts:
        return None
    if _looks_like_identifier(parts[0]):
        name = parts[0]
        desc = parts[1].strip() if len(parts) > 1 else ""
        return name, desc

    return None


def _collect_block(lines: list[str], start: int) -> tuple[list[str], int]:
    """Collect block lines until next blank line or recognized header."""
    out: list[str] = []
    i = start
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped == "":
            break
        if _HEADER_INPUT.match(stripped) or _HEADER_OUTPUT.match(stripped) or _HEADER_SIDE_EFFECT.match(stripped):
            break
        out.append(lines[i])
        i += 1
    return out, i


def _convert_input_block(block_lines: list[str]) -> list[str]:
    out: list[str] = []
    current_field: str | None = None
    for raw in block_lines:
        if raw.strip() == "":
            continue

        is_bullet = bool(re.match(r"^\s*[*\-]\s+", raw))
        if is_bullet:
            parsed = _parse_param_item(raw)
            if parsed is None:
                out.append(raw.strip())
                current_field = None
                continue
            name, desc = parsed
            if desc:
                out.append(f":param {name}: {desc}")
            else:
                out.append(f":param {name}:")
            current_field = "param"
            continue

        # Continuation line: attach only if it is indented in the source.
        if raw[:1].isspace() and current_field == "param":
            out.append(f"    {raw.strip()}")
        else:
            out.append(raw.strip())
            current_field = None
    return out


def _convert_output_block(block_lines: list[str]) -> list[str]:
    # If we see bullet items with return components, keep them as a bullet list under :returns:
    nonempty = [ln for ln in block_lines if ln.strip()]
    if not nonempty:
        return []

    # One-liner "`0` if ..." etc.
    if len(nonempty) == 1:
        s = _strip_bullet_prefix(nonempty[0].strip())
        return [f":returns: {s}"]

    bulletish = all(re.match(r"^\s*[*\-]\s+", ln) for ln in nonempty)
    if bulletish:
        out = [":returns:"]
        for ln in nonempty:
            item = _strip_bullet_prefix(ln.strip())
            out.append(f"* {item}")
        return out

    # Otherwise: fold into a single :returns: paragraph.
    text = " ".join(ln.strip() for ln in nonempty)
    return [f":returns: {text}"]


def convert_docstring(text: str) -> str:
    """Convert Markdown-ish docstring text to reStructuredText."""
    text = _convert_inline_code(text)

    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        if _HEADER_INPUT.match(stripped):
            out.append("")
            block, j = _collect_block(lines, i + 1)
            converted = _convert_input_block(block)
            out.extend(converted)
            i = j
            continue

        if _HEADER_OUTPUT.match(stripped):
            out.append("")
            block, j = _collect_block(lines, i + 1)
            converted = _convert_output_block(block)
            out.extend(converted)
            i = j
            continue

        # Keep side-effect header as-is but ensure inline code is reST-literal.
        if _HEADER_SIDE_EFFECT.match(stripped):
            out.append(_convert_inline_code(lines[i]))
            i += 1
            continue

        out.append(_convert_inline_code(lines[i]))
        i += 1

    # Clean up: collapse excessive blank lines
    cleaned: list[str] = []
    blank_run = 0
    for ln in out:
        if ln.strip() == "":
            blank_run += 1
            if blank_run <= 2:
                cleaned.append("")
        else:
            blank_run = 0
            cleaned.append(ln.rstrip())

    # Preserve trailing newline presence
    return "\n".join(cleaned).rstrip() + "\n"


@dataclass(frozen=True)
class _Span:
    start: int
    end: int


def _line_offsets(src: str) -> list[int]:
    lines = src.splitlines(keepends=True)
    offsets: list[int] = []
    pos = 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln)
    offsets.append(pos)
    return offsets


def _indent_at(src: str, lineno: int, col: int) -> str:
    line = src.splitlines(keepends=True)[lineno - 1]
    return line[:col]


def _choose_delim(doc: str) -> str:
    if '"""' not in doc:
        return '"""'
    if "'''" not in doc:
        return "'''"
    # Extremely rare: escape triple double-quotes.
    return '"""'


def _format_docstring_literal(doc: str, indent: str) -> str:
    doc = doc.rstrip("\n")
    delim = _choose_delim(doc)
    lines = doc.splitlines()
    if not lines:
        return f"{delim}{delim}"

    # Escape only if needed.
    if delim == '"""' and '"""' in doc:
        doc = doc.replace('"""', r'\\"\\"\\"')
        lines = doc.splitlines()

    if delim == "'''" and "'''" in doc:
        doc = doc.replace("'''", r"\\'\\'\\'")
        lines = doc.splitlines()

    if len(lines) == 1:
        return f"{delim}{lines[0]}{delim}"

    # Replacement span starts after leading indentation on the opening line.
    # So do NOT include indent before the opening delimiter; only subsequent lines.
    out_lines = [f"{delim}{lines[0]}"]
    out_lines.extend(f"{indent}{ln}" for ln in lines[1:])
    out_lines.append(f"{indent}{delim}")
    return "\n".join(out_lines)


def _docstring_nodes(tree: ast.AST):
    """Yield (Constant node, indent level) for all real docstrings."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.body:
                continue
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                yield first.value


def convert_file(path: Path) -> bool:
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False

    offsets = _line_offsets(src)
    replacements: list[tuple[_Span, str]] = []

    for const in _docstring_nodes(tree):
        if not hasattr(const, "lineno") or not hasattr(const, "end_lineno"):
            continue
        lineno = const.lineno
        col = const.col_offset
        end_lineno = const.end_lineno
        end_col = const.end_col_offset
        if lineno is None or end_lineno is None:
            continue

        start = offsets[lineno - 1] + col
        end = offsets[end_lineno - 1] + end_col

        original_literal = src[start:end]
        original_value = const.value
        if not isinstance(original_value, str):
            continue

        new_value = convert_docstring(original_value)
        indent = _indent_at(src, lineno, col)
        new_literal = _format_docstring_literal(new_value, indent)

        if new_literal != original_literal:
            replacements.append((_Span(start, end), new_literal))

    if not replacements:
        return False

    # Apply from back to front.
    new_src = src
    for span, repl in sorted(replacements, key=lambda x: x[0].start, reverse=True):
        new_src = new_src[: span.start] + repl + new_src[span.end :]

    if new_src != src:
        path.write_text(new_src, encoding="utf-8")
        return True

    return False


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    pkg_root = repo_root / "localuf"
    changed = 0
    for path in sorted(pkg_root.rglob("*.py")):
        if path.name.startswith("__pycache__"):
            continue
        if convert_file(path):
            changed += 1
    print(f"Converted docstrings in {changed} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
