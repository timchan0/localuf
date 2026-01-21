#!/usr/bin/env python3
"""Format reST-style docstrings for readability.

This is a conservative formatter:
- Only rewrites real docstrings (module/class/function first-statement strings).
- Does NOT touch other string literals.

Currently it normalizes a few patterns introduced by automated conversion:
- Removes excessive common indentation in multi-line docstrings.
- Rewrites inline-bulleted returns like ``:returns: * a * b`` into:

:returns:
* a
* b

- Aligns bullet lists directly under ``:returns:`` (no extra indentation).

Run from repo root:
python scripts/format_rest_docstrings.py
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path


_RE_RETURNS_INLINE_BULLETS = re.compile(r"^(?P<indent>\s*):returns:\s*[*\-]\s+(?P<body>.+?)\s*$")
_RE_RETURNS_LINE = re.compile(r"^(?P<indent>\s*):returns:\s*$")

_RE_FIELD_LIST = re.compile(r"^\s*:(param|returns|return|rtype|type|raises|raise)\b", re.IGNORECASE)
_RE_CODE_BLOCK = re.compile(r"^\s*\.\.\s+code-block::")


def _leading_ws_len(s: str) -> int:
    return len(s) - len(s.lstrip(" \t"))


def _normalize_alignment(doc: str) -> str:
    """Normalize indentation inside docstrings.
    
    After Python's usual docstring dedent (e.g. ``inspect.getdoc``), reST
    content should be left-aligned, with indentation used only for
    continuation lines.
    """
    lines = doc.splitlines()
    out: list[str] = []

    prev_kind: str | None = None  # 'field' | 'bullet' | 'other'
    in_literal_block = False

    for ln in lines:
        if not ln.strip():
            out.append("")
            prev_kind = None
            continue

        # Enter literal block on explicit markers.
        if _RE_CODE_BLOCK.match(ln) or (out and out[-1].rstrip().endswith("::")):
            in_literal_block = True

        # Exit literal block once we hit a non-indented, non-blank line.
        if in_literal_block and _leading_ws_len(ln) == 0 and ln.strip():
            in_literal_block = False

        if in_literal_block:
            out.append(ln.rstrip("\n"))
            prev_kind = "other"
            continue

        content = ln.lstrip(" \t")
        is_field = bool(_RE_FIELD_LIST.match(content))
        is_bullet = bool(re.match(r"^[*\-]\s+", content))

        if is_field:
            out.append(content)
            prev_kind = "field"
            continue

        if is_bullet:
            out.append(content)
            prev_kind = "bullet"
            continue

        # Continuation lines under a field list or bullet list.
        if prev_kind in {"field", "bullet"} and _leading_ws_len(ln) > 0:
            out.append("    " + content)
        else:
            out.append(content)
        prev_kind = "other"

    return "\n".join(out)


def _normalize_returns_inline_bullets(doc: str) -> str:
    out: list[str] = []
    for ln in doc.splitlines():
        m = _RE_RETURNS_INLINE_BULLETS.match(ln)
        if not m:
            out.append(ln)
            continue

        indent = m.group("indent")
        body = m.group("body")
        items = [s.strip() for s in re.split(r"\s+\*\s+", body) if s.strip()]

        out.append(f"{indent}:returns:")
        for item in items or [body.strip()]:
            out.append(f"{indent}* {item}")

    return "\n".join(out)


def _align_returns_bullets(doc: str) -> str:
    """Ensure bullets immediately under :returns: align with the field."""
    lines = doc.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)

        m = _RE_RETURNS_LINE.match(line)
        if not m:
            i += 1
            continue

        base = m.group("indent")
        i += 1
        while i < len(lines):
            nxt = lines[i]
            m_bullet = re.match(rf"^{re.escape(base)}\s+([*\-]\s+.*)$", nxt)
            if not m_bullet:
                break
            out.append(f"{base}{m_bullet.group(1)}")
            i += 1

    return "\n".join(out)


def format_docstring(doc: str) -> str:
    doc = _normalize_alignment(doc)
    doc = _normalize_returns_inline_bullets(doc)
    doc = _align_returns_bullets(doc)
    return doc


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
    return '"""'


def _format_docstring_literal(doc: str, indent: str) -> str:
    doc = doc.rstrip("\n")
    delim = _choose_delim(doc)
    lines = doc.splitlines()
    if not lines:
        return f"{delim}{delim}"

    if delim == '"""' and '"""' in doc:
        doc = doc.replace('"""', r'\\"\\"\\"')
        lines = doc.splitlines()

    if delim == "'''" and "'''" in doc:
        doc = doc.replace("'''", r"\\'\\'\\'")
        lines = doc.splitlines()

    if len(lines) == 1:
        return f"{delim}{lines[0]}{delim}"

    out_lines = [f"{delim}{lines[0]}"]
    out_lines.extend(f"{indent}{ln}" for ln in lines[1:])
    out_lines.append(f"{indent}{delim}")
    return "\n".join(out_lines)


def _docstring_nodes(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.body:
                continue
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
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
        lineno = getattr(const, "lineno", None)
        end_lineno = getattr(const, "end_lineno", None)
        col = getattr(const, "col_offset", None)
        end_col = getattr(const, "end_col_offset", None)
        if lineno is None or end_lineno is None or col is None or end_col is None:
            continue

        lineno_i = int(lineno)
        end_lineno_i = int(end_lineno)
        col_i = int(col)
        end_col_i = int(end_col)

        start = offsets[lineno_i - 1] + col_i
        end = offsets[end_lineno_i - 1] + end_col_i

        original_literal = src[start:end]
        original_value = const.value
        if not isinstance(original_value, str):
            continue

        new_value = format_docstring(original_value)
        if new_value == original_value:
            continue

        indent = _indent_at(src, lineno_i, col_i)
        new_literal = _format_docstring_literal(new_value, indent)
        if new_literal != original_literal:
            replacements.append((_Span(start, end), new_literal))

    if not replacements:
        return False

    new_src = src
    for span, repl in sorted(replacements, key=lambda x: x[0].start, reverse=True):
        new_src = new_src[: span.start] + repl + new_src[span.end :]

    if new_src != src:
        path.write_text(new_src, encoding="utf-8")
        return True
    return False


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    targets = [repo_root / "localuf", repo_root / "scripts"]

    changed = 0
    for root in targets:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            if any(part in {".git", "__pycache__"} for part in path.parts):
                continue
            if convert_file(path):
                changed += 1

    print(f"Formatted docstrings in {changed} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
