#!/usr/bin/env python3
"""Normalize bullet indentation under ``:returns:`` in docstrings.

VS Code’s docstring rendering tends to look better when bullet lists that
belong to a field list item are not additionally indented.

This script finds real docstrings (module/class/function docstrings) and
rewrites patterns like:

:returns:
* foo
* bar

to:

:returns:
* foo
* bar

Run from the repo root:
python scripts/fix_returns_bullet_indentation.py
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path


_RE_RETURNS_LINE = re.compile(r"^(?P<indent>\s*):returns:\s*$")


def normalize_returns_bullets(doc: str) -> str:
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
            m_bullet = re.match(rf"^{re.escape(base)} {{4}}([*\-]\s+.*)$", nxt)
            if not m_bullet:
                break
            out.append(f"{base}{m_bullet.group(1)}")
            i += 1

    return "\n".join(out)


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

    # The replacement span starts after leading indentation on the opening line.
    # So do NOT include indent before the opening delimiter; only subsequent lines.
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

        new_value = normalize_returns_bullets(original_value)
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
    changed = 0
    for path in sorted(repo_root.rglob("*.py")):
        if any(part in {".git", "__pycache__"} for part in path.parts):
            continue
        if convert_file(path):
            changed += 1
    print(f"Normalized :returns: bullets in {changed} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
