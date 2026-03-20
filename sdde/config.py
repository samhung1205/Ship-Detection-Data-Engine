from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import List, Optional

from .models import ClassMapping


_NAME_LINE_RE = re.compile(r"^\s*name\s*:\s*(?P<rest>.*)\s*$", re.MULTILINE)
_NC_LINE_RE = re.compile(r"^\s*nc\s*:\s*(?P<nc>\d+)\s*$", re.MULTILINE)


def _extract_yaml_list_from_name_field(text: str) -> Optional[List[str]]:
    """
    Extract YAML-ish inline list from the `name:` field.

    Supports the simple form used in this repo, e.g.
      name: ['Naval', 'Merchant']
    """

    m = _NAME_LINE_RE.search(text)
    if not m:
        return None

    rest = m.group("rest").strip()
    if not rest:
        return None

    # If name is multi-line and the list doesn't close on the first line,
    # attempt to accumulate until we see a trailing ']'.
    if rest.startswith("[") and not rest.rstrip().endswith("]"):
        # Naive fallback: join following lines until the first ']'.
        start = text.find(m.group(0))
        tail = text[m.end() :].splitlines(True)
        acc = rest
        for line in tail:
            acc += line.strip()
            if acc.endswith("]"):
                rest = acc
                break

    try:
        parsed = ast.literal_eval(rest)
    except (SyntaxError, ValueError):
        return None

    if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
        return None
    return parsed


def class_mapping_from_data_yaml(path: str | Path) -> ClassMapping:
    """
    Build a ClassMapping from `data.yaml` used by YOLO-style datasets.
    """

    p = Path(path)
    text = p.read_text(encoding="utf-8")

    names = _extract_yaml_list_from_name_field(text)
    if not names:
        raise ValueError(f"Could not parse `name` list from {p}")

    # nc is informational here; the names list order is canonical.
    # Keeping nc check lets us detect mismatch early.
    nc_m = _NC_LINE_RE.search(text)
    if nc_m:
        expected_nc = int(nc_m.group("nc"))
        if expected_nc != len(names):
            raise ValueError(
                f"`nc` mismatch in {p}: nc={expected_nc} but len(name)={len(names)}"
            )

    mapping = ClassMapping(names=names)
    mapping.validate()
    return mapping

