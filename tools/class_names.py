"""Strict class-name loading for CLIP-based comparison methods."""

from pathlib import Path
import re


def load_class_names(path, expected_count=14):
    """Load ordered English class names without inventing prompts.

    Use either one class name per line (line order is the zero-based label
    order), or ``<zero-based-label> <class name>``. Blank lines and comments
    beginning with ``#`` are ignored.
    """
    if not path:
        raise ValueError(
            "A real class-name mapping is required for CLIP experiments. "
            "Pass --class-names-file or set JMDA_CLASS_NAMES_FILE; synthetic "
            "class_1...class_14 prompts are not scientifically valid."
        )
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Class-name mapping does not exist: {source}")

    ordered = []
    indexed = {}
    with source.open('r', encoding='utf-8-sig') as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                label = int(parts[0])
                name = parts[1].strip()
                if not name:
                    raise ValueError(
                        f"Missing class name at {source}:{line_number}."
                    )
                if label in indexed:
                    raise ValueError(f"Duplicate class label {label} in {source}.")
                indexed[label] = name
            else:
                ordered.append(line)

    if indexed and ordered:
        raise ValueError(
            f"Do not mix indexed and ordered class-name rows in {source}."
        )
    if indexed:
        expected_labels = list(range(expected_count))
        if sorted(indexed) != expected_labels:
            raise ValueError(
                f"Indexed class names in {source} must cover labels "
                f"0..{expected_count - 1} exactly."
            )
        names = [indexed[label] for label in expected_labels]
    else:
        names = ordered

    if len(names) != expected_count:
        raise ValueError(
            f"Expected {expected_count} class names in {source}, got {len(names)}."
        )
    if len(set(name.casefold() for name in names)) != len(names):
        raise ValueError(f"Class names must be unique in {source}.")
    non_english = [
        name for name in names
        if not name.isascii() or not re.search(r'[A-Za-z]', name)
    ]
    if non_english:
        raise ValueError(
            'CLIP class names must be descriptive English text in label '
            f'order; invalid entry: {non_english[0]!r}.'
        )
    placeholders = [
        name for name in names
        if re.fullmatch(r'class[ _-]*\d+', name, flags=re.IGNORECASE)
    ]
    if placeholders:
        raise ValueError(
            f"Synthetic placeholder class names are forbidden: {placeholders[0]!r}."
        )
    return names


__all__ = ['load_class_names']
