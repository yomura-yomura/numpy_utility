import numpy as np
import numpy_utility as npu
import itertools
import datetime as dt

__all__ = ["print_structured_array"]


def numeric_as_str(n, digits=4):
    if npu.is_floating(n):
        n_digits = int(np.ceil(abs(np.log10(n))))

        if n >= 1:
            if n_digits < digits:
                return f"{n:.{digits - n_digits}f}"
            elif n_digits == digits:
                return f"{n:.0f}."
            else:
                return f"{n:.2e}"
        else:
            if n_digits < digits:
                return f"{n:.{digits}f}"
            else:
                return f"{n:.2e}"

    else:
        return f"{n}"


def as_str(o):
    if npu.is_numeric(o):
        return numeric_as_str(o)
    else:
        return str(o)


def _to_dict_as_str(a: np.ndarray, filled="-"):
    if a.dtype.names is None:
        if a.ndim == 0:
            a = as_str(a)
        elif a.ndim == 1:
            a = list(map(as_str, a))
        else:
            a = [f"({a.ndim}-dim not implemented)"] * len(a)

        if np.ma.isMaskedArray(a):
            return a.filled(filled)
        else:
            return a
    else:
        return {name: _to_dict_as_str(a[name], filled) for name in a.dtype.names}


def flatten_nested_dictionary(d: dict):
    stack = []

    def _parse(dct, *keys):
        for k, v in dct.items():
            if isinstance(v, dict):
                _parse(v, *keys, k)
            else:
                stack.append(((*keys, k), v))

    _parse(d)
    return dict(stack)


def structured_array_to_str(a: np.ndarray, totals=None, latest=None, spacing="  "):
    assert a.dtype.names is not None
    assert a.ndim == 1
    a_str_flatten_dict = flatten_nested_dictionary(_to_dict_as_str(a))
    max_n_length_of_values = list(map(lambda a: max(map(len, a)), a_str_flatten_dict.values()))

    n_depths_of_names = max(map(len, a_str_flatten_dict.keys()))
    max_n_length_of_names = list(map(lambda names: len(names[-1]), a_str_flatten_dict.keys()))
    max_n_length_for_col = list(map(max, zip(max_n_length_of_values, max_n_length_of_names)))

    def cols_to_text(cols):
        return spacing.join(f"{col:>{max_n_len}}" for max_n_len, col in zip(max_n_length_for_col, cols))

    columns_headers = [
        spacing.join(
            f"{name:^{sum(grouped_n_lens) + len(spacing) * (len(grouped_n_lens) - 1)}}"
            for name, grouped_n_lens in (
                (name, [e for _, e in grouped])
                for name, grouped in itertools.groupby(
                sorted(
                    zip(
                        [names[i] if i < len(names) else "" for names in a_str_flatten_dict.keys()],
                        max_n_length_for_col
                    ),
                    key=lambda r: r[0]
                ),
                key=lambda r: r[0]
            )
            )
        )
        for i in range(n_depths_of_names)
    ]

    # columns_header2 = cols_to_text([r for _, r in df.columns])
    table = "\n".join(
        cols_to_text(row)
        for row in zip(*a_str_flatten_dict.values())
    )

    lines = [
        *columns_headers,
        table,
        ""
    ]
    if totals is not None:
        totals_text = cols_to_text(totals)
        assert len(totals_text[:8].strip()) == 0
        lines.append(f"Totals: {totals_text[8:]}")
    if latest is not None:
        latest_text = cols_to_text(latest)
        assert len(latest_text[:8].strip()) == 0
        lines.append(f"Latest: {latest_text[8:]}")
    lines.extend([
        "",
        f"[{len(a)} rows]",
        f"[Current Date and Time:  {dt.datetime.now()}]"
    ])
    return "\n".join(lines)


def print_structured_array(a: np.ndarray, totals=None, latest=None, spacing="  "):
    print(structured_array_to_str(a, totals, latest, spacing))
