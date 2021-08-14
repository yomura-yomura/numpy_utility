import numpy as np
import numpy_utility as npu
import itertools
import datetime as dt

__all__ = ["print_structured_array"]


def _to_dict_as_str(a: np.ndarray):
    if a.dtype.names is None:
        return a.astype(str)
    else:
        return {name: _to_dict_as_str(a[name]) for name in a.dtype.names}


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
    # max_n_length_of_names = [
    #     list(map(lambda names: len(names[i]) if i < len(names) else 0, a_str_flatten_dict.keys()))
    #     for i in range(max(map(len, a_str_flatten_dict.keys())))
    # ]
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
                    zip(
                        [names[i] if i < len(names) else "" for names in a_str_flatten_dict.keys()],
                        max_n_length_for_col
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
