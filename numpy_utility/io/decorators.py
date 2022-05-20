import numpy as np
import functools

__all__ = [
    "save_and_load"
]


def save_and_load(filename, suffix=".npz", arg=None):
    import inspect
    import pathlib
    from string import Formatter
    _formatter = Formatter()

    def _outer(func):
        @functools.wraps(func)
        def _inner(*args, **kwargs):
            keys_from_args = {name: v for name, v in zip(inspect.signature(func).parameters.keys(), args)}
            for k in keys_from_args:
                if k in kwargs:
                    raise TypeError(f"{func.__name__}() got multiple values for argument '{k}'")

            kwargs.update({
                f"{param_info.name}": param_info.default
                for param_info in inspect.signature(func).parameters.values()
                if param_info.name not in kwargs
            })

            kwargs.update(keys_from_args)

            if not any([
                name not in kwargs or kwargs[name] is None
                for _, name, _, _ in _formatter.parse(filename)
            ]):
                buffer_name = pathlib.Path(filename.format(**kwargs)).with_suffix(suffix)

                if buffer_name.exists():
                    print(f"* load {buffer_name}")

                    npz_obj = np.load(buffer_name, allow_pickle=True)
                    if arg is None:
                        return npz_obj["arr_0"]
                    elif arg in npz_obj.keys():
                        return npz_obj[arg]
                else:
                    npz_obj = None

                results = func(**kwargs)
                print(f"* save {buffer_name}")
                if arg is None:
                    np.savez(buffer_name, results)
                else:
                    if npz_obj is None:
                        np.savez(buffer_name, **{f"{arg}": results})
                    else:
                        np.savez(buffer_name, **npz_obj, **{f"{arg}": results})

            return func(**kwargs)

        return _inner

    return _outer
