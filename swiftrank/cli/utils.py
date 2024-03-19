import sys
from typing import TypeAlias, Any

def print_and_exit(msg: str, code: int = 0):
    stream = sys.stdout if not code else sys.stderr
    print(msg, file=stream)
    exit(code)

def read_stdin(readlines: bool = False):
    """Read values from standard input (stdin). """
    if sys.stdin.isatty():
        return
    try:
        if readlines is False:
            return sys.stdin.read().rstrip('\n')
        return [_.strip('\n') for _ in sys.stdin if _]
    except KeyboardInterrupt:
        return


ObjectCollection: TypeAlias = dict[str, Any] | list[Any]
ObjectScalar: TypeAlias = bool | float | int | str
ObjectValue: TypeAlias = ObjectCollection | ObjectScalar    

def object_parser(obj: ObjectValue, schema: str) -> ObjectValue:
    import re
    from itertools import groupby

    if schema == '.':
        return obj
    
    usable_schema = '.['.join(re.split(r'(?:\[|\.\[)', schema))
    if not re.match(
        pattern=r"^(?:(?:[.](?:[\w]+|\[\d?\]))+)$", string=usable_schema
    ):
        print_and_exit(f'{schema!r} is not a valid schema.', code=1)

    def __inner__(_in: ObjectValue, keys: list[str]):
        for idx, key in enumerate(keys):
            _match = re.search(r'^\[(\d)?\]$', key)
            try:
                if _match is None:
                    _in = _in[key]
                    continue

                obj_idx = _match.group(1)
                if obj_idx is not None:
                    _in = _in[int(obj_idx)]
                    continue
            except (KeyError, IndexError):
                print_and_exit(f'{schema!r} schema not compatible with input data.', code=1)

            _keys = keys[idx + 1:]
            if not _keys:
                return _in

            return [__inner__(item, _keys) for item in _in][0]
        return _in
    return __inner__(
        obj, [k for k, _ in groupby(usable_schema.lstrip('.').split('.'))]
    )