import sys
from typing import TypeAlias, Any, Annotated

from cyclopts import App, Parameter

def __version__() -> str:
    import importlib.metadata

    try:
        return importlib.metadata.version("swiftrank")
    except importlib.metadata.PackageNotFoundError:
        import re, pathlib

        return re.search(
            r'name = "swiftrank"\nversion = "(.+?)"',
            (pathlib.Path(__file__).parent.parent / "pyproject.toml").read_text(),
        ).group(1)


cli = App(
    name="swiftrank", version=__version__(), help="Rerank contexts provided on stdin."
)

try:
	from signal import signal, SIGPIPE, SIG_DFL
	signal(SIGPIPE, SIG_DFL)
except ImportError:
	pass

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


@cli.command(name="process", help="STDIN processor. [ json | jsonl | yaml ]")
def build_processing_parameters(
    *,
    pre: Annotated[str, Parameter(name=('-r', '--pre'),
        help="schema for pre-processing input.", show_default=False)] = '.',
    ctx: Annotated[str, Parameter(name=('-c', '--ctx'),
        help="schema for extracting context.", show_default=False)] = '.',
    post: Annotated[str, Parameter(name=('-p', '--post'),
        help="schema for extracting field after reranking.", show_default=False
    )] = None
):  
    def preprocessor(_input: str):
        if _input.startswith(('{', '[')):
            from orjson import loads, JSONDecodeError
            try:
                return object_parser(loads(_input), pre)
            except JSONDecodeError:
                from io import StringIO
                with StringIO(_input) as handler:
                    return list(map(loads, handler))
            except Exception:
                print_and_exit("Malformed JSON object not parseable.", code=1)
        
        import yaml    
        try:
            return object_parser(yaml.safe_load(_input), pre)
        except yaml.MarkedYAMLError:
            return list(yaml.safe_load_all(_input))
        except yaml.YAMLError:
            print_and_exit("Input data format not valid.", code=1)
    
    return {'preprocessor': preprocessor, 'ctx_schema': ctx, 'post_schema': post}


@cli.meta.default
def __entry__(
    *tokens: Annotated[str, Parameter(show=False)], 
    query: Annotated[str, Parameter(
        name=("-q", "--query"), help="query for reranking evaluation.")],
    threshold: Annotated[float, Parameter(
        name=("-t", "--threshold"), help="filter contexts using threshold.")] = None,
    first: Annotated[bool, Parameter(
        name=("-f", "--first"), help="get most relevant context.", negative="", show_default=False)] = False
): 
    processing_params: dict = {}
    if tokens:
        processing_params = cli(tokens=tokens)
        _input = read_stdin()
        if not _input:
            return
        contexts = processing_params['preprocessor'](_input)
            
    else:
        contexts = read_stdin(readlines=True)

    ctx_schema = processing_params.get('ctx_schema', '.')
    if not isinstance(contexts, list):
        print_and_exit(object_parser(contexts, ctx_schema))

    if not all(contexts):
        print_and_exit("No contexts found on stdin", code=1)
    if len(contexts) == 1:
        print_and_exit(contexts[0])

    from . import settings
    from .ranker import Ranker, Tokenizer, ReRankPipeline

    _model = settings.DEFAULT_MODEL
    pipeline = ReRankPipeline(
        ranker=Ranker(_model), tokenizer=Tokenizer(_model)
    )
    try:
        reranked = list(pipeline.invoke(
            query=query, 
            contexts=contexts, 
            threshold=threshold, 
            key=lambda x: object_parser(x, ctx_schema)
        ))
    except TypeError:
        print_and_exit(
            'Context processing must result into string. Hint: `--ctx` flag might help.', code=1
        )

    post_schema = processing_params.get('post_schema') or ctx_schema
    if reranked and first:
        print_and_exit(
            object_parser(reranked[0]['context'], post_schema)
        )

    for mapping in reranked:
        print(object_parser(mapping["context"], post_schema))