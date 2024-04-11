from typing import Annotated

from cyclopts import App, Parameter, validators

try:
	from signal import signal, SIGPIPE, SIG_DFL
	signal(SIGPIPE, SIG_DFL)
except ImportError:
	pass

app = App(
    name="swiftrank", help="Rerank contexts provided on stdin.",
)

@app.command(name="process", help="STDIN processor. [ json | jsonl | yaml ]")
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
    from .utils import cli_object_parser, print_and_exit
    def preprocessor(_input: str):
        if _input.startswith(('{', '[')):
            from orjson import loads, JSONDecodeError
            try:
                return cli_object_parser(loads(_input), pre)
            except JSONDecodeError:
                try:
                    from io import StringIO
                    with StringIO(_input) as handler:
                        return list(map(loads, handler))
                except (JSONDecodeError, Exception):
                    print_and_exit("Input data format not valid.", code=1)
            except Exception:
                print_and_exit("Input data format not valid.", code=1)
        
        import yaml    
        try:
            return cli_object_parser(yaml.safe_load(_input), pre)
        except yaml.MarkedYAMLError:
            return list(yaml.safe_load_all(_input))
        except yaml.YAMLError:
            print_and_exit("Input data format not valid.", code=1)
    
    return {'preprocessor': preprocessor, 'ctx_schema': ctx, 'post_schema': post}


@app.meta.default
def __entry__(
    *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)], 
    query: Annotated[str, Parameter(
        name=("-q", "--query"), help="query for reranking evaluation.")],
    threshold: Annotated[float, Parameter(
        name=("-t", "--threshold"), help="filter contexts using threshold.", validator=validators.Number(gte=0.0, lte=1.0))] = None,
    first: Annotated[bool, Parameter(
        name=("-f", "--first"), help="get most relevant context.", negative="", show_default=False)] = False,
):
    from .utils import read_stdin, cli_object_parser, print_and_exit
    
    processing_params: dict = {}
    if tokens:
        processing_params = app(tokens=tokens)
        _input = read_stdin()
        if not _input:
            return
        contexts = processing_params['preprocessor'](_input)
    else:
        contexts = read_stdin(readlines=True)

    ctx_schema = processing_params.get('ctx_schema', '.')
    post_schema = processing_params.get('post_schema') or ctx_schema

    if not isinstance(contexts, list):
        print_and_exit(cli_object_parser(contexts, post_schema))

    if not contexts:
        print_and_exit("Nothing to rerank!", code=1)

    from .. import settings
    from ..ranker import ReRankPipeline

    pipeline = ReRankPipeline.from_model_id(settings.DEFAULT_MODEL)
    try:
        reranked = pipeline.invoke(
            query=query, 
            contexts=contexts, 
            threshold=threshold, 
            key=lambda x: cli_object_parser(x, ctx_schema)
        )

        if reranked and first:
            print_and_exit(
                cli_object_parser(reranked[0], post_schema)
            )
            
        for context in reranked:
            print(cli_object_parser(context, post_schema))

    except TypeError:
        print_and_exit(
            'Context processing must result into string.', code=1
        )

@app.meta.command(name="serve", help="Startup a swiftrank server")
def serve(
    *, 
    host: Annotated[str, Parameter(
        name=('--host'), help="Host name")] = '0.0.0.0',
    port: Annotated[int, Parameter(
        name=('--port',), help="Port number.")] = 12345
):
    from .api import _serve
    _serve(host=host, port=port)

if __name__ == "__main__":
    app.meta()