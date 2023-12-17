import sys

from typer import Typer, Option

cli = Typer(
    rich_help_panel='rich', 
    add_completion=False, 
    context_settings={"help_option_names": ["-h", "--help"]}
)

try:
	from signal import signal, SIGPIPE, SIG_DFL
	signal(SIGPIPE, SIG_DFL)
except ImportError:
	pass

def read_stdin(verify_tty: bool = False):
    """
    Read values from standard input (stdin). 
    If `verify_tty` is True, exit if no input has been piped.
    """
    if verify_tty and sys.stdin.isatty():
        return
    try:
        for line in sys.stdin:
            if line: 
                yield line.strip()
    except KeyboardInterrupt:
        return

@cli.command(
    name="swiftrank", help="Rerank contexts provided on stdin."
)
def __cli__(
    query: str = Option(
        ..., "--query", "-q", help="query for reranking evaluation.", show_default=False),
    threshold: float = Option(
        None, "--threshold", "-t", help="filter contexts using threshold.", show_default=False),
    first: bool = Option(
        False, "--first", "-f", help="get most relevant context.", is_flag=True),
):  
    from typer import echo

    contexts = list(read_stdin())
    if not contexts:
        echo("No contexts found on stdin", err=True)
        return

    from . import settings
    from .ranker import Ranker, Tokenizer, ReRankPipeline

    _model = settings.DEFAULT_MODEL
    pipeline = ReRankPipeline(
        ranker=Ranker(_model), tokenizer=Tokenizer(_model)
    )
    
    reranked = list(pipeline.invoke(
        query=query, contexts=contexts, threshold=threshold
    ))
    if reranked and first:
        echo(reranked[0]['context'])
        return

    for mapping in reranked:
        echo(mapping["context"])