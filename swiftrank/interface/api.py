from typing import Any, Optional

from fastapi import FastAPI, status
from fastapi.responses import ORJSONResponse
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, Field

from .utils import ObjectCollection, api_object_parser
from ..settings import MODEL_MAP
from ..ranker import ReRankPipeline

server = FastAPI()
pipeline_map: dict[str, ReRankPipeline] = {}

def get_pipeline(__id: str):
    if pipeline_map.get(__id) is None:
        pipeline_map[__id] = ReRankPipeline.from_model_id(__id)
    return pipeline_map[__id]


class SchemaContext(BaseModel):
    pre: Optional[str] = Field(None, description="schema for pre-processing input.")
    ctx: Optional[str] = Field(None, description="schema for extracting context.")
    post: Optional[str] = Field(None, description="schema for extracting field after reranking.")
    
class RerankContext(BaseModel):
    model: str = Field("ms-marco-TinyBERT-L-2-v2", description="model to use for reranking.")
    contexts: ObjectCollection = Field(..., description="contexts to rerank.")
    query: str = Field(..., description="query for reranking evaluation.")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="filter contexts using threshold.")
    map_score: bool = Field(False, description="map relevance score with context")
    schema_: Optional[SchemaContext] = Field(default=None, alias='schema')


@server.get('/models', response_class=ORJSONResponse)
def list_models():
    return list(MODEL_MAP.keys())

@server.post('/rerank')
def rerank_endpoint(ctx: RerankContext):
    if not ctx.contexts:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="contexts field cannot be an empty array or object"
        )

    if ctx.model not in MODEL_MAP:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{ctx.model!r} model is not available"
        )
    
    schema = ctx.schema_ or SchemaContext()
    if schema.pre is not None:
        contexts = api_object_parser(ctx.contexts, schema=schema.pre)
        if isinstance(contexts, list) and not contexts:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Empty array after pre-processing"
            )
        no_list_err = "Pre-processing must result into an array of objects"

    else:
        contexts = ctx.contexts
        no_list_err = "Expected an array of string or object. 'pre' schema might help"

    if not isinstance(contexts, list):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=no_list_err
        )

    ctx_schema = schema.ctx or '.'
    post_schema = schema.post or '.'
    pipeline = get_pipeline(ctx.model)
    try:
        if ctx.map_score is False:
            reranked = pipeline.invoke(
                query=ctx.query, 
                contexts=contexts, 
                threshold=ctx.threshold,
                key=lambda x: api_object_parser(x, ctx_schema)
            )
            
            return [api_object_parser(context, post_schema) for context in reranked]
        else:
            reranked_tup = pipeline.invoke_with_score(
                query=ctx.query, 
                contexts=contexts, 
                threshold=ctx.threshold,
                key=lambda x: api_object_parser(x, ctx_schema)
            )

            return [
                {'score': score, 'context': api_object_parser(context, post_schema)} 
                for (score, context) in reranked_tup
            ]
    except TypeError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail='Context processing must result into string'
        )
    
def _serve(host: str, port: int):
    import uvicorn
    try:
        uvicorn.run(server, host=host, port=port)
    except KeyboardInterrupt:
        exit(0)
    
if __name__ == "__main__":
    _serve(host='0.0.0.0', port=12345)