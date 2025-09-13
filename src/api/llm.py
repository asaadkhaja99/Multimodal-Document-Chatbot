import json
import queue
import threading

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.pipelines.indexing import IndexingPipeline
from src.pipelines.query import QueryPipeline

# Load environment variables
load_dotenv() 

# Init FastAPI applicaton
app = FastAPI(
    title="Haystack RAG Streaming API",
    description="An API that streams responses from a Haystack pipeline",
)

# Initialize Pipelines
indexing_pipeline = IndexingPipeline()
indexing_pipeline.run_pipeline()
query_pipeline = QueryPipeline(indexing_pipeline=indexing_pipeline)

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query(request: QueryRequest):
    def response_generator():
        stream_queue = queue.Queue()
        
        def run_pipeline_in_thread():
            chat_generator = query_pipeline.pipeline.get_component("chat_generator")
            unsafe_response = query_pipeline.pipeline.get_component("unsafe_response")

            def queue_streaming_callback(chunk):
                if chunk.content:
                    stream_queue.put(chunk.content)

            try:
                chat_generator.streaming_callback = queue_streaming_callback
                unsafe_response.streaming_callback = queue_streaming_callback

                query_pipeline.pipeline.run({
                    "text_embedder": {"text": request.query},
                    "chat_prompt_builder": {"question": request.query}
                })
            finally:
                stream_queue.put(None)

        thread = threading.Thread(target=run_pipeline_in_thread)
        thread.start()

        while True:
            chunk = stream_queue.get()
            if chunk is None:
                break
            # Format as a Server-Sent Event
            yield f"data: {json.dumps({'token': chunk})}\n\n"
        
        thread.join()

    return StreamingResponse(response_generator(), media_type="text/event-stream")
