import logging

import uvicorn
from haystack import tracing
from haystack.tracing.logging_tracer import LoggingTracer

from src.api.llm import app

# Enable Haystack tracking
tracing.tracer.is_content_tracing_enabled = True # to enable tracing/logging content (inputs/outputs)
tracing.enable_tracing(LoggingTracer(tags_color_strings={
    "haystack.component.input": "\x1b[1;31m", 
    "haystack.component.name": "\x1b[1;34m",
}))

# Set up logging
logging.basicConfig(
    format="%(levelname)s - %(name)s -  %(message)s", 
    level=logging.WARNING,
)
logging.getLogger("haystack").setLevel(logging.INFO)



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
