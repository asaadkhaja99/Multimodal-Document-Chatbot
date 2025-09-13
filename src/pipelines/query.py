from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.routers import LLMMessagesRouter
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.connectors.langfuse import \
    LangfuseConnector
from haystack_integrations.components.embedders.ollama import \
    OllamaTextEmbedder
from haystack_integrations.components.generators.ollama import \
    OllamaChatGenerator
from haystack_integrations.components.retrievers.chroma import \
    ChromaEmbeddingRetriever
from loguru import logger

from src.pipelines.custom_components import StaticResponseGenerator
from src.pipelines.utils import load_config


class QueryPipeline:
    def __init__(self, indexing_pipeline):
        logger.info("Initializing QueryPipeline...")
        self.config = load_config()
        self.indexing_pipeline = indexing_pipeline
        self.pipeline = Pipeline()
        self._build_pipeline()
        logger.info("QueryPipeline initialized successfully.")

    def _build_pipeline(self):
        logger.info("Building the RAG query pipeline...")
        # Load configs
        llm_config = self.config["llm"]
        embedder_config = self.config["embedder"]
        retriever_config = self.config["retriever"]
        moderation_config = self.config["moderation"]
        langfuse_config = self.config["langfuse"]

        template = [ChatMessage.from_user("""
        Given the following context, please answer the question as accurately as possible.
        If the answer cannot be found in the context, please say so.

        Context:
        {% for document in documents %}
        {{ document.content }}
        {% endfor %}

        Question: {{ question }}
        Answer:
        """)]
        
        document_store = self.indexing_pipeline.document_store
        
        guardrail_generator = OllamaChatGenerator(
            model=llm_config["guardrail_generator"]["model"], 
            url=llm_config["guardrail_generator"]["url"],
        )
        
        router = LLMMessagesRouter(
            chat_generator=guardrail_generator,
            output_names=["unsafe", "safe"],
            output_patterns=[
                moderation_config["router_patterns"]["unsafe"], 
                moderation_config["router_patterns"]["safe"]
            ],
        )
        
        unsafe_response_generator = StaticResponseGenerator()

        text_embedder = OllamaTextEmbedder(
            model=embedder_config["model"],
            url=embedder_config["ollama_url"]
        )
        
        retriever = ChromaEmbeddingRetriever(
            document_store=document_store,
            top_k=retriever_config["top_k"]
        )
        
        chat_generator = OllamaChatGenerator(
            model=llm_config["main_generator"]["model"],  
            url=llm_config["main_generator"]["url"],
        )
        
        chat_prompt_builder = ChatPromptBuilder(
            template=template,
            variables=['question', 'documents'],
            required_variables=['question'],
        )
        
        # Conditionally add the Langfuse tracer
        if langfuse_config.get("enabled", False):
            logger.info("Langfuse tracing enabled for the Query pipeline.")
            self.pipeline.add_component(
                "tracer",
                LangfuseConnector(langfuse_config["query_trace_name"])
            )

        self.pipeline.add_component("moderation_router", router)
        self.pipeline.add_component("unsafe_response", unsafe_response_generator)
        self.pipeline.add_component("text_embedder", text_embedder)
        self.pipeline.add_component("retriever", retriever)
        self.pipeline.add_component("chat_prompt_builder", chat_prompt_builder)
        self.pipeline.add_component("chat_generator", chat_generator)

        # Connect the components
        self.pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        self.pipeline.connect("retriever.documents", "chat_prompt_builder.documents")
        self.pipeline.connect("chat_prompt_builder.prompt", "moderation_router.messages")
        self.pipeline.connect("moderation_router.safe", "chat_generator.messages")
        self.pipeline.connect("moderation_router.unsafe", "unsafe_response.messages") # Pass messages to run method
        
        logger.success("RAG query pipeline built and connected successfully.")
