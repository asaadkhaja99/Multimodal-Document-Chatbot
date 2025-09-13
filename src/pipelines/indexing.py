from pathlib import Path

from haystack import Pipeline
from haystack.components.converters import PPTXToDocument, TextFileToDocument
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import RecursiveDocumentSplitter
from haystack.components.routers import FileTypeRouter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.connectors.langfuse import \
    LangfuseConnector
from haystack_integrations.components.embedders.ollama import \
    OllamaDocumentEmbedder
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from loguru import logger

from src.pipelines.custom_components import MultimodalPyPDFToDocument
from src.pipelines.utils import load_config


class IndexingPipeline:
    def __init__(self):
        logger.info("Initializing IndexingPipeline...")
        self.config = load_config()
        self.pipeline = Pipeline()
        self.document_store = None
        self._build_pipeline()
        logger.info("IndexingPipeline initialized successfully.")

    def _build_pipeline(self):
        logger.info("Building the indexing pipeline...")
        # Load configs for components
        store_config = self.config["document_store"]
        embedder_config = self.config["embedder"]
        splitter_config = self.config["splitter"]
        router_config = self.config["file_router"]
        langfuse_config = self.config["langfuse"]

        self.document_store = ChromaDocumentStore(
            collection_name=store_config["collection_name"],
            embedding_function=store_config.get("embedding_function", "default"),
            persist_path=store_config.get("persist_path", "./chroma_db"),
        )
        logger.debug(f"Initialized ChromaDocumentStore with collection '{store_config['collection_name']}'.")
        
        writer = DocumentWriter(
            document_store=self.document_store,
            policy=DuplicatePolicy.SKIP,
        )
        logger.debug("Initialized DocumentWriter with SKIP policy.")

        embedder = OllamaDocumentEmbedder(
            model=embedder_config["model"],
            url=embedder_config["ollama_url"],
            timeout=embedder_config["timeout"],
        )
        logger.debug(f"Initialized OllamaDocumentEmbedder with model '{embedder_config['model']}'.")

        splitter = RecursiveDocumentSplitter(
            split_unit=splitter_config["split_unit"],
            split_length=splitter_config["split_length"],
            split_overlap=splitter_config["split_overlap"],
        )
        logger.debug("Initialized RecursiveDocumentSplitter.")

        router = FileTypeRouter(mime_types=router_config["mime_types"])
        logger.debug("Initialized FileTypeRouter.")
        
        # Instantiate converters
        text_converter = TextFileToDocument()
        pptx_converter = PPTXToDocument()
        pdf_converter = MultimodalPyPDFToDocument()
        joiner = DocumentJoiner()

        # Add components to the pipeline

        # Conditionally add the Langfuse tracer
        if langfuse_config.get("enabled", False):
            logger.info("Langfuse tracing enabled for the Indexing pipeline.")
            self.pipeline.add_component(
                "tracer", 
                LangfuseConnector(langfuse_config["indexing_trace_name"])
            )
        else:
            logger.info("Langfuse tracing is disabled for the Indexing pipeline.")

        self.pipeline.add_component("router", router)
        self.pipeline.add_component("text_converter", text_converter)
        self.pipeline.add_component("pptx_converter", pptx_converter)
        self.pipeline.add_component("pdf_converter", pdf_converter)
        self.pipeline.add_component("joiner", joiner)
        self.pipeline.add_component("splitter", splitter)
        self.pipeline.add_component("embedder", embedder)
        self.pipeline.add_component("writer", writer)
        
        # Connect the components together

        # Routing to converters
        self.pipeline.connect("router.text/plain", "text_converter.sources")
        self.pipeline.connect(
            "router.application/"
            "vnd.openxmlformats-officedocument.presentationml.presentation",
            "pptx_converter.sources",
        )
        self.pipeline.connect("router.application/pdf", "pdf_converter.sources")

        # Joining documents from converters
        self.pipeline.connect("pdf_converter.documents", "joiner.documents")
        self.pipeline.connect("text_converter.documents", "joiner.documents")
        self.pipeline.connect("pptx_converter.documents", "joiner.documents")

        # After joining, split the documents into chunks
        self.pipeline.connect("joiner.documents", "splitter.documents")
        # Then, embed the chunks
        self.pipeline.connect("splitter.documents", "embedder.documents")
        # Finally, write the embedded chunks to the store
        self.pipeline.connect("embedder.documents", "writer.documents")

        logger.success("Indexing pipeline built and connected successfully.")

    def run_pipeline(self):
        data_directory = self.config["paths"]["data_directory"]
        logger.info(f"Starting indexing run for directory: '{data_directory}'")
        
        # Create a set of Path objects for the source files
        source_files = {Path(p) for p in Path(data_directory).glob("**/*") if p.is_file()}
        logger.debug(f"Found {len(source_files)} files in source directory.")
        
        if not source_files:
            logger.warning(f"No files found in '{data_directory}'. Aborting run.")
            return

        # Fetch indexed documents and their file paths, then produce a set of 
        # Path objects
        logger.info("Checking for already indexed files in the document store...")
        indexed_docs = self.document_store.filter_documents()
        indexed_filepaths = {Path(data_directory)/Path(doc.meta.get("file_path")) for doc in indexed_docs}
        
        logger.debug(f"Found {len(indexed_filepaths)} indexed file paths in the store.")
        
        # Calculate the difference
        files_to_add = list(source_files - indexed_filepaths)
        logger.debug(f"Files to be newly indexed: {len(files_to_add)}")

        if not files_to_add:
            logger.info("All documents are already indexed. No new files to process.")
            return

        logger.info(f"Running indexing pipeline on {len(files_to_add)} new files...")
        # The pipeline expects a list of Path objects
        self.pipeline.run({"router": {"sources": [p for p in files_to_add]}})
        logger.success("Indexing complete.")
