import base64
import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
from haystack import Document, component
from haystack.components.converters.utils import (
    get_bytestream_from_source,
    normalize_metadata,
)
from haystack.dataclasses import ByteStream, ChatMessage
from haystack_integrations.components.generators.ollama import \
    OllamaChatGenerator
from loguru import logger
from pypdf import PdfReader

from src.pipelines.utils import load_config

@component
class MultimodalPyPDFToDocument:
    """
    A component that converts PDF files to Documents, extracting both 
    text content and generating descriptions for embedded images using 
    the moondream:1.8b model via direct Ollama API calls.
    
    This component processes PDFs to extract:
    1. Text content using pypdf
    2. Images using pypdf, which are then described using the moondream 
    vision model via direct API calls
    
    Both text and image descriptions are returned as separate Document 
    objects.
    
    Prerequisites:
    - Ollama server running with moondream:1.8b model
    - Run: `ollama pull moondream:1.8b` to download the model
    """

    def __init__(self):
        """
        Initializes the component by loading all its parameters from the central config file.
        """
        logger.info("Initializing MultimodalPyPDFToDocument component from config.")
        
        # Load configurations from the YAML file
        config = load_config()
        converter_config = config["multimodal_converter"]
        # Assuming a structure for the vision model config as designed previously
        vision_llm_config = config["llm"]["vision_generator"]

        # Set parameters from the configuration
        self.store_full_path = converter_config["store_full_path"]
        self.min_image_size = converter_config["min_image_size_bytes"]
        self.max_images_per_page = converter_config["max_images_per_page"]
        self.image_description_prompt = converter_config["image_description_prompt"]
        self.ollama_url = vision_llm_config["url"]
        self.vision_model_name = vision_llm_config["model"]

        logger.debug(
            f"Configuration loaded: min_image_size={self.min_image_size}, "
            f"max_images_per_page={self.max_images_per_page}"
        )

        # Initialize the multimodal LLM using parameters from the config
        self.multimodal_llm = OllamaChatGenerator(
            model=self.vision_model_name,
            url=self.ollama_url,
            generation_kwargs=vision_llm_config.get("generation_kwargs", {})
        )
        logger.debug(
            "Initialized OllamaChatGenerator with model"
            f" '{self.vision_model_name}'."
        )

    def _extract_text_content(self, pdf_reader: PdfReader) -> str:
        """Extract text content from PDF using pypdf."""
        logger.debug(
            "Extracting text from PDF "
            f"with {len(pdf_reader.pages)} pages."
        )
        text_content = ""
        for page in pdf_reader.pages:
            text_content += page.extract_text() + "\n"
        
        stripped_content = text_content.strip()
        logger.debug(f"Extracted {len(stripped_content)} characters of text.")
        return stripped_content

    def _extract_and_describe_images(
        self, 
        pdf_reader: PdfReader, 
        source_info: str,   
    ) -> List[Document]:
        """
        Extract images from PDF using pypdf and generate descriptions 
        using moondream.
        """
        logger.debug(
            f"Starting image extraction and description for '{source_info}'.",
        )
        image_documents = []
        
        for page_num, page in enumerate(pdf_reader.pages):
            images_processed = 0
            
            if not page.images:
                continue

            logger.trace(
                f"Found {len(page.images)} image "
                f"objects on page {page_num + 1}.",
            )
            for img_index, image_file_object in enumerate(page.images):
                if images_processed >= self.max_images_per_page:
                    logger.warning(
                        f"Reached max image limit ({self.max_images_per_page})"
                        f" for page {page_num + 1}. Skipping remaining images"
                        " on this page."
                    )
                    break
                    
                image_bytes = image_file_object.data
                image_name = image_file_object.name
                
                # Skip small images (likely decorative)
                if len(image_bytes) < self.min_image_size:
                    logger.trace(
                        f"Skipping image '{image_name}' (size: "
                        f"{len(image_bytes)} bytes) as it is smaller "
                        f"than min_image_size ({self.min_image_size} bytes).",
                    )
                    continue
                
                # Generate description using moondream
                description = self._generate_image_description(
                    image_bytes, 
                    image_name,
                )
                
                # Create document for image description
                image_doc = Document(
                    content=description,
                    meta={
                        "content_type": "image_description",
                        "source_file": source_info,
                        "page_number": page_num + 1,
                        "image_index": img_index + 1,
                        "image_name": image_name,
                        "image_size_bytes": len(image_bytes),
                        "extraction_method": "multimodal_pypdf"
                    }
                )
                image_documents.append(image_doc)
                images_processed += 1
        
        logger.debug(
            f"Created {len(image_documents)} image description "
            f"documents for '{source_info}'.",
        )
        return image_documents

    def _generate_image_description(
            self, 
            image_bytes: bytes, 
            image_name: str,
        ) -> str:
        """
        Generate a description for an image using the moondream model 
        via Ollama.
        """
        logger.debug(
            f"Generating description for image "
            f"'{image_name}' via Ollama API.",
        )
        # Convert image to base64 for the model
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Call Ollama API directly
        ollama_payload = {
            "model": self.vision_model_name,
            "prompt": self.image_description_prompt,
            "images": [image_base64],
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=ollama_payload
            )
            response.raise_for_status() # Raise an exception for bad status codes
            
            result = response.json()
            description = result.get("response", "").strip()
            if description:
                logger.trace(
                    "Successfully generated "
                    f"description for '{image_name}'.",
                )
                return description
            else:
                logger.warning(
                    "Ollama API returned an "
                    f"empty description for '{image_name}'.",
                )

        except requests.RequestException as e:
            logger.error(
                "Ollama API request failed"
                f" for '{image_name}': {e}"
            )
        
        # Fallback if no description generated or API fails
        fallback_text = f"Image extracted from PDF (name: {image_name},\
            size: {len(image_bytes)} bytes)"
        logger.warning(f"Using fallback text for image '{image_name}'.")
        return fallback_text

    @component.output_types(documents=List[Document])
    def run(
        self,
        sources: List[Union[str, Path, ByteStream]],
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ):
        """
        Converts PDF files to documents, extracting both text and 
        generating image descriptions.
        
        :param sources:
            List of file paths or ByteStream objects to convert.
        :param meta:
            Optional metadata to attach to the documents.
            This value can be a list of dictionaries or a single 
                dictionary.
            If it's a single dictionary, its content is added to the 
                metadata of all produced documents.
            If it's a list, its length must match the number of sources, 
                as they are zipped together.
            For ByteStream objects, their `meta` is added to the output 
                documents.
        :returns:
            A dictionary with the following keys:
            - `documents`: A list of converted documents 
                (both text and image descriptions).
        """
        logger.info(
            f"Running MultimodalPyPDFToDocument on {len(sources)} source(s).",
        )
        documents = []
        meta_list = normalize_metadata(meta, sources_count=len(sources))
        
        for source, metadata in zip(sources, meta_list):
            try:
                bytestream = get_bytestream_from_source(source)
                source_info = str(source)\
                    if isinstance(source, (str, Path))\
                    else bytestream.meta.get("file_path", "ByteStream")
                logger.debug(f"Processing source: {source_info}")
                
                # Extract text content using pypdf
                pdf_reader = PdfReader(io.BytesIO(bytestream.data))
                text_content = self._extract_text_content(pdf_reader)
                
                # Prepare metadata for text document
                merged_metadata = {**bytestream.meta, **metadata}
                if not self.store_full_path and\
                    (file_path := bytestream.meta.get("file_path")):
                    merged_metadata["file_path"] = os.path.basename(file_path)
                
                # Create text document
                if text_content:
                    text_document = Document(
                        content=text_content,
                        meta={
                            **merged_metadata,
                            "content_type": "text",
                            "extraction_method": "multimodal_pypdf"
                        }
                    )
                    documents.append(text_document)
                    logger.debug(f"Created text document for '{source_info}'.")
                else:
                    logger.warning(
                        "No text content "
                        f"extracted from '{source_info}'."
                    )
                
                # Extract and describe images using pypdf
                image_documents = self._extract_and_describe_images(
                    pdf_reader, 
                    source_info,
                )
                
                # Add base metadata to image documents
                for img_doc in image_documents:
                    img_doc.meta.update(merged_metadata)
                
                documents.extend(image_documents)
                logger.info(f"Finished processing '{source_info}'.")

            except Exception as e:
                logger.error(f"Failed to process source: {source}. Error: {e}")
        
        logger.success(
            f"MultimodalPyPDFToDocument finished. "
            f"Generated a total of {len(documents)} documents."
        )
        return {"documents": documents}

@component
class StaticResponseGenerator:
    """
    A component that returns a static response for unsafe content,
    loading the response text from the config file.
    """
    
    def __init__(self):
        """
        Initializes the component by loading the static message 
        from config.
        """
        logger.debug("Initializing StaticResponseGenerator from config.")
        config = load_config()
        self.static_message = config["moderation"]["static_response"]
        self.streaming_callback = None
    
    @component.output_types(replies=List[ChatMessage])
    def run(self, messages: List[ChatMessage]) -> dict:
        """
        Generates a static response using the message loaded from 
        config.
        """
        logger.warning(
            "Content flagged as unsafe. Routing to static response generator.",
        )
        
        if self.streaming_callback:
            logger.debug("Streaming static response.")
            class StreamingChunk:
                def __init__(self, content):
                    self.content = content
            self.streaming_callback(StreamingChunk(self.static_message))
        
        reply_message = ChatMessage.from_assistant(self.static_message)
        logger.debug("Returning static safety response.")
        return {"replies": [reply_message]}
