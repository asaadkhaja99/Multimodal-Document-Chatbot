import pytest
import requests
import json
import asyncio
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models.base_model import DeepEvalBaseLLM
from dotenv import load_dotenv

from src.pipelines.utils import load_config

# Load environment variables
load_dotenv()

# Create a Custom DeepEval Model
class OllamaDeepEvalModel(DeepEvalBaseLLM):
    def __init__(self, model: str, ollama_url: str = "http://localhost:11434"):
        import requests
        self.requests = requests
        self.model = model
        self.url = f"{ollama_url}/api/generate"
        
    def load_model(self):
        return self.model

    def _call(self, prompt: str) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        try:
            response = self.requests.post(self.url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except self.requests.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            return ""

    def generate(self, prompt: str) -> str:
        return self._call(prompt)

    async def a_generate(self, prompt: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._call, prompt)

    def get_model_name(self):
        return self.model

# Set up the Evaluation Model and API endpoint
config = load_config()
evaluation_model = OllamaDeepEvalModel(model=config["llm"]["main_generator"]["model"])
API_URL = "http://localhost:8000/query" # URL of RAG pipeline endpoint

# Define the Questions and Golden Answers
@pytest.mark.parametrize(
    "query, expected_output",
    [
    ],
)
def test_rag_pipeline_binary_grade(query: str, expected_output: str):
    # Call the RAG pipeline API to get the actual output
    print(f"\nTesting query: '{query}'")
    
    response = requests.post(API_URL, json={"query": query}, stream=True)
    response.raise_for_status()

    actual_output = ""
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith('data:'):
                json_data = json.loads(decoded_line[len('data:'):])
                actual_output += json_data.get('token', '')

    print(f"Actual output from API: {actual_output}")

    # Define the custom evaluation logic using GEval
    correctness_metric = GEval(
        name="Correctness",
        criteria="""
        Correctness - The 'actual output' should be factually correct and semantically similar to the 'expected output'.
        
        Instructions:
        1. Compare the 'actual output' with the 'expected output'.
        2. If the 'actual output' accurately and correctly answers the original question based on the information in the 'expected output', score it as 1.
        3. If the 'actual output' is incorrect, contains significant factual errors, or is not semantically similar, score it as 0.
        
        Output a single integer: 1 for correct, 0 for incorrect.
        """,
        model=evaluation_model,
        evaluation_params=[LLMTestCaseParams.EXPECTED_OUTPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    )
    
    # Define the test case for DeepEval
    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        expected_output=expected_output
    )

    # Run the assertion
    assert_test(test_case, [correctness_metric])


# --- How to run this script ---
# 1. Ensure your FastAPI application is running.
# 2. Run the evaluation from your terminal:
#    pytest deepeval_test.py
