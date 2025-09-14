import pytest
import requests
import json
import asyncio
from deepeval import assert_test
from deepeval.metrics.g_eval.g_eval import GEval
from deepeval.test_case.llm_test_case import LLMTestCase, LLMTestCaseParams
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
        # Add JSON formatting instructions to ensure proper output for DeepEval GEval
        json_prompt = f"""{prompt}

Please respond with ONLY a valid JSON object in the following format:
{{
    "score": <integer 0 or 1>,
    "reason": "<brief explanation>",
    "steps": [
        {{
            "description": "<step description>",
            "score": <integer 0 or 1>
        }}
    ]
}}

Do not include any other text or formatting outside the JSON object."""

        payload = {
            "model": self.model,
            "prompt": json_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9
            }
        }

        try:
            response = self.requests.post(self.url, json=payload)
            response.raise_for_status()
            result = response.json()
            response_text = result.get("response", "").strip()

            # Try to extract JSON if the response contains extra text
            import re
            json_match = re.search(r'\{[^}]*"score"\s*:\s*[01].*?"steps"[^}]*\}', response_text, re.DOTALL)
            if json_match:
                return json_match.group()

            # Fallback: create proper format if response only has score and reason
            simple_json_match = re.search(r'\{[^}]*"score"\s*:\s*([01])[^}]*"reason"\s*:\s*"([^"]*)"[^}]*\}', response_text)
            if simple_json_match:
                score = simple_json_match.group(1)
                reason = simple_json_match.group(2)
                return f'{{"score": {score}, "reason": "{reason}", "steps": [{{"description": "{reason}", "score": {score}}}]}}'

            return response_text
        except self.requests.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            return '{"score": 0, "reason": "API error", "steps": [{"description": "API error occurred", "score": 0}]}'

    def generate(self, prompt: str) -> str:
        return self._call(prompt)

    async def a_generate(self, prompt: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._call, prompt)

    def get_model_name(self):
        return self.model

# Set up the Evaluation Model and API endpoint
config = load_config()
evaluation_model = OllamaDeepEvalModel(
    model=config["llm"]["evaluation_generator"]["model"],
    ollama_url=config["llm"]["evaluation_generator"]["url"]
)
API_URL = "http://localhost:8000/query" # URL of RAG pipeline endpoint

# Load test cases from config
test_cases = [(case["query"], case["expected_output"]) for case in config["evaluation"]["test_cases"]]

# Define the Questions and Golden Answers for "Attention Is All You Need" paper
@pytest.mark.parametrize(
    "query, expected_output",
    test_cases,
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
        Evaluate the correctness of the actual output compared to the expected output.

        Instructions:
        1. Compare the 'actual output' with the 'expected output' for factual accuracy.
        2. If the actual output correctly answers the question and contains the key facts from the expected output, assign score 1.
        3. If the actual output is incorrect, contains factual errors, is incomplete, or missing key information, assign score 0.
        4. This is a strict binary evaluation - NO partial credit. Either the answer is correct (1) or incorrect (0).

        You must respond with a valid JSON object containing:
        - "score": integer (0 or 1)
        - "reason": string explaining your decision
        - "steps": array with one step containing description and score
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
