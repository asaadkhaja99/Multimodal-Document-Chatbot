import json

import requests
import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="Haystack RAG Frontend",
    page_icon="🤖",
    layout="wide"
)

# --- UI Elements ---
st.title("🤖 Haystack RAG Frontend")
st.write("Ask a question and get a streaming response from the Haystack RAG pipeline.")

# The backend API endpoint
API_URL = "http://localhost:8000/query"

# User input
query = st.text_input("Enter your question:", placeholder="e.g., What is the capital of France?")

# --- API Interaction and Streaming ---
if st.button("Ask"):
    if query:
        st.subheader("Answer:")
        
        # Use a placeholder for the streaming text
        response_placeholder = st.empty()
        full_response = ""

        try:
            # Prepare the request payload
            payload = {"query": query}
            
            # Make the POST request with streaming enabled
            with requests.post(API_URL, json=payload, stream=True) as response:
                # Check for a successful response
                if response.status_code == 200:
                    # Iterate over the response chunks
                    for line in response.iter_lines():
                        if line:
                            # SSE lines start with "data: "
                            decoded_line = line.decode('utf-8')
                            if decoded_line.startswith('data:'):
                                # Extract the JSON part
                                json_str = decoded_line[len('data:'):].strip()
                                
                                try:
                                    # Parse the JSON and extract the token
                                    data = json.loads(json_str)
                                    token = data.get('token', '')
                                    
                                    # Append the token to the full response and update the placeholder
                                    full_response += token
                                    response_placeholder.markdown(full_response)
                                except json.JSONDecodeError:
                                    st.error("Error decoding a stream chunk.")
                                    break
                else:
                    st.error(f"Error: Received status code {response.status_code}")
                    st.text(f"Response body: {response.text}")

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to the API. Please ensure the backend is running at {API_URL}.")
            st.error(f"Error details: {e}")
    else:
        st.warning("Please enter a question.")
