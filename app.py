import streamlit as st
import requests
import json
import time
from PyPDF2 import PdfReader
from io import BytesIO
from typing import Optional

# Configure Streamlit
st.set_page_config(page_title="Chat with PDF")

# Constants
GENAI_API_KEY = "AIzaSyAWeNKsOj_pSoqvbsMz1tkYkGEhsJLzgR8"  # Move this to environment variables in production
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key={GENAI_API_KEY}"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Function to extract text from uploaded PDFs
def extract_text_from_pdfs(pdf_files) -> str:
    text = ""
    for pdf in pdf_files:
        try:
            pdf_reader = PdfReader(BytesIO(pdf.read()))
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n\n"  # Add spacing between pages
        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {str(e)}")
    return text.strip()

# Function to query the Gemini API with rate limiting and retries
def query_gemini(system_prompt: str, user_prompt: str) -> Optional[str]:
    payload = {
        "prompt": {
            "context": system_prompt,
            "examples": [],
            "messages": [
                {"author": "0", "content": user_prompt}
            ]
        }
    }
    headers = {
        "Content-Type": "application/json"
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                API_URL, 
                headers=headers, 
                json=payload,  # Using json parameter for automatic serialization
                timeout=10  # Add timeout
            )
            
            # Handle rate limiting
            if response.status_code == 429:
                wait_time = int(response.headers.get('Retry-After', RETRY_DELAY))
                st.warning(f"Rate limit reached. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
                
            response.raise_for_status()
            response_data = response.json()
            return response_data["candidates"][0]["content"]
            
        except requests.exceptions.HTTPError as http_err:
            if attempt == MAX_RETRIES - 1:  # Last attempt
                st.error(f"HTTP error occurred: {http_err}")
            else:
                time.sleep(RETRY_DELAY)
                
        except requests.exceptions.RequestException as e:
            st.error(f"Error querying Gemini: {str(e)}")
            break
            
    return None

# Function to chunk text if it's too long
def chunk_text(text: str, chunk_size: int = 4000) -> list:
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def main():
    st.title("Chat with PDF")
    st.write("Upload PDFs, ask questions, and have a conversation!")

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Upload PDF Files")
        pdf_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
        
        if pdf_files:
            with st.spinner("Extracting text from PDFs..."):
                extracted_text = extract_text_from_pdfs(pdf_files)
                # Store text chunks instead of full text
                st.session_state["context_chunks"] = chunk_text(extracted_text)
                st.success("Text extraction complete!")

    # Initialize conversation history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display conversation history
    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # User input for chat
    user_input = st.chat_input("Type your message here:")
    
    if user_input:
        # Show user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Append user input to conversation history
        st.session_state["messages"].append({"role": "user", "content": user_input})
        
        # Process each chunk of context
        full_response = ""
        chunks = st.session_state.get("context_chunks", [""])
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                for chunk in chunks:
                    system_prompt = f"""
                    You are a helpful assistant. Use the provided context to answer questions accurately and concisely.
                    If the context does not contain the required information, respond with "The information is not available in the provided context."
                    Context: {chunk}
                    """
                    
                    response = query_gemini(system_prompt, user_input)
                    if response and "not available in the provided context" not in response:
                        full_response = response
                        break
                
                if not full_response:
                    full_response = "The information is not available in the provided context."
                
                st.write(full_response)
                st.session_state["messages"].append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
