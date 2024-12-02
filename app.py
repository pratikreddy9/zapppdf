import streamlit as st
import requests
import json
from PyPDF2 import PdfReader
from io import BytesIO

# Configure Streamlit
st.set_page_config(page_title="Chat with PDF")

# Google Gemini API Key (directly within the script)
GENAI_API_KEY = "YOUR_API_KEY"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={GENAI_API_KEY}"

# Function to extract text from uploaded PDFs
def extract_text_from_pdfs(pdf_files):
    text = ""
    for pdf in pdf_files:
        try:
            pdf_reader = PdfReader(BytesIO(pdf.read()))
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                text += page_text
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    return text

# Function to query the Gemini API
def query_gemini(system_prompt, user_prompt):
    payload = json.dumps({
        "prompt": {
            "context": system_prompt,
            "examples": [],
            "messages": [
                {"author": "0", "content": user_prompt}
            ]
        }
    })
    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(API_URL, headers=headers, data=payload)
        response.raise_for_status()  # Raise an error for bad status codes
        response_data = response.json()
        return response_data["candidates"][0]["content"]
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err}")
    except Exception as e:
        st.error(f"Error querying Gemini: {e}")
    return None

# Main function
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
                st.session_state["context"] = extracted_text
                st.success("Text extraction complete!")

    # Initialize conversation history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display previous conversation
    if st.session_state.get("messages"):
        for msg in st.session_state["messages"]:
            if msg["role"] == "user":
                st.markdown(f"<div style='text-align: right; color: blue;'>**You:** {msg['content']}</div>", unsafe_allow_html=True)
            elif msg["role"] == "assistant":
                st.markdown(f"<div style='text-align: left; color: green;'>**Bot:** {msg['content']}</div>", unsafe_allow_html=True)

    # User input for chat
    user_input = st.text_input("Type your message here:")
    if user_input:
        # Append user input to conversation history
        st.session_state["messages"].append({"role": "user", "content": user_input})
        
        # Include context from PDFs if available
        context = st.session_state.get("context", "")
        system_prompt = f"""
        You are a helpful assistant. Use the provided context to answer questions accurately and concisely.
        If the context does not contain the required information, respond with "The information is not available in the provided context."
        Context: {context}
        """

        # Query Gemini with the conversation history
        response = query_gemini(system_prompt, user_input)
        if response:
            st.session_state["messages"].append({"role": "assistant", "content": response})
            st.experimental_rerun()  # Refresh to display the updated conversation

if __name__ == "__main__":
    main()
