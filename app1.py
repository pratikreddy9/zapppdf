import streamlit as st
import json
from PyPDF2 import PdfReader
from io import BytesIO
from openai import OpenAI
import time
from typing import Optional

# Configure Streamlit
st.set_page_config(page_title="Chat with PDF")

def extract_text_from_pdfs(pdf_files) -> str:
    text = ""
    for pdf in pdf_files:
        try:
            pdf_reader = PdfReader(BytesIO(pdf.read()))
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n\n"
        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {str(e)}")
    return text.strip()

def query_gpt4(system_prompt: str, user_prompt: str, gptkey: str, model="gpt-4") -> Optional[str]:
    try:
        client = OpenAI(api_key=gptkey)
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=model,
            temperature=0.7,
            max_tokens=1000
        ).choices[0]
        
        return chat_completion.message.content
        
    except Exception as e:
        st.error(f"Error querying GPT-4: {str(e)}")
        return None

def chunk_text(text: str, max_chunk_size: int = 6000) -> list:
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length > max_chunk_size:
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
            
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
    return chunks

def main():
    st.title("Chat with PDF")
    st.write("Upload PDFs and ask questions about their content!")

    # API Key input in sidebar
    with st.sidebar:
        st.title("Configuration")
        api_key = st.text_input("Enter your OpenAI API Key", type="password")
        st.title("Upload PDF Files")
        pdf_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
        
        if pdf_files:
            with st.spinner("Extracting text from PDFs..."):
                extracted_text = extract_text_from_pdfs(pdf_files)
                st.session_state["context_chunks"] = chunk_text(extracted_text)
                st.success("Text extraction complete!")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What would you like to know about the PDF?"):
        if not api_key:
            st.error("Please enter your OpenAI API key in the sidebar.")
            return
            
        # Append user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from GPT-4
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chunks = st.session_state.get("context_chunks", [""])
                best_response = None

                for chunk in chunks:
                    system_prompt = f"You are a helpful assistant. Answer questions based on the following context: {chunk}"
                    response = query_gpt4(system_prompt, prompt, api_key)
                    
                    if response and "information is not available" not in response.lower():
                        best_response = response
                        break
                
                if not best_response:
                    best_response = "I couldn't find relevant information in the provided documents. Could you please rephrase your question or ask about something else?"

                st.markdown(best_response)
                st.session_state.messages.append({"role": "assistant", "content": best_response})

if __name__ == "__main__":
    main()
