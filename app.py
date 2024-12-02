import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai
from io import BytesIO

# Configure Streamlit
st.set_page_config(page_title="Simple Conversational Bot")

# Google Gemini API Key (directly within the script)
GENAI_API_KEY = "AIzaSyAWeNKsOj_pSoqvbsMz1tkYkGEhsJLzgR8"
genai.configure(api_key=GENAI_API_KEY)

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

# Function to query the Gemini LLM
def query_gemini(messages):
    try:
        response = genai.chat(messages=messages, model="gemini-1.5-flash", temperature=0.3)
        return response.messages[-1]["content"]  # Return the latest response from the assistant
    except Exception as e:
        st.error(f"Error querying Gemini: {e}")
        return None

# Main function
def main():
    st.title("Simple Conversational Bot")
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
        st.session_state["messages"] = [
            {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions."}
        ]

    # Display previous conversation
    if "messages" in st.session_state:
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
        if "context" in st.session_state and st.session_state["context"]:
            context_message = {"role": "system", "content": f"Context: {st.session_state['context']}"}
            st.session_state["messages"].insert(1, context_message)
        
        # Query Gemini with the conversation history
        response = query_gemini(st.session_state["messages"])
        if response:
            st.session_state["messages"].append({"role": "assistant", "content": response})
            st.experimental_rerun()  # Refresh to display the updated conversation

if __name__ == "__main__":
    main()
