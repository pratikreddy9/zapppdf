import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai
from io import BytesIO

# Configure Streamlit
st.set_page_config(page_title="Chat With PDFs")

# Your Google API Key (directly in the script)
GENAI_API_KEY = "AIzaSyAWeNKsOj_pSoqvbsMz1tkYkGEhsJLzgR8"
genai.configure(api_key=GENAI_API_KEY)

# Function to extract text from PDFs
def getPdfText(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(BytesIO(pdf.read()))
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                st.write(f"Extracted text: {page_text[:500]}")  # Show the first 500 characters for inspection
                text += page_text
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    return text

# Function to query the LLM with extracted text
def query_llm_with_text(extracted_text, user_question):
    prompt = f"""
You are an intelligent assistant. Use the provided context to answer the user's question as accurately as possible. 
If the context does not contain the answer, respond with: "The answer is not available in the provided context." Do not guess or fabricate information.

Context:
{extracted_text}

Question:
{user_question}

Answer:
"""
    try:
        response = genai.generate_text(model="gemini-1.5-flash", prompt=prompt, temperature=0.3)
        return response.result
    except Exception as e:
        st.error(f"Error querying the LLM: {e}")
        return None

# Main function
def main():
    st.title("Chat with PDF")
    st.header("Upload your PDFs and ask questions")

    user_question = st.text_input("Ask a question based on the uploaded PDF(s)")
    pdf_docs = st.file_uploader("Upload your PDF files", type="pdf", accept_multiple_files=True)

    if st.button("Submit & Process"):
        if pdf_docs and user_question:
            with st.spinner("Processing..."):
                try:
                    raw_text = getPdfText(pdf_docs)
                    if raw_text.strip():
                        st.write("Querying the LLM with extracted text...")
                        llm_response = query_llm_with_text(raw_text, user_question)
                        if llm_response:
                            st.write(f"Answer: {llm_response}")
                        else:
                            st.error("No response from the LLM.")
                    else:
                        st.error("No text could be extracted from the uploaded PDF(s).")
                except Exception as e:
                    st.error(f"Processing error: {e}")
        else:
            st.error("Please upload PDF files and enter a question.")

if __name__ == "__main__":
    main()
