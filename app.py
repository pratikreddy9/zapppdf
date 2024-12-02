import streamlit as st
import requests
import json
import os
import pandas as pd
from PyPDF2 import PdfReader
from io import BytesIO
import base64

# Set up the page
st.set_page_config(page_title="Conversational Bot", layout="wide")

# Google Gemini API Key (Directly in Script)
GENAI_API_KEY = "AIzaSyAWeNKsOj_pSoqvbsMz1tkYkGEhsJLzgR8"

# Google Sheets URL and worksheet ID
spreadsheet_url = "https://docs.google.com/spreadsheets/d/1wgliY7XyZF-p4FUa1MiELUlQ3v1Tg6KDZzWuyW8AMo4/edit?gid=835818411"
worksheet_id = "835818411"

# Function to read Google Sheets data
@st.cache_data
def get_data_from_gsheet(url, worksheet_id):
    try:
        st.write(f"Reading from Google Sheets URL: {url} and Worksheet ID: {worksheet_id}")
        # Fetch data as a Pandas DataFrame
        data = pd.read_csv(url)
        return data
    except Exception as e:
        st.error(f"Error reading from Google Sheets: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error

data = get_data_from_gsheet(spreadsheet_url, worksheet_id)

# Construct the initial system message
initial_system_message = """
You are a conversational assistant providing detailed information. Use only the provided context.
Do not fabricate or guess information. Always respond professionally and clearly.
Product Information:
"""

if not data.empty:
    for _, row in data.iterrows():
        initial_system_message += f"Product: {row['Product Name']}, Definition: {row['Definition']}, Material: {row['Material']}, HS Code: {row['HS Code']}, Specifications: {row['Specifications']}\n"

# Extract text from uploaded PDFs
def extract_text_from_pdfs(uploaded_files):
    text = ""
    for pdf in uploaded_files:
        try:
            pdf_reader = PdfReader(BytesIO(pdf.read()))
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    return text

# Function to query the Google Gemini API
def query_gemini(messages):
    url = "https://generativelanguage.googleapis.com/v1beta2/models/gemini:generateText"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GENAI_API_KEY}",
    }
    payload = {
        "model": "gemini-1.5-flash",
        "messages": messages,
        "temperature": 0.3,
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response_data = response.json()
        return response_data.get("candidates", [{}])[0].get("output", "")
    except Exception as e:
        st.error(f"Error querying Gemini: {e}")
        return None

# Handle user input and generate a response
def handle_user_input(context, user_message):
    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": user_message},
    ]
    return query_gemini(messages)

# Main application
def main():
    st.title("Conversational Bot")
    st.write("Upload PDFs, ask questions, and receive contextual responses.")

    # File upload for PDFs
    pdf_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    pdf_text = extract_text_from_pdfs(pdf_files) if pdf_files else ""

    # Maintain chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "system", "content": initial_system_message}]

    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        elif message["role"] == "assistant":
            st.markdown(f"**Assistant:** {message['content']}")

    # User input
    user_message = st.text_input("Your message:")
    if st.button("Send"):
        if user_message:
            # Add user's message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_message})

            # Generate response
            context = initial_system_message + "\n" + pdf_text
            bot_response = handle_user_input(context, user_message)

            if bot_response:
                # Add assistant's response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
                st.experimental_rerun()

if __name__ == "__main__":
    main()
