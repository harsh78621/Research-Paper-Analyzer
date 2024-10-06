import os
import re
import spacy
import fitz  # PyMuPDF
import camelot
from pdf2image import convert_from_path
from pytesseract import image_to_string
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
import io
import streamlit as st

# Load environment variables
os.environ['GENAI_API_KEY'] = 'YOUR_GENAI_API_KEY'
genai.configure(api_key=os.environ['GENAI_API_KEY'])

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

class ResearchPaperAnalyzer:
    def __init__(self, model_name: str):
        self.model = genai.GenerativeModel(model_name=model_name)
        self.image_descriptions = {}
        self.tables_data = {}
        self.extracted_text = ""

    def extract_text(self, pdf_path):
        pages = convert_from_path(pdf_path)
        text = ""
        for page in pages:
            text += image_to_string(page)
        return text

    def segment_text(self, text):
        heading_pattern = r"^\d+\.\s.+"
        paragraphs = text.split("\n\n")
        sections = []
        for para in paragraphs:
            para = para.strip()
            if re.match(heading_pattern, para):
                sections.append({"type": "heading", "content": para})
            else:
                sections.append({"type": "paragraph", "content": para})
        return sections

    def rebuild_structure(self, sections):
        formatted_text = ""
        for section in sections:
            if section['type'] == 'heading':
                formatted_text += f"\n\n### {section['content']} ###\n\n"
            else:
                formatted_text += f"\n{section['content']}\n"
        return formatted_text

    def analyze_text_with_nlp(self, text):
        doc = nlp(text)
        tokens = [token.text for token in doc]
        pos_tags = [(token.text, token.pos_) for token in doc]
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return tokens, pos_tags, entities

    def process_text(self, text):
        sections = self.segment_text(text)
        structured_text = self.rebuild_structure(sections)

        analyzed_data = []
        for section in sections:
            tokens, pos_tags, entities = self.analyze_text_with_nlp(section['content'])
            analyzed_data.append({
                "section_type": section['type'],
                "content": section['content'],
                "tokens": tokens,
                "pos_tags": pos_tags,
                "entities": entities
            })

        return structured_text, analyzed_data

    def extract_tables(self, pdf_path):
        tables = camelot.read_pdf(pdf_path, pages='all')
        return tables

    def process_paper(self, pdf_path):
        self.extracted_text = self.extract_text(pdf_path)
        structured_text, _ = self.process_text(self.extracted_text)

        tables = self.extract_tables(pdf_path)
        for i, table in enumerate(tables):
            csv_data = table.df.to_csv(index=False)
            self.tables_data[f"table_{i + 1}"] = csv_data

    def handle_query(self, query):
        combined_info = f"{self.extracted_text}\n\nImage Descriptions:\n{self.image_descriptions}\n\nTables Data:\n{self.tables_data}"
        
        prompt = f"Based on the following information, answer the query:\n\n{combined_info}\n\nQuery: {query}"
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"An error occurred: {str(e)}"

# Streamlit UI setup
st.title("Research Paper Analyzer")
st.write("Upload a PDF document to analyze and ask questions about it.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    analyzer = ResearchPaperAnalyzer("gemini-1.5-pro-latest")
    
    # Process the uploaded PDF file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    analyzer.process_paper("temp.pdf")
    
    # User query input
    user_query = st.text_input("Ask a question about the document:")
    
    if st.button("Submit"):
        if user_query:
            response = analyzer.handle_query(user_query)
            st.write("Response:", response)
