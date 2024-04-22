import io

import requests
import streamlit as st
from pdfminer.converter import TextConverter
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage


def load_pdf_from_url(pdf_url):
    response = requests.get(pdf_url)
    response.raise_for_status()
    with io.BytesIO(response.content) as open_pdf_file:
        rsrcmgr = PDFResourceManager()
        retstr = io.StringIO()
        laparams = LAParams()
        device = TextConverter(rsrcmgr, retstr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.get_pages(open_pdf_file):
            interpreter.process_page(page)
            text = retstr.getvalue()
        device.close()
        retstr.close()
    return text


def main():
    st.set_page_config(layout="wide")
    st.title("PDF Interpreter")

    option = st.radio(
        "Choose an option:", ("Enter a PDF URL", "Upload a PDF file")
    )
    if option == "Enter a PDF URL":
        pdf_url = st.text_input("Enter PDF URL here.")
        if pdf_url:
            pdf_text = load_pdf_from_url(pdf_url)
    elif option == "Upload a PDF file":
        pdf_file = st.file_uploader("Upload PDF here.", type=["pdf"])
        if pdf_file:
            pdf_text = extract_text(pdf_file)


if __name__ == "__main__":
    main()
