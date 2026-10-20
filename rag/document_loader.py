"""Load PDF, TXT, and Markdown uploads."""
from io import BytesIO
from langchain_core.documents import Document
from pypdf import PdfReader

def extract_uploaded_documents(uploaded_files):
    documents = []
    skipped = []
    for uploaded_file in uploaded_files:
        name = uploaded_file.name
        raw = uploaded_file.getvalue()
        file_docs = []
        if name.lower().endswith(".pdf"):
            reader = PdfReader(BytesIO(raw))
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    file_docs.append(Document(page_content=text, metadata={"source": name, "page": page_num}))
        elif name.lower().endswith((".txt", ".md")):
            text = raw.decode("utf-8", errors="replace")
            if text.strip():
                file_docs.append(Document(page_content=text, metadata={"source": name, "page": None}))
        if file_docs:
            documents.extend(file_docs)
        else:
            skipped.append(name)
    return documents, skipped
