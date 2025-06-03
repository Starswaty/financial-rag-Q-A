import fitz  # PyMuPDF
import docx

def parse_document(uploaded_file):
    file_type = uploaded_file.filename.split(".")[-1].lower()
    content = uploaded_file.file.read()
    if file_type == "pdf":
        doc = fitz.open(stream=content, filetype="pdf")
        return "\n".join([page.get_text("text") for page in doc])
    elif file_type == "docx":
        from io import BytesIO
        doc = docx.Document(BytesIO(content))
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    elif file_type == "txt":
        return content.decode("utf-8")
    else:
        raise ValueError("Unsupported file type")