import pdfplumber, docling, markitdown
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
from markitdown import MarkItDown

source = "./day7/example.pdf"

#pdfplumber
with pdfplumber.open(source) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
with open("./day7/pdfplumber_md.md", "w", encoding='utf-8') as file:
    file.write(text)

#docling
converter = DocumentConverter()
result = converter.convert(source)
result.document.save_as_markdown(filename="./day7/docling_md.md")


#markitdown
md = MarkItDown(docintel_endpoint="<document_intelligence_endpoint>")
result = md.convert(source)
with open("./day7/mrk_md.md", "w", encoding='utf-8') as file:
    file.write(result.text_content)
