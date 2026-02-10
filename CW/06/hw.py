from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions,
    VlmPipelineOptions
)
from docling.datamodel.pipeline_options_vlm_model import (
    ApiVlmOptions,ResponseFormat
)
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.document_converter import DocumentConverter, PdfFormatOption

SOURCE_PATH = "./cw06/sample_table.pdf"

def olmocr2_vlm_options(
    model: str = "allenai/olmOCR-2-7B-1025-FP8",
    hostname_and_port: str = "https://ws-01.wade0426.me/v1/",
    prompt: str = "Convert this page to markdown.",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    api_key: str = "",) -> ApiVlmOptions:


    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
   
    options = ApiVlmOptions(
        url=f"{hostname_and_port}chat/completions",
        params=dict(
            model=model,
            max_tokens=max_tokens,
        ),
        headers=headers,
        prompt=prompt,
        timeout=120,  # olmocr2 可能需要較長處理時間
        scale=2.0,  # 圖片縮放比例
        temperature=temperature,
        response_format=ResponseFormat.MARKDOWN,
    )
    return options



pineline_options = PdfPipelineOptions()
pineline_options.do_ocr = True


#============================
#==========RapidOCR==========
#============================
pineline_options.ocr_options = RapidOcrOptions(force_full_page_ocr=True)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pineline_options)
    }
)
result = converter.convert(SOURCE_PATH)
result.document.save_as_markdown(filename="./cw06/rapid_docling_sample.md")



#============================
#==========OLM OCR2==========
#============================
pineline_options = VlmPipelineOptions(enable_remote_services=True)
pineline_options.vlm_options = olmocr2_vlm_options()
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pineline_options,
            pipeline_cls=VlmPipeline
        )
    }
)
result = converter.convert(SOURCE_PATH)
result.document.save_as_markdown(filename="./cw06/olm_docling_sample.md")