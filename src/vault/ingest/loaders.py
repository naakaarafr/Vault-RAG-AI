"""Document loaders supporting digital PDFs, DOCX, and scanned PDF OCR fallback."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExtractedPage:
    """Represent a single page of text extracted from a document."""

    page_number: int
    text: str


class DocumentLoader:
    """Base class for document loaders."""

    def load(self, file_path: str | Path) -> list[ExtractedPage]:
        raise NotImplementedError


class PDFLoader(DocumentLoader):
    """PDF loader using pdfplumber with automatic pytesseract OCR fallback for scanned pages."""

    def __init__(self, ocr_threshold_chars: int = 50) -> None:
        self.ocr_threshold_chars = ocr_threshold_chars

    def load(self, file_path: str | Path) -> list[ExtractedPage]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document file not found: {path}")

        extracted_pages: list[ExtractedPage] = []
        low_text_pages: list[int] = []

        # 1. Attempt digital text extraction via pdfplumber
        try:
            import pdfplumber

            with pdfplumber.open(path) as pdf:
                for idx, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    cleaned = text.strip()
                    extracted_pages.append(ExtractedPage(page_number=idx, text=cleaned))
                    if len(cleaned) < self.ocr_threshold_chars:
                        low_text_pages.append(idx)
        except Exception:
            # If digital extraction fails completely, mark all for fallback
            low_text_pages = [1]
            extracted_pages = []

        # 2. If pages have near-empty text, execute OCR fallback
        if low_text_pages:
            ocr_pages = self._ocr_fallback(path, target_pages=low_text_pages)
            ocr_dict = {p.page_number: p.text for p in ocr_pages}

            if not extracted_pages:
                extracted_pages = ocr_pages
            else:
                for i, page in enumerate(extracted_pages):
                    if page.page_number in ocr_dict and ocr_dict[page.page_number].strip():
                        extracted_pages[i] = ExtractedPage(
                            page_number=page.page_number,
                            text=ocr_dict[page.page_number],
                        )

        return extracted_pages

    def _ocr_fallback(self, file_path: Path, target_pages: list[int]) -> list[ExtractedPage]:
        """OCR fallback using pypdfium2 to render page image and pytesseract to extract text."""
        results: list[ExtractedPage] = []
        try:
            import pypdfium2 as pdfium
            import pytesseract

            pdf = pdfium.PdfDocument(file_path)
            for page_num in target_pages:
                if page_num < 1 or page_num > len(pdf):
                    continue
                page = pdf[page_num - 1]
                # Render page to PIL image at 300 DPI
                image = page.render(scale=300 / 72).to_pil()
                text = pytesseract.image_to_string(image) or ""
                results.append(ExtractedPage(page_number=page_num, text=text.strip()))
        except Exception:
            # Fallback gracefully if tesseract binary or pypdfium2 is unavailable
            for page_num in target_pages:
                results.append(
                    ExtractedPage(
                        page_number=page_num,
                        text="[OCR unavailable: Scanned page text fallback]",
                    )
                )

        return results


class DocxLoader(DocumentLoader):
    """DOCX document loader using python-docx."""

    def load(self, file_path: str | Path) -> list[ExtractedPage]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document file not found: {path}")

        try:
            import docx

            doc = docx.Document(path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            full_text = "\n\n".join(paragraphs)
            return [ExtractedPage(page_number=1, text=full_text)]
        except Exception as exc:
            raise RuntimeError(f"Failed to load DOCX file '{path}': {exc}") from exc
