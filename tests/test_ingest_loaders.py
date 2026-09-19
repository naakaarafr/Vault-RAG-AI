"""Unit tests for document loaders and scanned PDF OCR fallback."""

from pathlib import Path

import pytest

from vault.ingest.loaders import ExtractedPage, PDFLoader


def test_pdf_loader_ocr_fallback_triggered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify scanned PDF with near-empty text triggers OCR fallback method."""
    loader = PDFLoader(ocr_threshold_chars=50)
    dummy_pdf_path = tmp_path / "scanned_doc.pdf"
    dummy_pdf_path.write_bytes(b"%PDF-1.4 scanned content dummy bytes")

    # Mock pdfplumber returning near-empty text
    class MockPDFPage:
        def extract_text(self):
            return "   "  # 3 spaces (near-empty)

    class MockPDF:
        pages = [MockPDFPage()]
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    import pdfplumber
    monkeypatch.setattr(pdfplumber, "open", lambda path: MockPDF())

    # Mock _ocr_fallback method
    def mock_ocr(file_path: Path, target_pages: list[int]):
        return [ExtractedPage(page_number=1, text="OCR Extracted Text from Scanned Image")]

    monkeypatch.setattr(loader, "_ocr_fallback", mock_ocr)

    pages = loader.load(dummy_pdf_path)
    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert "OCR Extracted Text" in pages[0].text
