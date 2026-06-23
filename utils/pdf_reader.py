"""PDF text extraction utilities for Campus Digital Twin AI."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class PDFReaderError(Exception):
    """Base exception for PDF reader failures."""


class InvalidPDFError(PDFReaderError):
    """Raised when a PDF is missing, unreadable, encrypted, or corrupted."""


def extract_text_from_pdf(uploaded_pdf: str | Path | bytes | bytearray | BinaryIO) -> str:
    """Extract plain text from an uploaded PDF.

    Args:
        uploaded_pdf: A Streamlit uploaded file, file-like binary object, bytes,
            bytearray, or filesystem path.

    Returns:
        Extracted plain text with page text separated by blank lines.

    Raises:
        InvalidPDFError: If the file is missing, corrupted, encrypted, or unreadable.
    """
    pdf_stream = _to_binary_stream(uploaded_pdf)

    try:
        reader = PdfReader(pdf_stream)
    except (PdfReadError, OSError, ValueError, TypeError) as exc:
        raise InvalidPDFError("Unable to read PDF. The file may be corrupted or invalid.") from exc

    if reader.is_encrypted:
        try:
            decrypt_result = reader.decrypt("")
        except Exception as exc:
            raise InvalidPDFError("Unable to read encrypted PDF.") from exc

        if decrypt_result == 0:
            raise InvalidPDFError("Unable to read encrypted PDF.")

    page_text: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            raise InvalidPDFError(f"Unable to extract text from page {page_number}.") from exc

        cleaned_text = _clean_text(text)
        if cleaned_text:
            page_text.append(cleaned_text)

    return "\n\n".join(page_text).strip()


def _to_binary_stream(uploaded_pdf: str | Path | bytes | bytearray | BinaryIO) -> BytesIO | BinaryIO:
    if isinstance(uploaded_pdf, (str, Path)):
        path = Path(uploaded_pdf)
        if not path.exists() or not path.is_file():
            raise InvalidPDFError("PDF file does not exist.")
        try:
            return BytesIO(path.read_bytes())
        except OSError as exc:
            raise InvalidPDFError("Unable to open PDF file.") from exc

    if isinstance(uploaded_pdf, (bytes, bytearray)):
        if not uploaded_pdf:
            raise InvalidPDFError("PDF file is empty.")
        return BytesIO(uploaded_pdf)

    getvalue = getattr(uploaded_pdf, "getvalue", None)
    if callable(getvalue):
        data = getvalue()
        if not data:
            raise InvalidPDFError("PDF file is empty.")
        return BytesIO(data)

    read = getattr(uploaded_pdf, "read", None)
    seek = getattr(uploaded_pdf, "seek", None)
    if callable(read):
        try:
            if callable(seek):
                seek(0)
            data = read()
            if callable(seek):
                seek(0)
        except OSError as exc:
            raise InvalidPDFError("Unable to read uploaded PDF.") from exc

        if not data:
            raise InvalidPDFError("PDF file is empty.")
        return BytesIO(data)

    raise InvalidPDFError("Unsupported PDF input type.")


def _clean_text(text: str) -> str:
    lines = (" ".join(line.split()) for line in text.splitlines())
    return "\n".join(line for line in lines if line)
