"""Utilities for scanning raw documents for RGPD article 9 hints."""

from .scanner import scan_documents
from .vocabulary import load_vocabulary

__all__ = ["load_vocabulary", "scan_documents"]
