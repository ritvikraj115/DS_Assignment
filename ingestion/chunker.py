from __future__ import annotations

import hashlib
import re
import uuid

from .models import Chunk, ContentBlock


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class StructureAwareChunker:
    def __init__(self, chunk_size: int = 1000, overlap: int = 150):
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("chunk_size must be positive and overlap must be in [0, chunk_size)")
        self.chunk_size, self.overlap = chunk_size, overlap

    def _split_long(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]
        # Split prose on sentence/whitespace boundaries; keep fenced code intact when possible.
        if text.lstrip().startswith("```"):
            return [text]
        parts = re.split(r"(?<=[.!?])\s+|\n", text)
        result: list[str] = []
        current = ""
        for part in parts:
            if len(part) > self.chunk_size:
                if current:
                    result.append(current)
                    current = ""
                start = 0
                step = max(1, self.chunk_size - self.overlap)
                while start < len(part):
                    result.append(part[start : start + self.chunk_size])
                    start += step
                continue
            candidate = f"{current}\n{part}".strip() if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current)
                carry = current[-self.overlap :] if self.overlap else ""
                current = f"{carry}\n{part}".strip()
        if current:
            result.append(current)
        return result

    def chunk(self, blocks: list[ContentBlock], document_id: str) -> list[Chunk]:
        grouped: list[tuple[dict, list[str]]] = []
        for block in blocks:
            section = {"title": block.section.title, "level": block.section.level, "path": block.section.path}
            if not grouped or grouped[-1][0] != section:
                grouped.append((section, []))
            grouped[-1][1].append(block.text)
        chunks: list[tuple[str, dict]] = []
        occurrences: dict[str, int] = {}
        for section, texts in grouped:
            current = ""
            for text in texts:
                candidate = f"{current}\n\n{text}".strip() if current else text
                # Keep tables/code blocks together even when they exceed the target size.
                atomic = text.lstrip().startswith("```") or (text.lstrip().startswith("|") and "\n|" in text)
                if len(candidate) <= self.chunk_size:
                    current = candidate
                    continue
                if current:
                    chunks.append((current, section))
                    carry = current[-self.overlap :] if self.overlap else ""
                else:
                    carry = ""
                if len(text) > self.chunk_size and not atomic:
                    split_source = f"{carry}\n\n{text}".strip() if carry else text
                    chunks.extend((part, section) for part in self._split_long(split_source))
                    current = ""
                else:
                    current = f"{carry}\n\n{text}".strip() if carry else text
            if current:
                chunks.append((current, section))
        result: list[Chunk] = []
        for text, section in chunks:
            digest = sha256_text(text)
            occurrence = occurrences.get(digest, 0)
            occurrences[digest] = occurrence + 1
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{digest}:{occurrence}"))
            result.append(Chunk(chunk_id, text, section, digest))
        return result
