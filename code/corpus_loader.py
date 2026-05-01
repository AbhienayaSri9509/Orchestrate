"""
Corpus Loader & TF-IDF Retriever (pure Python — no sklearn/numpy required)

Loads all markdown files from data/{hackerrank,claude,visa}/, chunks them,
and builds a TF-IDF index for retrieval. Provides a retrieve() function
that returns the most relevant corpus chunks for a given query.
"""

import re
import math
from pathlib import Path
from collections import Counter
from typing import List, Dict, Optional, Tuple

from config import (
    CORPUS_DIRS, RETRIEVAL_TOP_K, CHUNK_SIZE, CHUNK_OVERLAP,
)

# English stop words (built-in, no nltk needed)
STOP_WORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "shall","can","need","this","that","these","those","i","you","he","she",
    "it","we","they","what","which","who","whom","how","when","where","why",
    "all","each","every","both","few","more","most","other","some","such",
    "no","not","only","same","so","than","too","very","just","because","as",
    "until","while","about","against","between","into","through","during",
    "before","after","above","below","up","down","out","off","over","under",
    "again","then","once","here","there","my","your","his","her","its","our",
    "their","if","our","s","t","re","ve","ll","d","m",
}


class CorpusChunk:
    """Represents a chunk of corpus text with metadata."""

    def __init__(self, text: str, source_company: str, file_path: str, title: str):
        self.text = text
        self.source_company = source_company
        self.file_path = file_path
        self.title = title

    def __repr__(self):
        return f"CorpusChunk(company={self.source_company}, title={self.title[:50]})"


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, remove stop words."""
    tokens = re.findall(r"[a-z][a-z0-9']*", text.lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


class CorpusRetriever:
    """
    Loads the support corpus, chunks documents, builds a TF-IDF index,
    and retrieves relevant chunks for queries — using only Python stdlib.
    """

    def __init__(self):
        self.chunks: List[CorpusChunk] = []
        # Per-chunk token frequency lists
        self._chunk_tfs: List[Dict[str, float]] = []
        # Document frequency per term
        self._df: Dict[str, int] = {}
        self._num_docs: int = 0
        self._load_corpus()
        self._build_index()

    # ── Corpus loading ────────────────────────────────────────────────────────

    def _strip_frontmatter(self, text: str) -> str:
        if text.startswith("---"):
            end = text.find("---", 3)
            if end != -1:
                text = text[end + 3:].strip()
        return text

    def _extract_title(self, text: str, file_path: str) -> str:
        match = re.search(r'^#\s+(.+)$', text, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return Path(file_path).stem.replace("-", " ").title()

    def _chunk_text(self, text: str) -> List[str]:
        words = text.split()
        if len(words) <= CHUNK_SIZE:
            return [text]
        chunks, start = [], 0
        while start < len(words):
            end = start + CHUNK_SIZE
            chunks.append(" ".join(words[start:end]))
            start = end - CHUNK_OVERLAP
            if start >= len(words):
                break
        return chunks

    def _load_corpus(self):
        total_files = 0
        for company, corpus_dir in CORPUS_DIRS.items():
            if not corpus_dir.exists():
                print(f"  [WARN] Corpus directory not found: {corpus_dir}")
                continue
            for md_file in corpus_dir.rglob("*.md"):
                try:
                    raw = md_file.read_text(encoding="utf-8", errors="ignore")
                    text = self._strip_frontmatter(raw)
                    title = self._extract_title(text, str(md_file))
                    # Clean markdown
                    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
                    text = re.sub(r'!\[[^\]]*\]\([^\)]+\)', '', text)
                    text = re.sub(r'\|[-:]+\|', '', text)
                    text = re.sub(r'\n{3,}', '\n\n', text)
                    for chunk_text in self._chunk_text(text):
                        if len(chunk_text.strip()) > 50:
                            self.chunks.append(CorpusChunk(
                                text=chunk_text.strip(),
                                source_company=company,
                                file_path=str(md_file.relative_to(corpus_dir.parent.parent)),
                                title=title,
                            ))
                    total_files += 1
                except Exception as e:
                    print(f"  [WARN] Failed to load {md_file}: {e}")
        print(f"  Loaded {total_files} files → {len(self.chunks)} chunks")

    # ── Index building ────────────────────────────────────────────────────────

    def _build_index(self):
        """Build TF-IDF vectors for all chunks using pure Python."""
        if not self.chunks:
            raise ValueError("No corpus chunks loaded!")

        self._num_docs = len(self.chunks)

        # Compute term frequencies for each chunk
        for chunk in self.chunks:
            tokens = _tokenize(chunk.text)
            counts = Counter(tokens)
            total = max(len(tokens), 1)
            # Sublinear TF: 1 + log(tf)
            tf = {t: 1 + math.log(c / total + 1) for t, c in counts.items()}
            self._chunk_tfs.append(tf)
            for term in counts:
                self._df[term] = self._df.get(term, 0) + 1

        print(f"  TF-IDF index built: {self._num_docs} chunks, {len(self._df)} unique terms")

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log((self._num_docs + 1) / (df + 1)) + 1  # smoothed IDF

    def _tfidf_vec(self, tf_dict: Dict[str, float]) -> Dict[str, float]:
        return {t: tf * self._idf(t) for t, tf in tf_dict.items()}

    def _cosine(self, vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        """Cosine similarity between two sparse TF-IDF vectors."""
        dot = sum(vec_a.get(t, 0) * v for t, v in vec_b.items())
        norm_a = math.sqrt(sum(v * v for v in vec_a.values())) or 1e-9
        norm_b = math.sqrt(sum(v * v for v in vec_b.values())) or 1e-9
        return dot / (norm_a * norm_b)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        company_hint: Optional[str] = None,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> List[Tuple[CorpusChunk, float]]:
        """
        Retrieve the most relevant corpus chunks for a query.
        Uses cosine similarity over TF-IDF vectors (pure Python).
        """
        # Build query vector
        q_tokens = _tokenize(query)
        q_counts = Counter(q_tokens)
        q_total = max(len(q_tokens), 1)
        q_tf = {t: 1 + math.log(c / q_total + 1) for t, c in q_counts.items()}
        q_vec = self._tfidf_vec(q_tf)

        if not q_vec:
            return []

        # Score all chunks
        scores = []
        for i, tf_dict in enumerate(self._chunk_tfs):
            chunk_vec = self._tfidf_vec(tf_dict)
            score = self._cosine(q_vec, chunk_vec)

            # 20% boost for matching company
            if company_hint and self.chunks[i].source_company == company_hint.lower():
                score *= 1.2

            scores.append((i, score))

        # Sort and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scores[:top_k]:
            if score > 0.01:
                results.append((self.chunks[idx], score))

        return results

    def get_top_score(self, query: str, company_hint: Optional[str] = None) -> float:
        results = self.retrieve(query, company_hint, top_k=1)
        return results[0][1] if results else 0.0
