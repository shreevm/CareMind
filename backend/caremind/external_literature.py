import logging
import re
import xml.etree.ElementTree as ET
from html import unescape

import httpx

from .config import Settings
from .schemas import RetrievedChunk

logger = logging.getLogger(__name__)


class ExternalLiteratureSearch:
    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def enabled(self) -> bool:
        return self.settings.external_literature_search_enabled

    @property
    def medlineplus_enabled(self) -> bool:
        return self.settings.medlineplus_health_topics_enabled

    def search_pubmed(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        if not self.enabled:
            logger.info("external.pubmed.skip reason=disabled")
            return []
        safe_query = self.sanitize_query(query)
        if not safe_query:
            logger.info("external.pubmed.skip reason=empty_safe_query")
            return []
        limit = max(1, min(top_k or self.settings.external_search_max_results, 10))
        logger.info("external.pubmed.search.start query=%s top_k=%s", safe_query, limit)
        try:
            pmids = self._search_ids(safe_query, limit)
            if not pmids:
                logger.info("external.pubmed.search.done query=%s results=0", safe_query)
                return []
            chunks = self._fetch_abstracts(pmids)
            logger.info("external.pubmed.search.done query=%s results=%s", safe_query, len(chunks))
            return chunks[:limit]
        except Exception as exc:
            logger.exception("external.pubmed.search.failed error=%s", exc.__class__.__name__)
            return []

    def sanitize_query(self, query: str) -> str:
        cleaned = re.sub(r"\b(?:Mr|Mrs|Ms|Miss|Dr)\.?\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}", " ", query)
        cleaned = re.sub(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", " ", cleaned)
        cleaned = re.sub(r"\b(patient|mr|mrs|ms|name|named)\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ?.,;:")
        if any(term in cleaned.lower() for term in ["case report", "similar case", "published case"]):
            return cleaned
        return f"{cleaned} case report".strip()

    def sanitize_health_topic_query(self, query: str) -> str:
        cleaned = re.sub(r"\b(?:Mr|Mrs|Ms|Miss|Dr)\.?\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}", " ", query)
        cleaned = re.sub(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", " ", cleaned)
        cleaned = re.sub(
            r"\b(patient|mr|mrs|ms|name|named|explain|tell me about|what is|what are|define)\b",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"[^A-Za-z0-9\s\-/]", " ", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    def search_medlineplus(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        if not self.medlineplus_enabled:
            logger.info("external.medlineplus.skip reason=disabled")
            return []
        safe_query = self.sanitize_health_topic_query(query)
        if not safe_query:
            logger.info("external.medlineplus.skip reason=empty_safe_query")
            return []
        limit = max(1, min(top_k or self.settings.medlineplus_max_results, 10))
        logger.info("external.medlineplus.search.start query_chars=%s top_k=%s", len(safe_query), limit)
        params = {
            "db": "healthTopics",
            "term": safe_query,
            "rettype": "topic",
            "retmax": str(limit),
            "tool": self.settings.medlineplus_tool,
        }
        if self.settings.medlineplus_email:
            params["email"] = self.settings.medlineplus_email
        try:
            with httpx.Client(timeout=self.settings.medlineplus_timeout_seconds) as client:
                response = client.get(self.settings.medlineplus_base_url, params=params)
                response.raise_for_status()
            chunks = self._parse_medlineplus_xml(response.text)
            logger.info("external.medlineplus.search.done query_chars=%s results=%s", len(safe_query), len(chunks))
            return chunks[:limit]
        except Exception as exc:
            logger.exception("external.medlineplus.search.failed error=%s", exc.__class__.__name__)
            return []

    def _base_params(self) -> dict[str, str]:
        params = {"tool": self.settings.ncbi_tool}
        if self.settings.ncbi_email:
            params["email"] = self.settings.ncbi_email
        if self.settings.ncbi_api_key:
            params["api_key"] = self.settings.ncbi_api_key
        return params

    def _search_ids(self, query: str, limit: int) -> list[str]:
        params = {
            **self._base_params(),
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": str(limit),
            "sort": "relevance",
        }
        with httpx.Client(timeout=self.settings.external_search_timeout_seconds) as client:
            response = client.get(f"{self.settings.ncbi_base_url.rstrip('/')}/esearch.fcgi", params=params)
            response.raise_for_status()
            payload = response.json()
        return payload.get("esearchresult", {}).get("idlist", [])

    def _fetch_abstracts(self, pmids: list[str]) -> list[RetrievedChunk]:
        params = {
            **self._base_params(),
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
        with httpx.Client(timeout=self.settings.external_search_timeout_seconds) as client:
            response = client.get(f"{self.settings.ncbi_base_url.rstrip('/')}/efetch.fcgi", params=params)
            response.raise_for_status()
        return self._parse_pubmed_xml(response.text)

    def _parse_pubmed_xml(self, body: str) -> list[RetrievedChunk]:
        root = ET.fromstring(body)
        chunks = []
        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//PMID") or ""
            title = " ".join((article.findtext(".//ArticleTitle") or "Untitled PubMed result").split())
            abstract_parts = [
                " ".join("".join(part.itertext()).split())
                for part in article.findall(".//Abstract/AbstractText")
            ]
            abstract = " ".join(part for part in abstract_parts if part).strip()
            journal = article.findtext(".//Journal/Title") or "PubMed"
            year = article.findtext(".//PubDate/Year") or article.findtext(".//PubDate/MedlineDate") or ""
            text = f"Title: {title}\nJournal: {journal}"
            if year:
                text += f"\nYear: {year}"
            if abstract:
                text += f"\nAbstract: {abstract}"
            else:
                text += "\nAbstract: No abstract available from PubMed."
            chunks.append(
                RetrievedChunk(
                    chunk_id=f"pubmed:{pmid}",
                    document_id=f"pubmed:{pmid}",
                    document_name=f"PubMed {pmid}: {title}",
                    text=text,
                    score=None,
                )
            )
        return chunks

    def _parse_medlineplus_xml(self, body: str) -> list[RetrievedChunk]:
        root = ET.fromstring(body)
        chunks: list[RetrievedChunk] = []
        for index, document in enumerate(root.findall(".//document"), start=1):
            values: dict[str, list[str]] = {}
            for content in document.findall("./content"):
                name = content.attrib.get("name", "")
                if not name:
                    continue
                raw = "".join(content.itertext())
                cleaned = self._clean_medlineplus_text(raw)
                if cleaned:
                    values.setdefault(name, []).append(cleaned)
            title = self._first(values, "title") or self._first(values, "name") or "MedlinePlus health topic"
            url = self._first(values, "url") or self._first(values, "healthTopicUrl")
            summary = (
                self._first(values, "FullSummary")
                or self._first(values, "summary")
                or self._first(values, "snippet")
                or ""
            )
            groups = values.get("groupName", [])[:3]
            if not summary and not groups:
                continue
            text_parts = [f"Title: {title}", "Source: MedlinePlus"]
            if url:
                text_parts.append(f"URL: {url}")
            if groups:
                text_parts.append(f"Topic groups: {', '.join(groups)}")
            if summary:
                text_parts.append(f"Summary: {summary}")
            chunk_id = f"medlineplus:{self._slug(title) or index}"
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    document_id=chunk_id,
                    document_name=f"MedlinePlus: {title}",
                    text="\n".join(text_parts),
                    score=None,
                    metadata={"source": "medlineplus", "url": url or ""},
                )
            )
        return chunks

    def _first(self, values: dict[str, list[str]], key: str) -> str:
        items = values.get(key) or []
        return items[0] if items else ""

    def _clean_medlineplus_text(self, text: str) -> str:
        text = re.sub(r"<[^>]+>", " ", text)
        text = unescape(text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _slug(self, text: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
        return slug[:80]
