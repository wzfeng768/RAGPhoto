"""
Web search tools for Agentic RAG v2.

This module provides:
1) AcademicMultiSourceSearcher: arXiv/PubMed/Semantic Scholar/CORE search
2) WebSearchTool: orchestrator-facing tool with fallback to DuckDuckGo
"""

from __future__ import annotations

import html
import logging
import re
from urllib.parse import parse_qs, unquote, urlparse
from typing import Any, Dict, Iterable, List, Optional

import requests

from .base_tool import BaseTool

logger = logging.getLogger(__name__)


class AcademicMultiSourceSearcher:
    """
    Multi-source academic searcher.

    Sources:
    - arXiv
    - PubMed
    - Semantic Scholar
    - CORE
    """

    DEFAULT_SOURCES = ["arxiv", "pubmed", "semantic_scholar", "core"]

    def __init__(
        self,
        enabled_sources: Optional[List[str]] = None,
        semantic_scholar_key: str = "",
        core_key: str = "",
        pubmed_email: str = "user@example.com",
        timeout: int = 10,
    ):
        self.enabled_sources = enabled_sources or self.DEFAULT_SOURCES.copy()
        self.semantic_scholar_key = semantic_scholar_key
        self.core_key = core_key
        self.pubmed_email = pubmed_email
        self.timeout = timeout

    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search across enabled academic sources and deduplicate."""
        if not query or not query.strip():
            return []

        query = query.strip()
        enabled = [s for s in self.enabled_sources if s in self.DEFAULT_SOURCES]
        if not enabled:
            return []

        results_per_source = max(1, max_results // len(enabled))
        all_results: List[Dict[str, Any]] = []

        if "arxiv" in enabled:
            all_results.extend(self._search_arxiv(query, results_per_source))
        if "pubmed" in enabled:
            all_results.extend(self._search_pubmed(query, results_per_source))
        if "semantic_scholar" in enabled:
            all_results.extend(self._search_semantic_scholar(query, results_per_source))
        if "core" in enabled:
            all_results.extend(self._search_core(query, results_per_source))

        deduped = self._deduplicate_results(all_results)
        return deduped[:max_results]

    def _search_arxiv(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        try:
            import arxiv

            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )

            results: List[Dict[str, Any]] = []
            for paper in search.results():
                results.append(
                    {
                        "title": paper.title,
                        "url": paper.entry_id,
                        "pdf_url": paper.pdf_url,
                        "abstract": paper.summary,
                        "authors": ", ".join([a.name for a in paper.authors]) if paper.authors else "",
                        "published": paper.published.strftime("%Y-%m-%d") if paper.published else None,
                        "source": "arxiv",
                        "arxiv_id": paper.entry_id.split("/")[-1] if paper.entry_id else None,
                    }
                )
            return results
        except ImportError:
            logger.warning("arxiv package not installed; skipping arXiv search.")
            return []
        except Exception as exc:
            logger.warning("arXiv search failed: %s", exc)
            return []

    def _search_pubmed(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        try:
            from Bio import Entrez

            Entrez.email = self.pubmed_email

            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
            record = Entrez.read(handle)
            handle.close()

            id_list = record.get("IdList", [])
            if not id_list:
                return []

            handle = Entrez.efetch(db="pubmed", id=id_list, retmode="xml")
            records = Entrez.read(handle)
            handle.close()

            results: List[Dict[str, Any]] = []
            for article in records.get("PubmedArticle", []):
                try:
                    medline = article["MedlineCitation"]
                    article_data = medline["Article"]

                    authors: List[str] = []
                    for author in article_data.get("AuthorList", [])[:5]:
                        if "LastName" in author and "Initials" in author:
                            authors.append(f"{author['LastName']} {author['Initials']}")

                    pmid = str(medline.get("PMID", ""))
                    doi = None
                    for eid in article_data.get("ELocationID", []):
                        try:
                            if eid.attributes.get("EIdType") == "doi":
                                doi = str(eid)
                                break
                        except Exception:
                            continue

                    abstract_text = ""
                    abstract = article_data.get("Abstract", {})
                    if isinstance(abstract, dict):
                        abstract_list = abstract.get("AbstractText", [])
                        if abstract_list:
                            abstract_text = str(abstract_list[0])

                    results.append(
                        {
                            "title": str(article_data.get("ArticleTitle", "")),
                            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                            "abstract": abstract_text,
                            "authors": ", ".join(authors),
                            "published": None,
                            "source": "pubmed",
                            "pubmed_id": pmid,
                            "doi": doi,
                        }
                    )
                except Exception:
                    continue
            return results
        except ImportError:
            logger.warning("biopython not installed; skipping PubMed search.")
            return []
        except Exception as exc:
            logger.warning("PubMed search failed: %s", exc)
            return []

    def _search_semantic_scholar(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        try:
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                "query": query,
                "limit": max_results,
                "fields": "title,abstract,authors,year,url,openAccessPdf,externalIds",
            }
            headers: Dict[str, str] = {}
            if self.semantic_scholar_key:
                headers["x-api-key"] = self.semantic_scholar_key

            response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            results: List[Dict[str, Any]] = []
            for paper in data.get("data", []):
                authors = [a.get("name", "") for a in paper.get("authors", [])[:5]]
                open_access_pdf = paper.get("openAccessPdf") or {}
                ext_ids = paper.get("externalIds") or {}

                results.append(
                    {
                        "title": paper.get("title", ""),
                        "url": paper.get("url", ""),
                        "pdf_url": open_access_pdf.get("url"),
                        "abstract": paper.get("abstract", ""),
                        "authors": ", ".join(a for a in authors if a),
                        "published": str(paper.get("year")) if paper.get("year") else None,
                        "source": "semantic_scholar",
                        "doi": ext_ids.get("DOI"),
                        "arxiv_id": ext_ids.get("ArXiv"),
                        "pubmed_id": ext_ids.get("PubMed"),
                    }
                )
            return results
        except Exception as exc:
            logger.warning("Semantic Scholar search failed: %s", exc)
            return []

    def _search_core(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        try:
            url = "https://api.core.ac.uk/v3/search/works"
            params = {"q": query, "limit": max_results}
            headers: Dict[str, str] = {}
            if self.core_key:
                headers["Authorization"] = f"Bearer {self.core_key}"

            response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
            if response.status_code == 401:
                logger.info("CORE API key required/invalid; skipping CORE.")
                return []

            response.raise_for_status()
            data = response.json()

            results: List[Dict[str, Any]] = []
            for paper in data.get("results", []):
                fulltext_urls = paper.get("sourceFulltextUrls") or []
                authors = paper.get("authors") or []
                results.append(
                    {
                        "title": paper.get("title", ""),
                        "url": fulltext_urls[0] if fulltext_urls else "",
                        "pdf_url": paper.get("downloadUrl"),
                        "abstract": paper.get("abstract", ""),
                        "authors": ", ".join(a.get("name", "") for a in authors[:5] if a.get("name")),
                        "published": str(paper.get("yearPublished")) if paper.get("yearPublished") else None,
                        "source": "core",
                        "doi": paper.get("doi"),
                    }
                )
            return results
        except Exception as exc:
            logger.warning("CORE search failed: %s", exc)
            return []

    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        unique_results: List[Dict[str, Any]] = []

        for result in results:
            identifiers: List[str] = []
            if result.get("url"):
                identifiers.append(str(result["url"]))
            if result.get("doi"):
                identifiers.append(f"doi:{result['doi']}")
            if result.get("arxiv_id"):
                identifiers.append(f"arxiv:{result['arxiv_id']}")
            if result.get("pubmed_id"):
                identifiers.append(f"pubmed:{result['pubmed_id']}")

            if not identifiers:
                continue

            if any(identifier in seen for identifier in identifiers):
                continue

            for identifier in identifiers:
                seen.add(identifier)
            unique_results.append(result)

        return unique_results


class WebSearchTool(BaseTool):
    """
    Web search tool for the agent.

    Strategy:
    - academic mode: use academic multi-source search only
    - duckduckgo mode: use DuckDuckGo only
    - both mode: academic first, fallback to DuckDuckGo when empty/error
    """

    def __init__(
        self,
        max_results: int = 5,
        mode: str = "both",
        enabled_sources: Optional[List[str]] = None,
        semantic_scholar_key: str = "",
        core_key: str = "",
        pubmed_email: str = "user@example.com",
        timeout: int = 10,
        engine: str = "duckduckgo",
    ):
        super().__init__(name="web_search")
        self.max_results = max_results
        self.mode = (mode or "both").lower()
        self.engine = (engine or "duckduckgo").lower()
        self.timeout = timeout

        self.academic_searcher = AcademicMultiSourceSearcher(
            enabled_sources=enabled_sources,
            semantic_scholar_key=semantic_scholar_key,
            core_key=core_key,
            pubmed_email=pubmed_email,
            timeout=timeout,
        )

    def get_description(self) -> str:
        return (
            "Searches latest web information with academic sources "
            "(arXiv/PubMed/Semantic Scholar/CORE) and optional DuckDuckGo fallback."
        )

    def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        max_results = int(kwargs.get("max_results", self.max_results))
        if not query or not query.strip():
            return {
                "query": query,
                "contexts": [],
                "results": [],
                "fallback_used": False,
                "providers_used": [],
                "total_results": 0,
            }

        query = query.strip()
        providers_used: List[str] = []
        fallback_used = False
        records: List[Dict[str, Any]] = []

        academic_allowed = self.mode in {"academic", "both"}
        ddg_allowed = self.mode in {"duckduckgo", "both"} and self.engine == "duckduckgo"

        if academic_allowed:
            try:
                academic_records = self.academic_searcher.search(query, max_results=max_results)
                if academic_records:
                    providers_used.extend(sorted({r.get("source", "academic") for r in academic_records}))
                    records = academic_records
            except Exception as exc:
                logger.warning("Academic web search failed: %s", exc)

        # Fallback only if no results from academic branch.
        if not records and ddg_allowed:
            fallback_used = academic_allowed
            ddg_records = self._search_duckduckgo(query, max_results=max_results)
            if ddg_records:
                providers_used.append("duckduckgo")
                records = ddg_records

        contexts = self._build_contexts(records)
        return {
            "query": query,
            "contexts": contexts,
            "results": records,
            "fallback_used": fallback_used,
            "providers_used": providers_used,
            "total_results": len(contexts),
        }

    def _search_duckduckgo(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.info("duckduckgo-search package not installed; using HTTP fallback.")
            return self._search_duckduckgo_html(query, max_results)

        rows: List[Dict[str, Any]] = []
        try:
            with DDGS() as ddgs:
                rows = list(self._ddg_text(ddgs, query, max_results))
        except Exception as exc:
            logger.warning("DuckDuckGo search failed: %s", exc)
            return self._search_duckduckgo_html(query, max_results)

        records: List[Dict[str, Any]] = []
        for row in rows:
            title = row.get("title", "")
            url = row.get("href", "") or row.get("url", "")
            body = row.get("body", "") or row.get("snippet", "")
            if not (title or body):
                continue
            records.append(
                {
                    "title": title,
                    "url": url,
                    "abstract": body,
                    "source": "duckduckgo",
                    "authors": "",
                    "published": None,
                }
            )

        return self._deduplicate_by_url(records)[:max_results]

    def _search_duckduckgo_html(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Lightweight DDG fallback using HTML endpoint when DDGS package is unavailable.
        """
        try:
            response = requests.get(
                "https://duckduckgo.com/html/",
                params={"q": query},
                headers={"User-Agent": "Mozilla/5.0 AgenticRAG/2.0"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            page = response.text
        except Exception as exc:
            logger.warning("DuckDuckGo HTTP fallback failed: %s", exc)
            return []

        title_pattern = re.compile(
            r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            re.IGNORECASE | re.DOTALL,
        )
        snippet_pattern = re.compile(
            r'<(?:a|div)[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|div)>',
            re.IGNORECASE | re.DOTALL,
        )

        title_matches = title_pattern.findall(page)
        snippets = [self._strip_tags(html.unescape(match)) for match in snippet_pattern.findall(page)]

        records: List[Dict[str, Any]] = []
        for idx, (href, raw_title) in enumerate(title_matches):
            title = self._strip_tags(html.unescape(raw_title)).strip()
            url = self._normalize_ddg_href(html.unescape(href))
            abstract = snippets[idx].strip() if idx < len(snippets) else ""
            if not (title or abstract):
                continue
            records.append(
                {
                    "title": title,
                    "url": url,
                    "abstract": abstract,
                    "source": "duckduckgo",
                    "authors": "",
                    "published": None,
                }
            )
            if len(records) >= max_results:
                break

        return self._deduplicate_by_url(records)

    def _ddg_text(self, ddgs: Any, query: str, max_results: int) -> Iterable[Dict[str, Any]]:
        """
        Handle different duckduckgo-search versions.
        """
        try:
            yield from ddgs.text(query, max_results=max_results)
            return
        except TypeError:
            pass

        # Older signature fallback
        yield from ddgs.text(keywords=query, max_results=max_results)

    def _build_contexts(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        contexts: List[Dict[str, Any]] = []
        for rank, record in enumerate(records):
            source = (record.get("source") or "web").lower()
            title = record.get("title", "").strip()
            abstract = (record.get("abstract") or record.get("snippet") or "").strip()
            authors = (record.get("authors") or "").strip()
            published = record.get("published")
            url = (record.get("url") or "").strip()

            content_lines = []
            if title:
                content_lines.append(title)
            if abstract:
                content_lines.append(f"Summary: {abstract}")
            if authors:
                content_lines.append(f"Authors: {authors}")
            if published:
                content_lines.append(f"Published: {published}")
            if url:
                content_lines.append(f"URL: {url}")

            content = "\n".join(content_lines).strip()
            if not content:
                continue

            contexts.append(
                {
                    "content": content,
                    "source": f"web:{source}",
                    "similarity": self._score_for_result(source=source, rank=rank),
                    "metadata": {
                        "title": title,
                        "url": url,
                        "authors": authors,
                        "published": published,
                        "provider": source,
                    },
                }
            )
        return contexts

    def _score_for_result(self, source: str, rank: int) -> float:
        base_scores = {
            "arxiv": 0.80,
            "pubmed": 0.78,
            "semantic_scholar": 0.76,
            "core": 0.72,
            "duckduckgo": 0.66,
        }
        base = base_scores.get(source, 0.62)
        score = base - (0.02 * rank)
        return max(0.35, min(0.95, score))

    def _deduplicate_by_url(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        deduped: List[Dict[str, Any]] = []
        for row in rows:
            url = (row.get("url") or "").strip()
            key = url or row.get("title", "")
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        return deduped

    def _strip_tags(self, text: str) -> str:
        cleaned = re.sub(r"<[^>]+>", " ", text)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def _normalize_ddg_href(self, href: str) -> str:
        """
        Convert DDG redirect links to target URLs when possible.
        """
        if not href:
            return ""

        parsed = urlparse(href)
        qs = parse_qs(parsed.query)
        if "uddg" in qs and qs["uddg"]:
            return unquote(qs["uddg"][0])
        return href

