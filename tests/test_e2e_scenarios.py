"""
Phase 28: canonical business scenarios traced through as much real code as this
environment allows. Scenarios 4-5 use a REAL ephemeral (in-memory, not persistent)
Chroma collection and REAL chunking/indexing/search code -- the one thing that can't
be real is the embedding MODEL itself, since sentence-transformers needs a model
download this environment's network allowlist doesn't include (huggingface.co isn't
reachable here). In its place: a deterministic bag-of-words hashing vectorizer --
a real, legitimate (if simplistic) embedding technique in its own right, not fake
data -- standing in for the semantic model. Everything downstream of "turn text into
a vector" (chunking, storage, cosine-similarity retrieval, agent tool dispatch) is the
genuine production code path, not mocked.
"""
import hashlib
import re
import math
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

import chromadb
import httpx
import pytest

from src.ingestion.document_loader import load_all_documents
from src.ingestion.chunker import chunk_text
from src.retrieval.search import semantic_search
from src.agent.agent import run_agent

VECTOR_DIM = 128
STOPWORDS = {"the", "a", "an", "of", "to", "for", "and", "or", "is", "are", "be", "this", "that", "in", "on", "at", "as", "with", "by"}


def _hashing_embed(text: str) -> list[float]:
    """Deterministic bag-of-words hashing vectorizer standing in for the real
    sentence-transformers model in this network-restricted environment. Tokenizes,
    drops stopwords, hashes each remaining word into one of VECTOR_DIM buckets, counts,
    L2-normalizes -- so two texts sharing distinctive vocabulary land closer together
    under cosine similarity, the same property a real embedding model provides, just
    via lexical overlap rather than learned semantics."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]

    vector = [0.0] * VECTOR_DIM
    for w in words:
        bucket = int(hashlib.md5(w.encode()).hexdigest(), 16) % VECTOR_DIM
        vector[bucket] += 1.0

    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


@pytest.fixture
def indexed_ephemeral_corpus():
    """Real documents, real chunking, real ephemeral Chroma, fake (hashing) embeddings.
    Yields nothing -- just sets up the collection and patches embed_query/get_collection
    for the duration of the test so src/retrieval/search.py's real code runs against it."""
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(name="test_policy_documents")

    docs = load_all_documents("data/documents")
    ids, texts, embeddings, metadatas = [], [], [], []
    for doc in docs:
        for chunk in chunk_text(doc.text):
            chunk_id = f"{doc.document_id}::{chunk.chunk_index}"
            ids.append(chunk_id)
            texts.append(chunk.text)
            embeddings.append(_hashing_embed(chunk.text))
            metadatas.append({
                "document_id": doc.document_id, "title": doc.title,
                "version": doc.version or "", "section_title": chunk.section_title or "",
                "chunk_index": chunk.chunk_index,
            })
    collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    with patch("src.retrieval.search.get_collection", return_value=collection), \
         patch("src.retrieval.search.embed_query", side_effect=_hashing_embed):
        yield collection


# ---------------------------------------------------------------------------
# Scenario 4: "What does the procurement policy say about high-value purchases?"
# question -> RAG (real chunk/index/search) -> policy chunk -> citation -> answer
# ---------------------------------------------------------------------------

def test_scenario_4_policy_question_retrieves_correct_section(indexed_ephemeral_corpus):
    results = semantic_search("What approval is required for purchase orders above $10,000?", top_k=3)

    assert len(results) > 0, "hashing vectorizer found nothing above min_similarity -- check corpus/query overlap"
    top = results[0]
    assert top.document_id == "procurement-policy"
    assert "approval" in (top.section_title or "").lower() or "approval" in top.text.lower()


def test_scenario_4_unrelated_question_does_not_confidently_match(indexed_ephemeral_corpus):
    """A question with essentially no vocabulary overlap with the corpus should not
    return a confident match -- this is the retrieval-side half of the grounding
    guarantee docs/agent-design.md describes."""
    results = semantic_search("What is the weather forecast for Tuesday?", top_k=3, min_similarity=0.3)
    assert results == [] or all(r.similarity_score < 0.5 for r in results)


# ---------------------------------------------------------------------------
# Scenario 5 (flagship): "Why are high-value orders breaching SLA, and does our
# procurement policy explain the additional step?"
# question -> agent -> [analytics tool (HTTP-real, server-fake) + RAG tool (fully
# real via indexed_ephemeral_corpus)] -> grounded synthesis -> answer + citations
# ---------------------------------------------------------------------------

def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_use_block(name, input_, block_id):
    return SimpleNamespace(type="tool_use", name=name, input=input_, id=block_id)


def fake_llm_response(*blocks):
    return SimpleNamespace(content=list(blocks))


def test_scenario_5_flagship_combined_question(indexed_ephemeral_corpus):
    """The one thing scripted here is the LLM's own reasoning (no real API key
    available) -- but both tools it calls run real code: get_sla_metrics goes through
    a real httpx.Client against a MockTransport server (real HTTP layer, fake
    backend), and search_policy_documents runs the fully real retrieval chain against
    indexed_ephemeral_corpus. The citations in the final response are not scripted --
    the document citation text is whatever the real search actually found."""
    def mock_ops_performance_handler(request):
        assert "/metrics/sla" in str(request.url)
        return httpx.Response(200, json=[{"case_count": 50, "breach_count": 9, "breach_rate_pct": 18.0}])

    mock_http_client = httpx.Client(transport=httpx.MockTransport(mock_ops_performance_handler))

    with patch("src.tools.client._get_client", return_value=(mock_http_client, False)):
        llm_client = MagicMock()
        llm_client.messages.create.side_effect = [
            fake_llm_response(
                tool_use_block("get_sla_metrics", {}, "t1"),
                tool_use_block(
                    "search_policy_documents",
                    {"query": "approval required for high value purchase orders"}, "t2",
                ),
            ),
            fake_llm_response(text_block(
                "SLA breaches are elevated (18.0%) and the Procurement Policy requires "
                "secondary approval for high-value orders, which likely explains the extra delay."
            )),
        ]

        result = run_agent(
            "Why are high-value orders breaching SLA, and does our procurement policy "
            "explain the extra step?",
            client=llm_client,
        )

    assert set(result.tools_used) == {"get_sla_metrics", "search_policy_documents"}

    data_citations = [c for c in result.citations if c["kind"] == "data"]
    doc_citations = [c for c in result.citations if c["kind"] == "document"]
    assert data_citations, "expected a data citation from the real (mocked-transport) SLA call"
    assert doc_citations, "expected a document citation from the REAL retrieval chain, not scripted"
    assert "procurement-policy" in doc_citations[0]["reference"].lower() or "procurement policy" in doc_citations[0]["reference"].lower()
