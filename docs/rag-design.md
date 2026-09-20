# RAG Design

## Corpus format: markdown + YAML frontmatter, not PDF
The 10 synthetic policy documents (`data/documents/*.md`) are written as markdown with frontmatter
(title, version, effective_date, department) rather than actual PDF/DOCX files. Chosen because:
easy to author realistically, easy to diff/version in git, and the content is what matters for
RAG quality -- format-parsing complexity (PDF layout extraction) would be solved problem overhead
that doesn't demonstrate anything new here. `src/ingestion/document_loader.py` also supports
`.pdf` via `pypdf` as a fallback path, so the pipeline isn't hard-coded to markdown only.

## Chunking: section-aware, not fixed-size sliding window
`src/ingestion/chunker.py` splits on markdown headers first, falling back to a size-bounded
sliding window only for a section that's still too long after that split. This is a deliberate
choice, not a default: these policy documents are short and each numbered section (e.g. "Section
4.2 Secondary Approval") is a self-contained rule. A generic fixed-size window risks cutting a
rule in half across two chunks -- a citation to "Section 4.2" should retrieve the whole rule, not
an arbitrary token-count slice of it. A fixed-size window would be the right default for long-form
unstructured prose (a contract, a long report); it's the wrong default for short, numbered policy
sections like this corpus.

## Chunk size / overlap: 400 tokens, 60 overlap
`config/retrieval.yaml`. In practice, section-aware splitting means most chunks here never
approach 400 tokens (`test_chunk_text_on_real_procurement_policy_produces_multiple_sections`
confirms real chunks stay well under it) -- the numbers mainly matter for the sliding-window
fallback path, sized to be roughly a paragraph's worth of context with enough overlap that a
rule split across the boundary isn't lost entirely on either side.

## Embedding model: local (sentence-transformers, all-MiniLM-L6-v2), not an API call
No per-embedding cost, no network dependency during indexing, and the model is small enough to
run on CPU comfortably at this document volume (a handful of documents, a few dozen chunks). A
hosted embedding API (OpenAI, Cohere, etc.) would be the better choice at real production scale
with a large, frequently-changing corpus -- not needed here, and adding an external API dependency
for indexing four documents would be complexity without benefit.

## Vector store: Chroma, local persistent mode, not a hosted vector DB
Same reasoning as the embedding model: this project's scale doesn't need Pinecone/Weaviate/etc.
Local persistence (`CHROMA_PERSIST_DIR`) means the whole system runs with `docker compose up` and
no external account setup, which matters for a portfolio project someone else needs to actually
run. `src/retrieval/vector_store.py` isolates all Chroma-specific calls behind plain functions
(`upsert_chunks`, `delete_document_chunks`, `count_chunks`) -- swapping to a hosted store later
would mean rewriting that one file, not anything that calls it.

## Re-indexing safety
`index_document()` calls `delete_document_chunks()` before re-adding a document's chunks. Without
this, re-indexing an updated document that now has fewer chunks than its previous version would
leave stale chunks from the old version behind in the vector store, silently. This was designed in
from the start rather than discovered as a bug later -- worth naming as a deliberate design choice
if asked, since "how do you handle document updates" is a fair RAG-system question.

## What's NOT built yet (by phase)
- **Hybrid retrieval / reranking** -- Tier 3 in FEATURES.md, not attempted yet
- **Retrieval evaluation set** -- Phase 15, not built yet (this doc covers ingestion, Phase 14;
  evaluation is next)
- **Query rewriting** -- Tier 3, not attempted
