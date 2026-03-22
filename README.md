# sparse_search

This repository currently serves as the versioned home for the sparse neural search design work that was previously only captured in conversation.

## Contents
- `docs/sparse-neural-search-design-notes.md` — running architecture notes and implementation guidance for hybrid lexical + sparse retrieval.
- `docs/design.md` — a more concrete integration-oriented design for adding a sparse weighted index beside the positional lexical index.

## Current design snapshot
The design notes currently recommend:
- keeping the existing positional lexical index for exact, phrase, and proximity behavior,
- adding a second sparse weighted inverted index keyed by `feature_id`,
- using sparse dot-product accumulation for retrieval,
- supporting low-weight feature expansion for semantic/variant matching,
- starting with roughly `32` sparse features per short document, and
- combining lexical and sparse channels with a normalized weighted sum.

## Suggested next implementation steps
1. Define document and query sparse-vector types.
2. Build the sparse weighted inverted index alongside the positional index.
3. Implement sparse score accumulation and hybrid score fusion.
4. Add evaluation slices for exact, variant, alias, and ambiguity-sensitive queries.

For the detailed rationale and rollout guidance, see `docs/sparse-neural-search-design-notes.md`.
For a more implementation-oriented view of the index layout, query flow, and scoring model, see `docs/design.md`.
