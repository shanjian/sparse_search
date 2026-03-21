# Sparse neural search design notes

This document captures the running decisions from our discussion so that each round of interaction leaves a durable record.

## Round 1: Initial architecture framing

### Discussion summary
- The current engine already has a positional inverted index where each posting looks like `(doc_id, pos0, pos1, ...)`.
- Sparse neural search should be modeled as a second inverted index over sparse weighted features rather than as a replacement for inverted indexing.
- In the sparse index, each posting is `(doc_id, weight)` and retrieval uses weighted overlap between sparse query and document vectors.
- The lexical and sparse indexes serve different purposes: lexical search remains best for exact matches, phrase/proximity search, and explainability, while sparse retrieval improves semantic matching and vocabulary mismatch.

### Decisions
- Keep the positional inverted index as part of the system.
- Add a separate sparse weighted index keyed by sparse feature/token ID.
- Use sparse dot-product style scoring for the sparse retriever.
- Treat sparse keys as generic features, not necessarily literal surface terms.

## Round 2: Early design decisions and corpus guidance

### Discussion summary
- We want to settle a few early assumptions before optimizing the implementation.
- Hybrid retrieval is preferred over replacing the positional engine outright.
- Online generation of sparse query vectors is assumed to be inexpensive enough for serving.
- The corpus consists mostly of short documents, and very small-weight sparse features will be trimmed.

### Decisions
- Use a hybrid design: keep the positional inverted index and add a sparse weighted index alongside it.
- Assume query-side sparse encoding can be generated cheaply online.
- Plan to prune tiny-weight features from document sparse vectors.

### Guidance for mostly short documents
For mostly short documents, a practical starting target is to keep roughly **16 to 64 sparse features per document after pruning**, with **32 features** being a strong default starting point.

Recommended rollout bands:
- **Very short documents** (titles, short messages, brief metadata): start around **8 to 24** features.
- **Typical short documents** (short paragraphs, issue summaries, comments, snippets): start around **16 to 48** features.
- **Upper bound for early experiments**: allow up to **64** features before evaluating whether the extra recall is worth the larger postings and latency.

Why this range is a good early choice:
- Short documents usually do not need large sparse vectors because there is less semantic content to preserve.
- Trimming aggressively keeps posting lists smaller, reduces memory use, and lowers query-time accumulation cost.
- Sparse retrieval often gets most of its benefit from the top-weighted features; tiny weights tend to add noise faster than value.

### Practical pruning advice
Use these as initial heuristics rather than fixed rules:
- Keep the **top-K features per document** after encoding.
- Also apply a **minimum weight threshold** to drop clearly insignificant features.
- Start evaluation with:
  - `K = 32` as the default,
  - a comparison sweep at `K in {16, 32, 48, 64}`.
- If recall saturates early, prefer the smaller K.
- If your model produces many near-zero expansions, increase the threshold before increasing K.

### What to validate next
The right feature count depends on evaluation rather than theory alone. The next experiment should measure:
- recall / MRR / nDCG lift versus the lexical baseline,
- average number of postings touched per query,
- latency impact of larger K,
- how often low-weight features contribute to top-ranked results.

## Current working design
- **Lexical index:** positional postings for exact term and phrase/proximity search.
- **Sparse index:** weighted postings `(doc_id, weight)` for sparse neural retrieval.
- **Retrieval mode:** hybrid.
- **Query encoding:** online and assumed cheap.
- **Document pruning default for early experiments:** start with **top 32 sparse features per short document**, then tune within the 16 to 64 range based on relevance and latency measurements.

## Round 3: Feature expansion for variant matching

### Discussion summary
- We want sparse retrieval to support feature expansion so that documents can match closely related query forms even when the literal surface form differs.
- A concrete example is allowing a document containing `Korea president` to also match a query like `korean president`.
- This is a primary strength of sparse neural retrieval: the encoder can assign non-zero weight to related expansion features that do not appear verbatim in the original document text.

### Decisions
- Use the sparse index to carry **expanded features**, not just literal observed tokens.
- Continue to keep the positional lexical index unchanged for exact and phrase matching.
- Treat feature expansion as a **soft weighted match**, not as a rewrite of the stored document text.
- Favor model-driven expansion weights over large rule dictionaries as the main mechanism.

### Working interpretation
For a document like `Korea president`, the sparse document encoder should be allowed to emit a vector such as:
- `(korea, 1.0)`
- `(president, 0.9)`
- `(korean, 0.4)`
- `(south_korea, 0.3)`

Then a query like `korean president` can overlap on:
- the literal/shared concept around `president`, and
- the expanded feature `korean`, even if `korean` never appeared verbatim in the document.

This means the sparse index is not just an alternate storage format for lexical terms. It is a weighted feature space that can include:
- inflectional variants,
- derivational variants,
- aliases,
- closely related entities or concepts,
- learned semantic expansions.

### Important boundary
Feature expansion should improve recall, but it should not replace exact matching behavior:
- Phrase queries and exact token constraints should still rely on the positional lexical index.
- Sparse expansion should act as an additional retrieval channel for soft semantic matching.
- Expansion weights should usually be lower than the weights of directly observed core features unless evaluation shows otherwise.

### Practical guidance for early implementation
For an initial implementation, prefer this ordering:
1. **Literal feature retention:** keep high weight on features directly supported by the document text.
2. **Light expansion:** allow a smaller set of non-literal expanded features such as `korean` for `Korea`.
3. **Aggressive pruning:** expanded features should be pruned more aggressively than literal ones if they have very small weights.

A useful early heuristic is:
- keep all strong literal features that survive normal pruning,
- allow expansion features into the top-K set only if their weights are competitive,
- optionally use a stricter minimum threshold for non-literal expansions.

### What to validate next for expansion
The next evaluation should specifically measure whether expansion helps cases like `Korea` -> `korean` without creating too many false positives:
- gain on morphological/variant queries,
- precision impact from added expansion terms,
- contribution of expanded versus literal features in top-ranked results,
- whether expanded features need a separate pruning threshold or cap.

## Updated working design
- **Lexical index:** positional postings for exact term and phrase/proximity search.
- **Sparse index:** weighted postings `(doc_id, weight)` for sparse neural retrieval and feature expansion.
- **Retrieval mode:** hybrid.
- **Query encoding:** online and assumed cheap.
- **Document pruning default for early experiments:** start with **top 32 sparse features per short document**, then tune within the 16 to 64 range based on relevance and latency measurements.
- **Expansion policy:** use sparse features to represent soft expansions such as `Korea` -> `korean`, while keeping exact and phrase semantics in the lexical index.

## Round 4: Concrete implementation direction

### Discussion summary
- The previous notes captured the architectural intent, but the next useful step is to make the design concrete enough to implement.
- The key requirement is to support hybrid retrieval while allowing sparse feature expansion for soft matches such as `Korea` -> `korean`.
- We want a minimal design that fits naturally beside the existing positional index rather than replacing it.

### Decisions
- Implement the sparse retriever as a **separate weighted inverted index** keyed by `feature_id`.
- Keep the existing positional index unchanged and query both channels during retrieval.
- Score the sparse channel with a weighted overlap accumulator using query and document feature weights.
- Treat expansion features as first-class sparse features, but keep them lower-weighted and more aggressively pruned than literal features.

### Minimal document-side data model
For each document, store a pruned sparse vector:
- `doc_sparse = [(feature_id, feature_weight, feature_kind)]`

Where:
- `feature_id` is the sparse vocabulary ID,
- `feature_weight` is the document-side sparse weight,
- `feature_kind` is optional metadata such as `literal` or `expanded`.

The sparse inverted index then becomes:
- `feature_id -> [(doc_id, doc_weight)]`

The `feature_kind` does not need to be stored in the posting if it is only used during indexing and pruning. It may still be useful in debug output or offline analysis.

### Minimal query-time algorithm
1. Encode the query into a sparse vector: `[(feature_id, query_weight)]`.
2. For each query feature, read the corresponding posting list.
3. For each posting `(doc_id, doc_weight)`, accumulate:
   - `score_sparse[doc_id] += query_weight * doc_weight`
4. Combine the sparse score with the lexical score using a hybrid fusion rule.
5. Return the top-k documents.

### Recommended first hybrid scoring rule
Start with a simple weighted sum after per-channel normalization:
- `score_final = alpha * score_lexical + beta * score_sparse`

For early experiments:
- start with `alpha = 1.0`,
- start with `beta = 0.3` to `0.7`,
- then tune on validation queries.

The reason to start with a smaller sparse weight is that expansion features can improve recall quickly but may also introduce soft false positives if boosted too aggressively.

### Expansion policy for implementation
For a document containing `Korea president`, the encoder can emit both literal and expanded features:
- literal: `korea`, `president`
- expanded: `korean`, `south_korea`

Implementation guidance:
- Literal features should usually keep the strongest weights.
- Expanded features should be allowed into the sparse vector only if their weights remain meaningful after pruning.
- When pruning, prefer dropping low-weight expanded features before dropping similarly weighted literal features.

A practical implementation rule is:
- keep top `K_literal` literal features,
- keep top `K_expanded` expanded features,
- or enforce one shared top-K with a stricter threshold for expanded features.

A good first experiment for mostly short documents is:
- total top-K around `32`,
- with no more than `8` to `12` expansion features unless evaluation shows a larger budget is beneficial.

### Retrieval boundary between the two indexes
Use the lexical index for:
- exact term matching,
- phrase queries,
- proximity constraints,
- snippets and highlighting.

Use the sparse index for:
- soft semantic matching,
- morphological or derivational variants,
- aliases and closely related concepts,
- learned feature expansion.

This boundary keeps the implementation understandable and prevents sparse expansion from weakening exact-match behavior.

### Suggested evaluation matrix
To decide whether the implementation is working well, evaluate at least these slices separately:
- exact-match queries,
- variant/morphology queries such as `Korea` vs `korean`,
- alias/entity expansion queries,
- short-query ambiguous cases where expansion may hurt precision.

Track:
- Recall@k / MRR / nDCG,
- sparse postings touched per query,
- latency p50 / p95,
- average number of expansion features retained per document,
- fraction of final score coming from expanded features.

## Current implementation target
- **Architecture:** hybrid lexical + sparse retrieval.
- **Sparse posting format:** `(doc_id, weight)`.
- **Sparse scoring:** sparse dot-product accumulation.
- **Expansion behavior:** allow low-weight non-literal features such as `korean` for a document containing `Korea`.
- **Pruning default for short documents:** start near **top 32 total sparse features**, with a smaller sub-budget for expansions.
- **First fusion rule:** weighted sum of lexical and sparse scores after per-channel normalization.
