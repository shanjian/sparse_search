# Learned Sparse Integration Design

## Goal

Integrate learned sparse retrieval into the existing C++ search stack without weakening the current positional inverted index.

The main idea is:

- Keep the current lexical positional index for exact match, phrase match, and proximity logic.
- Add a separate sparse index for learned weighted terms that survive pruning.
- Combine both scores at query time.

## Why A Separate Index

The current lexical postings have a different meaning from learned sparse postings.

Lexical postings:

- based on original document tokens
- keep positions
- support phrase queries like `"bank of america"` or `"korean president"`

Sparse postings:

- based on learned weighted terms
- may include expansion terms not present literally in the text
- do not have meaningful positions
- are used for semantic matching, not phrase reconstruction

Because of that, sparse terms should not be inserted into the positional index with fake positions.

## Proposed Index Layout

### Lexical Index

Keep the current structure unchanged:

```cpp
struct LexicalPosting {
  DocId doc_id;
  uint32_t tf;
  PositionOffset positions_offset;
};
```

Conceptually:

```text
lexical_index[term] -> [(doc_id, tf, positions...)]
```

### Sparse Index

Add a second field/index:

```cpp
struct SparsePosting {
  DocId doc_id;
  float weight;
};
```

Conceptually:

```text
sparse_index[sparse_term] -> [(doc_id, learned_weight)]
```

This sparse term dictionary should be separate from the lexical term dictionary. Even if the token strings look similar, the tokenization/vocabulary semantics are different.

## Offline Document Pipeline

Document sparse vectors should be generated offline in Python and then imported into the C++ index build.

Pipeline:

1. Encode each document with the sparse model.
2. Get a sparse term->weight map.
3. Prune low-value terms.
4. Export surviving `(doc_id, term, weight)` tuples.
5. Build the sparse postings lists in C++.

Suggested exported row:

```text
doc_id<TAB>sparse_term<TAB>weight
```

Or integerized:

```text
doc_id<TAB>sparse_term_id<TAB>weight
```

## Pruning

Do not keep all vocabulary terms.

Typical practice is to keep only important active terms, for example by:

- top-K per document
- minimum weight threshold
- max active dimensions

Document-side pruning is especially important because it controls:

- index size
- posting list growth
- retrieval latency

Query-side pruning can be lighter because query vectors are small and generated online.

## Query-Time Pipeline

For a query like `korean president`:

1. Run the existing lexical query path.
2. Run the sparse query encoder.
3. Prune query sparse terms if needed.
4. Fetch sparse postings for each sparse query term.
5. Accumulate sparse scores.
6. Merge lexical and sparse scores.

Pseudo-flow:

```cpp
for (auto& [term, q_weight] : sparse_query_terms) {
  for (auto& posting : sparse_index[term]) {
    sparse_score[posting.doc_id] += q_weight * posting.weight;
  }
}
```

## Scoring

Sparse score is a weighted overlap:

```text
sparse_score(doc) = sum_t query_weight[t] * doc_weight[t]
```

Final score can be a fusion:

```text
final_score(doc) =
    lexical_score(doc)
  + phrase_boost(doc)
  + alpha * sparse_score(doc)
```

Where:

- `lexical_score` can be BM25 or your existing ranker
- `phrase_boost` comes from the positional index
- `alpha` tunes sparse influence

## Example: `korean president`

Assume the sparse query encoder outputs:

```text
korean: 2.8
president: 2.8
korea: 2.6
presidents: 2.2
leader: 1.2
```

### Good Document

Document A is about `korea president` and its sparse doc vector includes:

```text
president: 3.0
korea: 2.2
korean: 0.8
```

Sparse score:

```text
2.8*3.0 + 2.6*2.2 + 2.8*0.8
```

### Bad Document

Document B mostly matches `korean korea` and its sparse doc vector includes:

```text
korean: 2.0
korea: 1.7
president: 0.1
```

Sparse score:

```text
2.8*2.0 + 2.6*1.7 + 2.8*0.1
```

Document A wins because it matches the important semantic concept `president`, not just related nationality terms.

This is the key point:

- we do not need to explicitly rewrite the query to `korea president`
- the sparse overlap already rewards documents that contain the better semantic combination

Phrase matching still stays on the lexical side only.

## Candidate Generation Options

There are 2 practical ways to integrate sparse scoring.

### Option 1: Unified Retrieval

Use both indexes to generate candidates and score in one pass.

Pros:

- best recall
- full hybrid retrieval

Cons:

- more integration work

### Option 2: Lexical First, Sparse Augmentation

Use lexical retrieval for initial candidates, then add sparse scores on that candidate set.

Pros:

- easier first integration
- preserves current retrieval behavior

Cons:

- sparse recall benefit is limited by lexical candidate recall

For a first C++ rollout, Option 2 is lower risk.

## C++ Data Structures

One reasonable sketch:

```cpp
using SparseTermId = uint32_t;

struct SparsePosting {
  DocId doc_id;
  float weight;
};

struct SparsePostingsList {
  std::vector<SparsePosting> postings;
};

class SparseIndex {
 public:
  const SparsePostingsList* Find(SparseTermId term_id) const;

 private:
  std::vector<SparsePostingsList> postings_;
};
```

Query-time accumulator:

```cpp
absl::flat_hash_map<DocId, float> sparse_scores;
```

Or your existing scorer heap / accumulator if one already exists.

## Tokenization And Vocabulary

Do not assume the sparse model vocabulary matches the lexical analyzer vocabulary.

Important rules:

- sparse terms come from the model tokenizer/vocabulary
- lexical terms come from your current analyzer
- treat them as separate namespaces

This avoids bugs around:

- subword tokens
- casing
- punctuation normalization
- multilingual tokenization behavior

## Storage And Compression

Sparse postings can often be stored compactly:

- sort postings by `doc_id`
- delta encode `doc_id`
- quantize weights if needed later

Keep v1 simple first:

- raw float weights
- straightforward postings build

Optimize compression only after quality is validated.

## Suggested First Rollout

1. Keep lexical retrieval exactly as-is.
2. Build a separate sparse index offline.
3. Encode queries online in Python or C++.
4. Add sparse scoring as an extra score term.
5. Tune `alpha`.
6. Validate phrase queries are unchanged on the lexical path.

## Recommendation

The recommended architecture is:

- positional lexical index for exact and phrase logic
- separate pruned sparse index for learned semantic matching
- hybrid score fusion at query time

This gives you semantic expansion benefits without corrupting the positional semantics of the existing index.
