# Hoopla

A search engine for movies

## Search algorithm

- Case insensitive
- Remove punctuation
- Split text into tokens
- Ignore stop words
- Use Porter's stemming algorithm
- Use an inverted index to find relevant documents
- Boolean search like m AND n, m OR n, m NOT n (not implemented)
- Use BM25 to rank results
  1. Results sort by term frequency
  2. Inverse document frequency (high score = rare term)
  3. Term frequency saturation
  4. Length normalization (the score is based on the length of the document, shorter documents score are boosted, longer documents score are reduced)
