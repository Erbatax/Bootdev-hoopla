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

## Chunking algorithm

- Split text into sentences
- Strip whitespace
- Remove empty sentences
- Combine sentences into chunks, with overlap. E.g. chunk 1 = sentence 1, sentence 2, sentence 3, chunk 2 = sentence 3, sentence 4, sentence 5
- Create embedding for each chunk

### More precise embedding, but need more computing power

- ColBERT ? create embedding for each word
- Late chunking ? one embedding for the entire document, then one for each word

## Combined search

- Combine search results from both algorithms.

1. The score are normalized to 0-1 and combined using a weighted average.
2. Another method to compare is to only use the position in the list, first, second, third, etc (rrf score).

## LLM

- We can also use Large language models before or after the search

1. User enters a query
2. LLM processes the query

- Fixes typos
- Expands meaning
- Breaks apart complex queries
- Adds missing context

3. Search is performed with the improved query
4. Results are returned

## The Rag Pipeline

1. The user enters a query.
2. The search system retrieves relevant documents.
3. The LLM processes the query and retrieved documents.

- Summarizes information
- Answers questions

4. The generated response is returned to the user.

## Agentic system (slower but more intelligent)

while not done: # Choose tool based on what we learned
tool = pick_next_tool(previous_results) // Keyword search, semantic search, actor search, review search etc # Search with that tool
results = tool.search(query) # Update our knowledge
previous_results.append(results)
