# StripThatNoise: Entity-Aware NLP Preprocessor

A lightweight NLP preprocessing tool that cleans raw `.txt` input for downstream use in LLM/ML workflows — while protecting named entities (people, places, organizations, dates, etc.) from being stripped or distorted during cleaning.

Built with **spaCy**, **NLTK**, and **Streamlit**.

---

## What it does

Given raw text, the pipeline runs through the following stages, in order:

1. **Text cleaning** — normalizes newlines and collapses extra whitespace.
2. **Named Entity Recognition (NER)** — runs first, on the full, unmodified text, since NER models rely on capitalization, punctuation, and grammatical context (articles, prepositions) to correctly detect entities.
3. **Tokenization** — splits the cleaned text into words and sentences.
4. **Stopword removal** — filters out common English stopwords (e.g. "the", "is", "of"), **except** when a word falls inside a detected entity's character span (e.g. the "of" in "Bank of America" is preserved).
5. **Stemming** — reduces words to a crude root form (e.g. "running" → "run"), skipped for protected entity words.
6. **Lemmatization** — reduces words to their dictionary base form (e.g. "theories" → "theory"), also skipped for protected entity words.
7. **POS tagging** — tags each token with its part of speech.

Entities are protected by **character position**, not by string matching — a word is only preserved when it physically falls inside an entity's span in the original text. This means a stopword like "of" is protected only when it's genuinely part of an entity name, not everywhere it appears in the document.

## Output

The tool displays each pipeline stage in the browser and lets you download the full result as a structured JSON file, containing:

- `original_text` — your unmodified input
- `entities` — extracted entities with their labels (`PERSON`, `ORG`, `GPE`, `DATE`, etc.)
- `sentences` / `tokens` — raw sentence and word tokenization
- `filtered_tokens` — stopwords removed, entities preserved
- `stemmed_tokens` — stemmed version of the filtered tokens
- `lemmatized_tokens` — lemmatized version of the tokens
- `pos_tags` — part-of-speech tag for each token

## Running it locally

```bash
# Clone the repo, then from the project root:
python -m venv venv
venv\Scripts\activate        # Windows (PowerShell)
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
streamlit run app.py
```

`requirements.txt` includes the spaCy model wheel directly, so no separate `spacy download` step is needed. NLTK's tokenizer and stopword data download automatically on first run.

## Known Limitations

- **Small model, occasional misses.** This project uses spaCy's `en_core_web_sm`, a lightweight statistical model. It can miss or misclassify less common entities — book/movie titles, abstract theories, or unusual names — especially in sentences with typos or unconventional grammar. Larger models (`en_core_web_lg` or `en_core_web_trf`) would improve accuracy but require more disk space/compute.
- **No custom entity rules.** The tool relies entirely on spaCy's default statistical NER — it doesn't include a rule-based `EntityRuler` for guaranteed recognition of specific known terms (e.g. a fixed list of book titles).
- **Two tokenizers, two behaviors.** The raw `tokens` field uses NLTK's `word_tokenize`, while `filtered_tokens`/`stemmed_tokens`/`lemmatized_tokens` are built from spaCy's tokenization (needed to track entity character positions). This means the two token lists may occasionally split text slightly differently (e.g. around punctuation or contractions).
- **English only.** Both spaCy's `en_core_web_sm` and NLTK's stopword list are English-specific; the pipeline isn't designed for multilingual input.
- **No persistence.** Each run is processed fresh in memory — there's no database or session history beyond the single JSON download.
- **Not built for very large documents.** The tool is designed around short-to-medium `.txt` inputs; very large files may be slow since the entire text is parsed as a single spaCy document.
- **Educational/portfolio project.** This was built as a hands-on demonstration of an entity-aware NLP preprocessing pipeline, not a production-grade or enterprise text-cleaning library. Expect rough edges on complex, deeply nested, or highly irregular text.

## Tech Stack

- [spaCy](https://spacy.io/) — NER, POS tagging, lemmatization
- [NLTK](https://www.nltk.org/) — tokenization, stopword list, stemming (Porter Stemmer)
- [Streamlit](https://streamlit.io/) — web interface
