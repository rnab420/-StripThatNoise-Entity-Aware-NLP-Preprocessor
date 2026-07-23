#Import necessary libraries

import nltk
import spacy
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer

# Download necessary NLTK resources if not already present

def ensure_nltk_resources():
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name)

def clean_text(text: str) -> str:
    """Remove newlines and extra whitespace from text."""
    text = text.replace("\r\n", " ").replace("\n", " ")
    # Collapse multiple spaces into one
    return " ".join(text.split())

@st.cache_resource
def load_spacy_model():
    """Loads and caches the spaCy model so it only runs once."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download

        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

@st.cache_resource
def get_stopwords():
    """Loads and caches English stopwords."""
    return set(stopwords.words("english"))

# Call this at the top of your script/pipeline
web = load_spacy_model()
stemmer = PorterStemmer()

# Extracting entities to keep them safe from being stripped or stemmed.

def extract_entities(doc) -> list:
    entities = []
    for ent in doc.ents:
        entities.append({"Entity": ent.text, "Label": ent.label_})
    return entities

#Tokenizing the text into words and sentences.

def tokenize(text: str) -> dict:
    return {
        "sentences": sent_tokenize(text),
        "words": word_tokenize(text),
    }

#Entity span tracking - protects words by POSITION, not spelling.

def get_entity_spans(doc) -> set[int]:
    """Return the set of character indices that fall inside any entity."""
    spans = set()
    for ent in doc.ents:
        spans.update(range(ent.start_char, ent.end_char))
    return spans

def is_protected(token, entity_spans: set[int]) -> bool:
    """Check whether a spaCy token overlaps an entity span, by position."""
    return any(
        i in entity_spans for i in range(token.idx, token.idx + len(token.text))
    )

#Removing stopwords + stemming, done together since stemming operates on
#the filtered set. Both use entity_spans (character positions) instead of
#matching word strings, so a stopword like "of" is only protected when it
#sits inside an actual entity span (e.g. "Theory of Relativity"), not
#everywhere it appears in the text.

def process_tokens(doc, entity_spans: set[int]) -> tuple[list[str], list[str]]:
    stop_words = get_stopwords()
    filtered = []
    stemmed = []
    for token in doc:
        if token.is_space:
            continue
        protected = is_protected(token, entity_spans)
        lower = token.text.lower()
        if protected:
            filtered.append(token.text)
            stemmed.append(token.text)
        elif lower not in stop_words and token.text.isalpha():
            filtered.append(lower)
            stemmed.append(stemmer.stem(lower))
    return filtered, stemmed

#Lemmatization.

def lemmatize(doc, entity_spans: set[int]) -> list[str]:
    lemmas = []
    for token in doc:
        if not token.is_alpha:
            continue
        if is_protected(token, entity_spans):
            lemmas.append(token.text)
        else:
            lemmas.append(token.lemma_.lower())
    return lemmas

# Part of Speech tagging.
def pos_tag(doc) -> list[dict]:
    return [{"text": token.text, "pos": token.pos_} for token in doc]

# ORCHESTRATION

def run_pipeline(text: str) -> dict:

    ensure_nltk_resources()

    # Parse text once with spaCy and reuse the doc
    doc = web(text)
    
    # Extract entities and their character spans
    entities_list = extract_entities(doc)
    entity_spans = get_entity_spans(doc)

    token_data = tokenize(text)
    words = token_data["words"]
    filtered_tokens, stemmed_tokens = process_tokens(doc, entity_spans)
    lemmatized_tokens = lemmatize(doc, entity_spans)
    pos_tags = pos_tag(doc)

    return {
        "original_text": text,
        "entities": entities_list,
        "sentences": token_data["sentences"],
        "tokens": words,
        "filtered_tokens": filtered_tokens,
        "stemmed_tokens": stemmed_tokens,
        "lemmatized_tokens": lemmatized_tokens,
        "pos_tags": pos_tags,}