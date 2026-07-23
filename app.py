import json
import streamlit as st
from pipeline import run_pipeline

# --- Run pipeline results ---
def input_process(text):
    """Run pipeline on input text."""
    return run_pipeline(text)

st.set_page_config(page_title="StripThatNoise")

st.title("StripThatNoise: Entity-Aware NLP Preprocessor")
st.caption("Raw text files are full of noise that clutters LLM context windows and skews embeddings. StripThatNoise takes your text, strips out fluff using smart tokenization, stopword filtering, stemming, lemmatization, and locks down entities so you don't lose key context. Once processed, it exports your sanitized, structured tokens straight into a clean JSON file for downstream pipelines.")

# --- Sidebar Navigation ---
st.sidebar.markdown(f"<h1 style='font-size: 28px; margin-top: 0;'>Index</h1>", unsafe_allow_html=True)

# --- Input ---
input_mode = st.radio("Input method:", ["Type/paste text", "Upload a .txt file"])

text = ""
if input_mode == "Type/paste text":
    text = st.text_area("Enter text:", height=150, placeholder="Paste your text here...")
else:
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")

# --- Run pipeline ---
if st.button("Run"):
    st.session_state["_ran"] = True
    if text.strip():
        with st.spinner("Processing..."):
            result = input_process(text)

        sections = [
            "Original Text",
            "Entities Extracted",
            "Tokens",
            "Stopword Removal",
            "Stemmed Tokens",
            "Lemmatized Tokens",
            "POS Tags"
        ]

        st.sidebar.markdown("---")
        for section in sections:
            st.sidebar.markdown(f"[{section}](#{section.lower().replace(' ', '-')})")
        st.sidebar.markdown("---")

        st.markdown(f"<h2 id='original-text'>Original Text</h2>", unsafe_allow_html=True)
        st.write(result["original_text"])

        st.markdown(f"<h2 id='entities-extracted'>Entities Extracted</h2>", unsafe_allow_html=True)
        if result["entities"]:
            st.table(result["entities"])
        else:
            st.write("No entities found.")

        st.markdown(f"<h2 id='tokens'>Tokens</h2>", unsafe_allow_html=True)
        st.write(result["tokens"])

        st.markdown(f"<h2 id='stopword-removal'>Stopword Removal</h2>", unsafe_allow_html=True)
        st.write(result["filtered_tokens"])

        st.markdown(f"<h2 id='stemmed-tokens'>Stemmed Tokens</h2>", unsafe_allow_html=True)
        st.write(result["stemmed_tokens"])

        st.markdown(f"<h2 id='lemmatized-tokens'>Lemmatized Tokens</h2>", unsafe_allow_html=True)
        st.write(result["lemmatized_tokens"])

        st.markdown(f"<h2 id='pos-tags'>POS Tags</h2>", unsafe_allow_html=True)
        st.table(result["pos_tags"])

        # --- Download JSON ---
        st.markdown("---")
        json_output = json.dumps(result, indent=2)
        st.download_button(
            label="⬇️ Download results as JSON",
            data=json_output,
            file_name="preprocessedtext.json",
            mime="application/json",
        )
elif text.strip() == "" and st.session_state.get("_ran", False):
    st.warning("Please enter or upload some text first.")

# --- Footnote ---
st.divider()
st.markdown("*This is a lightweight project built for standard .txt preprocessing with known limitations.*")