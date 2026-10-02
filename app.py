# Import packages
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import text_analytics as ta

# AT Global level
# Page setup
st.set_page_config(
    page_title="Group 14 - News Topic Modelling",
    page_icon="📰",
    layout="wide"
)

# Pipeline Caching - So app doesn't load and re-run the entire pipeline every time a page is selected
@st.cache_resource(show_spinner="Running Text Analytics Pipeline (Loading, Modeling, Evaluating)...")
def get_pipeline_data():
    return ta.run_pipeline()

@st.cache_resource(show_spinner="Building word cloud...")
def get_wordcloud_cached():
    return ta.build_wordcloud(df_clean["lemmatized_tokens"])

# Unpack cached results
(df_clean, n_duplicates_dropped, tfidf, lda_model, nmf_model,
 lda_topic_words, nmf_topic_words, results) = get_pipeline_data()

N_TOPICS = ta.N_TOPICS

# Show a matplotlib figure, then close it to free memory (Streamlit re-runs this file on every click)
def show_figure(fig):
    st.pyplot(fig)
    plt.close(fig)

# PAGE 1 — HOME
def page1():
    st.title("📰 Group 14 - News Topic Modelling")
    st.markdown("""
    **MSc Business Analytics - Text Analytics Group Project**

    This app applies unsupervised topic modelling to the **20 Newsgroups** dataset.

    - **Representation:** TF-IDF
    - **Models:** Latent Dirichlet Allocation (LDA) and Non-negative Matrix Factorization (NMF)
    - **Evaluation:** Topic coherence, topic diversity, topic interpretability
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Documents (after cleaning)", f"{len(df_clean):,}")
    col2.metric("Real categories", df_clean["label_name"].nunique())
    col3.metric("Duplicate documents removed", n_duplicates_dropped)
    st.info("Use the sidebar to explore the data, the models, and try the topic predictor.")

# PAGE 2 — CORPORA VIEWER
def page2():
    st.subheader("Corpora Viewer")

    if st.checkbox("Show raw documents (before cleaning)"):
        st.dataframe(df_clean[["raw_text", "label_name"]].head(200), use_container_width=True)

    if st.checkbox("Show cleaned / preprocessed documents"):
        st.dataframe(df_clean[["preprocessed_text", "label_name"]].head(200), use_container_width=True)

    if st.checkbox("Show category counts"):
        st.bar_chart(df_clean["label_name"].value_counts())

# PAGE 3 — TEXT PREPROCESSING
def page3():
    st.subheader("Text Preprocessing Pipeline")
    idx = st.slider("Pick a document to inspect", 0, len(df_clean) - 1, 0)

    # 1. Original text
    st.markdown("### 1. Original text")
    st.write(df_clean.loc[idx, "raw_text"][:500])

    # 2. After regex cleaning
    st.markdown("### 2. After regex cleaning")
    st.write(df_clean.loc[idx, "cleaned_text"][:500])

    # 3. Tokenised
    st.markdown("### 3. Tokenisation")
    st.write(df_clean.loc[idx, "tokens"][:20])

    # 4. Stopword removal
    st.markdown("### 4. Stopword removal")
    st.write(df_clean.loc[idx, "filtered_tokens"][:20])

    # 5. Lemmatisation
    st.markdown("### 5. Lemmatisation")
    st.write(df_clean.loc[idx, "lemmatized_tokens"][:20])

    # 6. Final preprocessed text
    st.markdown("### 6. Final preprocessed text")
    st.write(df_clean.loc[idx, "preprocessed_text"][:500])

# PAGE 4 — EXPLORATORY TEXT ANALYTICS
def page4():
    st.subheader("Exploratory Text Analytics")

    # Document length distribution
    st.markdown("### Document length (tokens per document)")
    st.write(df_clean["doc_length"].describe())

    fig1, ax1 = plt.subplots()
    ax1.hist(df_clean["doc_length"], bins=50)
    ax1.set_xlabel("Tokens per document")
    ax1.set_ylabel("Number of documents")
    show_figure(fig1)

    wc, word_freq = get_wordcloud_cached()
    top_20 = word_freq.most_common(20)

    # Top 20 most frequent words
    st.markdown("### Top 20 most frequent words (whole corpus)")
    words, counts = zip(*top_20)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.barh(words, counts)
    ax2.invert_yaxis()
    ax2.set_xlabel("Frequency")
    show_figure(fig2)

    # Word cloud
    st.markdown("### Word cloud (whole corpus)")
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.imshow(wc, interpolation="bilinear")
    ax3.axis("off")
    show_figure(fig3)

    st.markdown(f"**Vocabulary size:** {len(word_freq):,} unique words")

# PAGE 5 — TOPIC EXPLORER
def page5():
    st.subheader("Topic Explorer")
    model_choice = st.radio("Choose a model", ["LDA", "NMF"], horizontal=True)

    topic_words = lda_topic_words if model_choice == "LDA" else nmf_topic_words
    topic_col = "lda_topic" if model_choice == "LDA" else "nmf_topic"
    topic_num = st.selectbox("Choose a topic", list(range(N_TOPICS)))

    st.markdown(f"### Top words for {model_choice} Topic {topic_num}:")
    st.write(", ".join(topic_words[topic_num]))

    subset = df_clean[df_clean[topic_col] == topic_num]
    st.markdown(f"**{len(subset)} documents assigned to this topic**")
    st.bar_chart(subset["label_name"].value_counts())

    st.markdown("### Sample documents assigned to this topic:")
    st.dataframe(subset[["preprocessed_text", "label_name"]].head(10), use_container_width=True)

# PAGE 6 — MODEL EVALUATION
def page6():
    st.subheader("Model Evaluation: Coherence, Diversity, Interpretability")

    # Overall comparison
    st.markdown("### Overall comparison")
    comparison_df = pd.DataFrame({
        "Metric": ["Average topic coherence - project formula (closer to 0 = better)",
                   "Average topic coherence - standard UMass (closer to 0 = better)",
                   "Topic diversity (higher = better)",
                   "Overall topic purity (higher = better)"],
        "LDA": [round(float(np.mean(results["lda_coherence"])), 4),
                round(float(np.mean(results["lda_coherence_std"])), 4),
                round(results["lda_diversity"], 4),
                round(results["lda_purity"], 4)],
        "NMF": [round(float(np.mean(results["nmf_coherence"])), 4),
                round(float(np.mean(results["nmf_coherence_std"])), 4),
                round(results["nmf_diversity"], 4),
                round(results["nmf_purity"], 4)],
    })
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    n_unassigned = int((df_clean["lda_topic"] == ta.UNASSIGNED).sum())
    st.caption(f"{n_unassigned} documents contain none of the 5,000 model words, so they are left out "
               "of purity. The project formula divides by the lower-ranked word of each pair; the "
               "standard formula (Mimno et al., 2011) divides by the higher-ranked word.")

    # Per-topic coherence
    st.markdown("### Per-topic coherence scores")
    model_choice = st.radio("Model", ["LDA", "NMF"], horizontal=True, key="coherence_model")
    coherence_scores = results["lda_coherence"] if model_choice == "LDA" else results["nmf_coherence"]
    topic_words = lda_topic_words if model_choice == "LDA" else nmf_topic_words

    coherence_df = pd.DataFrame({
        "Topic": list(range(N_TOPICS)),
        "Coherence": [round(c, 4) for c in coherence_scores],
    })
    st.bar_chart(coherence_df.set_index("Topic"))

    # Topic purity detail
    st.markdown("### Topic purity detail (interpretability proxy)")
    purity_summary = results["lda_purity_summary"] if model_choice == "LDA" else results["nmf_purity_summary"]
    purity_df = pd.DataFrame(purity_summary)
    purity_df["top_words"] = [", ".join(topic_words[t][:6]) for t in purity_df["topic"]]
    st.dataframe(purity_df, use_container_width=True, hide_index=True)

# PAGE 7 — TRY IT YOURSELF
def page7():
    st.subheader("Try It Yourself")

    # Text input
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    user_input = st.text_area("Enter text for prediction:", key="user_input", height=150)
    file_upload = st.file_uploader("Or upload a text file", type=["txt"])

    # File input
    file_content = ""
    if file_upload is not None:
        try:
            file_content = file_upload.read().decode("utf-8")
            st.success("File uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

    # File content takes priority over text input
    final_input = file_content if file_content else user_input

    # Create a row with two button columns and an empty spacer
    col1, col2, col3 = st.columns([1, 1, 4])

    # Predict button
    with col1:
        predict_button = st.button("Predict Topic", use_container_width=True)

    # Clear button (uses a callback, because Streamlit does not allow the text box
    # value to be changed after the text box has been drawn on the page)
    with col2:
        def clear_input():
            st.session_state.user_input = ""
        st.button("Clear", use_container_width=True, on_click=clear_input)

    # Empty space
    with col3:
        st.empty()

    # Prediction Logic
    if predict_button:
        if not final_input.strip():
            st.warning("Please enter text or upload a file first!")
        else:
            try:
                # Clean the text exactly like the training documents, then convert to TF-IDF features
                processed = ta.preprocess_one_document(final_input)
                user_input_tfidf = tfidf.transform([processed])

                # No usable words means no prediction (otherwise the answer would always be Topic 0)
                if user_input_tfidf.nnz == 0:
                    st.warning("None of the words in this text are in the models' 5,000-word "
                               "vocabulary, so no topic can be predicted.")
                else:
                    lda_pred = int(lda_model.transform(user_input_tfidf)[0].argmax())
                    nmf_pred = int(nmf_model.transform(user_input_tfidf)[0].argmax())

                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.markdown(f"### LDA prediction: Topic {lda_pred}")
                        st.write(", ".join(lda_topic_words[lda_pred]))
                    with res_col2:
                        st.markdown(f"### NMF prediction: Topic {nmf_pred}")
                        st.write(", ".join(nmf_topic_words[nmf_pred]))

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

# SIDEBAR NAVIGATION
pages = {
    "Home": page1,
    "Corpora Viewer": page2,
    "Text Preprocessing": page3,
    "Exploratory Text Analytics": page4,
    "Topic Explorer": page5,
    "Model Evaluation": page6,
    "Try It Yourself": page7,
}

select_page = st.sidebar.selectbox("Select a page", list(pages.keys()))
pages[select_page]()
