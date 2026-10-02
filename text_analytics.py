"""
Group 14 - Text Analytics Project: News Topic Modelling
=========================================================
Dataset: 20 Newsgroups (sklearn.datasets.fetch_20newsgroups)
Representation: TF-IDF
Models: LDA (Model 1) and NMF (Model 2)
Evaluation: topic coherence, topic diversity, topic interpretability (purity)

This file holds all of the text analytics steps. app.py only displays the results.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
import re
from collections import Counter
from itertools import combinations
import numpy as np
import pandas as pd
from wordcloud import WordCloud

import nltk  # natural language toolkit
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

# ---------------------------------------------------------------------------
# GLOBAL SETTINGS
# ---------------------------------------------------------------------------
N_TOPICS = 15        # number of topics to discover; identical for both models for a fair comparison
N_TOP_WORDS = 10     # number of top words shown for every topic
RANDOM_STATE = 42    # reproducible results across runs
UNASSIGNED = -1      # label for documents that contain none of the model's words

# ---------------------------------------------------------------------------
# NLTK SETUP
# ---------------------------------------------------------------------------
# Download the required NLTK packages, but only if they are missing
for nltk_path, nltk_package in [("tokenizers/punkt_tab", "punkt_tab"),
                                ("corpora/stopwords", "stopwords"),
                                ("corpora/wordnet", "wordnet")]:
    try:
        nltk.data.find(nltk_path)
    except LookupError:
        nltk.download(nltk_package, quiet=True)

# Standard English stopwords, extended with project-specific "noise" words
# that survived NLTK's default list but showed up dominating topics during EDA.
CUSTOM_STOPWORDS = {"would", "one", "get", "like", "know", "dont", "also", "think",
                    "could", "im", "thing", "say", "well", "may", "make", "use",
                    "used", "using"}
stop_words = set(stopwords.words("english")).union(CUSTOM_STOPWORDS)

lemmatizer = WordNetLemmatizer()


# ---------------------------------------------------------------------------
# STEP 2: REGEX CLEANING
# ---------------------------------------------------------------------------
# Cleans raw text before tokenization: lowercase, strip emails/URLs/HTML/
# punctuation/underscores/digits, and normalize whitespace.
def clean_text(texts):
    cleaned = []
    for text in texts:
        text = text.lower()
        text = re.sub(r'\S+@\S+', '', text)              # Remove emails
        text = re.sub(r'http\S+|www\.\S+', '', text)     # Remove URLs
        text = re.sub(r'<.*?>', '', text)                 # Remove HTML
        text = re.sub(r'[^\w\s]', '', text)               # Remove punctuation
        text = re.sub(r'_+', ' ', text)                   # Remove leftover underscore runs (old-style signature separators)
        text = re.sub(r'\d+', '', text)                   # Remove digits
        text = re.sub(r'\s+', ' ', text).strip()          # Normalize whitespace
        cleaned.append(text)
    return cleaned


# Runs every preprocessing step on ONE piece of text (used by the "Try It Yourself" page)
def preprocess_one_document(raw_text):
    cleaned = clean_text([raw_text])[0]
    tokens = word_tokenize(cleaned)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    tokens = [w for w in tokens if 2 < len(w) <= 20]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# STEPS 1-3: GET DATA & TEXT PREPROCESSING (tokenize -> stopwords -> lemmatize -> dedupe)
# ---------------------------------------------------------------------------
# Fetch the 20 Newsgroups dataset directly from scikit-learn (no manual download needed).
# remove=(...) strips email headers, signatures, and quoted-reply text so the model
# doesn't pick up boilerplate instead of actual topical content.
def load_and_prepare_data():
    data = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
    )

    # One row per document: original text, numeric label, and category name.
    # The original text stays in the same row as its cleaned versions, so the two
    # can never get out of step when duplicate rows are dropped later.
    df_clean = pd.DataFrame({
        "raw_text": data.data,
        "label_id": data.target,
        "label_name": [data.target_names[i] for i in data.target],
    })

    # Drop any documents that ended up empty after stripping headers/footers/quotes
    df_clean["raw_text"] = df_clean["raw_text"].str.strip()
    df_clean = df_clean[df_clean["raw_text"].str.len() > 0].reset_index(drop=True)
    n_before = len(df_clean)

    # Regex cleaning
    df_clean["cleaned_text"] = clean_text(df_clean["raw_text"])

    # Tokenise each document separately
    df_clean["tokens"] = df_clean["cleaned_text"].apply(word_tokenize)

    # Remove stopwords from each document's token list
    df_clean["filtered_tokens"] = df_clean["tokens"].apply(
        lambda tokens: [word for word in tokens if word not in stop_words]
    )

    # Lemmatize: reduce words to their dictionary form (e.g. "pens" -> "pen")
    df_clean["lemmatized_tokens"] = df_clean["filtered_tokens"].apply(
        lambda tokens: [lemmatizer.lemmatize(word) for word in tokens]
    )

    # Drop very short tokens (stray letters) AND implausibly long ones (corrupted strings)
    df_clean["lemmatized_tokens"] = df_clean["lemmatized_tokens"].apply(
        lambda tokens: [word for word in tokens if 2 < len(word) <= 20]
    )

    # Rejoin each token list into a single string (the format TfidfVectorizer expects)
    df_clean["preprocessed_text"] = df_clean["lemmatized_tokens"].apply(lambda tokens: " ".join(tokens))

    # Drop exact-duplicate documents (cross-posted messages formed their own fake topic)
    df_clean = df_clean.drop_duplicates(subset="preprocessed_text").reset_index(drop=True)
    n_duplicates_dropped = n_before - len(df_clean)

    # Document length (in tokens), used in the exploratory analysis
    df_clean["doc_length"] = df_clean["lemmatized_tokens"].apply(len)

    return df_clean, n_duplicates_dropped


# ---------------------------------------------------------------------------
# STEPS 5-7: MODEL BUILDING (TF-IDF, LDA, NMF)
# ---------------------------------------------------------------------------
def build_models(documents):
    # STEP 5: TEXT REPRESENTATION (TF-IDF)
    # No train/test split here - topic modelling (LDA/NMF) is unsupervised, so
    # the models are fit on the entire cleaned corpus at once.
    tfidf = TfidfVectorizer(
        max_df=0.90,       # ignore words that appear in >90% of documents (too common, uninformative)
        min_df=5,          # ignore words that appear in fewer than 5 documents (too rare, noise)
        max_features=5000  # cap vocabulary size to the 5000 most informative words
    )
    tfidf_matrix = tfidf.fit_transform(documents)

    # STEP 6: BUILD MODEL 1 - LDA (Latent Dirichlet Allocation)
    # n_jobs=1 on purpose: with n_jobs=-1 the LDA result changes with the number of CPU
    # cores, so a laptop, the cloud server and the report would all show different topics.
    lda_model = LatentDirichletAllocation(
        n_components=N_TOPICS,
        random_state=RANDOM_STATE,
        max_iter=20,       # number of passes LDA makes while refining its topics
        n_jobs=1
    )
    # fit_transform trains LDA on the TF-IDF matrix and returns, for every
    # document, how strongly it belongs to each of the topics.
    lda_topics = lda_model.fit_transform(tfidf_matrix)

    # STEP 7: BUILD MODEL 2 - NMF (Non-negative Matrix Factorization)
    nmf_model = NMF(
        n_components=N_TOPICS,   # same as LDA so the comparison is fair
        random_state=RANDOM_STATE,
        max_iter=500,            # NMF typically needs more passes than LDA to stabilize
        init="nndsvd"            # smarter, more consistent starting point than random init
    )
    nmf_topics = nmf_model.fit_transform(tfidf_matrix)

    return tfidf, tfidf_matrix, lda_model, lda_topics, nmf_model, nmf_topics


# Assign each document to its single best-fitting (highest-scoring) topic.
# A document with no model words has an all-zero TF-IDF row; argmax of zeros is 0,
# which would silently dump it into topic 0, so those documents are marked UNASSIGNED.
def assign_topics(doc_topic_matrix, tfidf_matrix):
    topics = doc_topic_matrix.argmax(axis=1)
    topics[np.asarray(tfidf_matrix.getnnz(axis=1)) == 0] = UNASSIGNED
    return topics


# Top N highest-scoring words for every topic (works for both LDA and NMF,
# since both store their topic-word scores in `.components_`).
def get_topic_top_words(model, feature_names, n_top_words=N_TOP_WORDS):
    topic_word_lists = []
    for topic in model.components_:
        top_indices = topic.argsort()[-n_top_words:][::-1]
        topic_word_lists.append([feature_names[i] for i in top_indices])
    return topic_word_lists


# ---------------------------------------------------------------------------
# EVALUATION METRICS
# ---------------------------------------------------------------------------
# For every word we score, find the set of documents that contain it.
# Built once, so coherence does not re-read all documents for every word pair.
def build_doc_index(tokenized_docs, words):
    wanted = set(words)
    index = {word: set() for word in wanted}
    for doc_id, doc in enumerate(tokenized_docs):
        for word in wanted.intersection(doc):
            index[word].add(doc_id)
    return index


# UMass-style coherence: for each pair of top words in a topic, measures how often
# they co-occur across documents (log-scaled), then averages over all word pairs.
# Values closer to 0 (less negative) indicate a more coherent topic.
# standard=False: project formula, divides by the document count of the LOWER-ranked word.
# standard=True:  textbook UMass (Mimno et al., 2011), divides by the HIGHER-ranked word.
def compute_umass_coherence(topic_words, doc_index, standard=False):
    scores = []
    for w1, w2 in combinations(topic_words, 2):
        d_divisor = len(doc_index[w1]) if standard else len(doc_index[w2])
        d_w1_w2 = len(doc_index[w1] & doc_index[w2])
        score = np.log((d_w1_w2 + 1) / d_divisor) if d_divisor > 0 else 0
        scores.append(score)
    return float(np.mean(scores)) if scores else 0.0


# Share of unique words across all topics' top words (1.0 = no word is repeated)
def compute_topic_diversity(topic_word_lists):
    all_words = [w for topic in topic_word_lists for w in topic]
    return len(set(all_words)) / len(all_words)


# Purity: for each topic, the share of its documents that come from ONE real category.
# The real labels are used only to check the results, never to train the models.
def compute_topic_purity(df, topic_col, label_col):
    df = df[df[topic_col] != UNASSIGNED]   # documents with no topic are left out
    total_docs = len(df)
    total_majority = 0
    topic_summary = []
    for topic_num in sorted(df[topic_col].unique()):
        subset = df[df[topic_col] == topic_num]
        label_counts = subset[label_col].value_counts()
        majority_label = label_counts.idxmax()
        majority_count = label_counts.max()
        purity = majority_count / len(subset)
        total_majority += majority_count
        topic_summary.append({
            "topic": int(topic_num),
            "dominant_category": majority_label,
            "purity": round(float(purity), 3),
            "num_docs": int(len(subset))
        })
    return topic_summary, total_majority / total_docs


def evaluate_models(lda_topic_words, nmf_topic_words, tokenized_docs, df_clean):
    all_top_words = [w for topic in lda_topic_words + nmf_topic_words for w in topic]
    doc_index = build_doc_index(tokenized_docs, all_top_words)

    # df_clean already carries "lda_topic", "nmf_topic" and "label_name" (the real newsgroup)
    lda_purity_summary, lda_purity = compute_topic_purity(df_clean, "lda_topic", "label_name")
    nmf_purity_summary, nmf_purity = compute_topic_purity(df_clean, "nmf_topic", "label_name")

    return {
        "lda_coherence": [compute_umass_coherence(w, doc_index) for w in lda_topic_words],
        "nmf_coherence": [compute_umass_coherence(w, doc_index) for w in nmf_topic_words],
        "lda_coherence_std": [compute_umass_coherence(w, doc_index, standard=True) for w in lda_topic_words],
        "nmf_coherence_std": [compute_umass_coherence(w, doc_index, standard=True) for w in nmf_topic_words],
        "lda_diversity": compute_topic_diversity(lda_topic_words),
        "nmf_diversity": compute_topic_diversity(nmf_topic_words),
        "lda_purity": lda_purity,
        "nmf_purity": nmf_purity,
        "lda_purity_summary": lda_purity_summary,
        "nmf_purity_summary": nmf_purity_summary,
    }


def build_wordcloud(tokens_series):
    all_tokens = [w for tokens in tokens_series for w in tokens]
    word_freq = Counter(all_tokens)
    wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)
    return wc, word_freq


# ---------------------------------------------------------------------------
# RUN THE WHOLE PIPELINE (app.py calls this once and caches the result)
# ---------------------------------------------------------------------------
def run_pipeline():
    df_clean, n_duplicates_dropped = load_and_prepare_data()

    (tfidf, tfidf_matrix, lda_model, lda_topics,
     nmf_model, nmf_topics) = build_models(df_clean["preprocessed_text"])

    # Assign every document to its best topic under each model
    df_clean["lda_topic"] = assign_topics(lda_topics, tfidf_matrix)
    df_clean["nmf_topic"] = assign_topics(nmf_topics, tfidf_matrix)

    feature_names = tfidf.get_feature_names_out()
    lda_topic_words = get_topic_top_words(lda_model, feature_names)
    nmf_topic_words = get_topic_top_words(nmf_model, feature_names)

    results = evaluate_models(lda_topic_words, nmf_topic_words,
                              df_clean["lemmatized_tokens"], df_clean)

    return (df_clean, n_duplicates_dropped, tfidf, lda_model, nmf_model,
            lda_topic_words, nmf_topic_words, results)
