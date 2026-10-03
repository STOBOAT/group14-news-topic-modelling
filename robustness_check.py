"""
Group 14 - Robustness check: is "NMF beats LDA" a fluke?
=========================================================
LDA is a randomised algorithm, so this script trains it several times (different random
seeds, plus seed 42 with four parallel workers) and scores every run exactly like the app does
(coherence, diversity, purity). NMF (nndsvd start) is deterministic, so it is trained once.

Usage:   python robustness_check.py            (full corpus; takes about 15 to 20 minutes)
         python robustness_check.py --sample 3000   (quick smoke test on 3,000 posts)
"""
import sys
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import TfidfVectorizer

import text_analytics as ta

# Optional: score only a random sample of posts (for a quick test)
sample_size = int(sys.argv[sys.argv.index("--sample") + 1]) if "--sample" in sys.argv else None

df_clean, n_duplicates_dropped = ta.load_and_prepare_data()
if sample_size:
    df_clean = df_clean.sample(sample_size, random_state=0).reset_index(drop=True)

# Same TF-IDF settings as ta.build_models
tfidf = TfidfVectorizer(max_df=0.90, min_df=5, max_features=5000)
tfidf_matrix = tfidf.fit_transform(df_clean["preprocessed_text"])
feature_names = tfidf.get_feature_names_out()


# Train one model, then score it with the same functions the app uses
def score_model(model):
    doc_topic = model.fit_transform(tfidf_matrix)
    work = df_clean.copy()
    work["topic"] = ta.assign_topics(doc_topic, tfidf_matrix)
    topic_words = ta.get_topic_top_words(model, feature_names)
    doc_index = ta.build_doc_index(work["lemmatized_tokens"], [w for topic in topic_words for w in topic])
    coherence = np.mean([ta.compute_umass_coherence(words, doc_index) for words in topic_words])
    diversity = ta.compute_topic_diversity(topic_words)
    _, purity = ta.compute_topic_purity(work, "topic", "label_name")
    return coherence, diversity, purity


runs = [("NMF", NMF(n_components=ta.N_TOPICS, random_state=ta.RANDOM_STATE, max_iter=500, init="nndsvd")),
        ("LDA seed 42, 1 worker (the app)", LatentDirichletAllocation(n_components=ta.N_TOPICS, random_state=42, max_iter=20, n_jobs=1)),
        ("LDA seed 42, 4 workers", LatentDirichletAllocation(n_components=ta.N_TOPICS, random_state=42, max_iter=20, n_jobs=4))]
runs += [(f"LDA seed {seed}", LatentDirichletAllocation(n_components=ta.N_TOPICS, random_state=seed, max_iter=20, n_jobs=-1))
         for seed in (0, 1, 2, 3, 4)]

print(f"\n{'Run':36s} {'Coherence':>10s} {'Diversity':>10s} {'Purity':>8s}")
results = {}
for name, model in runs:
    results[name] = score_model(model)
    c, d, p = results[name]
    print(f"{name:36s} {c:10.3f} {d:10.3f} {p:8.3f}", flush=True)

# Summary: how often does NMF beat LDA?
nmf_c, nmf_d, nmf_p = results["NMF"]
lda_runs = [v for k, v in results.items() if k.startswith("LDA")]
print(f"\nNMF has better coherence than LDA in {sum(nmf_c > c for c, _, _ in lda_runs)} of {len(lda_runs)} LDA runs")
print(f"NMF has better purity than LDA in    {sum(nmf_p > p for _, _, p in lda_runs)} of {len(lda_runs)} LDA runs")
print(f"LDA diversity matches or beats NMF in {sum(d >= nmf_d - 0.0005 for _, d, _ in lda_runs)} of {len(lda_runs)} LDA runs")
