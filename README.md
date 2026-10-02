# Group 14: News Topic Modelling (MSc Business Analytics, Text Analytics)

Unsupervised topic modelling of the **20 Newsgroups** dataset (about 18,000 forum posts) using
**TF-IDF** with two models, **LDA** and **NMF**, evaluated on topic **coherence**, **diversity**
and **purity**. An interactive **Streamlit** app lets you explore the data, compare the models and
predict the topic of your own text.

**Live app:** 

## Files

| File | What it does |
|------|--------------|
| `app.py` | The Streamlit website (7 pages). |
| `text_analytics.py` | All the logic: cleaning, TF-IDF, LDA, NMF, evaluation. |
| `requirements.txt` | Exact library versions the project was tested with. |

## Run it on your own computer

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first start takes a few minutes: it downloads the dataset and trains both models.
After that, every click is instant.

## Pages

Home, Corpora Viewer, Text Preprocessing, Exploratory Text Analytics, Topic Explorer,
Model Evaluation, Try It Yourself.

## Team (Group 14)

Fianu Adze Kofi Dovlo, Nana Ama Banafo, Stephen Ofori-Boateng, Kwame Ababio,
Obiri-Yeboah Boateng Kwame.
