import pandas as pd
import re
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

import os
from dotenv import load_dotenv

load_dotenv()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN 

def preprocess_lemma(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)

    tokens = text.split()
    tokens = [tok for tok in tokens if tok not in stop_words]
    tagged = pos_tag(tokens)

    lemmas = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged]
    return " ".join(lemmas)

df_reviews = pd.read_json(f'{os.environ["DIR_PATH"]}/dataset/IMDB_reviews.json', lines=True)
df_reviews = df_reviews[['user_id', 'movie_id', 'review_text', 'is_spoiler']]
df_reviews['review_text'] = df_reviews['review_text'].apply(preprocess_lemma)
df_reviews['is_spoiler'] = df_reviews['is_spoiler'].astype(int)

df_movie_details = pd.read_json(f'{os.environ["DIR_PATH"]}/dataset/IMDB_movie_details.json', lines=True)
df_movie_details = df_movie_details[['plot_synopsis', 'plot_summary', 'movie_id']]
df_movie_details['plot_summary'] = df_movie_details['plot_summary'].apply(preprocess_lemma)
df_movie_details['plot_synopsis'] = df_movie_details['plot_synopsis'].apply(preprocess_lemma)

df = pd.merge(df_reviews, df_movie_details, on='movie_id', how='left')
print("Done Cleaning")
print(df.head())

df.to_json(f'{os.environ["DIR_PATH"]}/dataset/cleaned_data.json',
                   orient = 'records', indent = 4)
print("Done storing")



