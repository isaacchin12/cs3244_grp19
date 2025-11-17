import pandas as pd
import re
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

df_reviews = pd.read_json('/SFS/project/ry/dp_sgteam/catherine/ada/dataset/IMDB_reviews.json', lines=True)

print(f"Dataset shape: {df_reviews.shape}")
df_reviews.head()

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

df_reviews['review_text'] = df_reviews['review_text'].apply(preprocess_lemma)
df_reviews['is_spoiler'] = df_reviews['is_spoiler'].astype(int)

print("Done Cleaning")

df_reviews.to_json("/SFS/project/ry/dp_sgteam/catherine/ada/dataset/cleaned_data.json",
                   orient = 'records', indent = 4)

print("Done storing")