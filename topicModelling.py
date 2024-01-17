import os
import re
import gensim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from wordcloud import WordCloud
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from nltk.stem import WordNetLemmatizer
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import nltk
nltk.download('wordnet')

# Used to get similar results each time the code is ran
np.random.seed(2018)

# The following functions cleans the data by removing emojis, urls, mentions (@) and hashtags (#)
def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

def remove_url(text):
    return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

def remove_mentions(text):
    return re.sub(r'@\w+', '', text)

def remove_hashtags(text):
    return re.sub(r'#\w+', '', text)

# The following functions lemmatize and pre process the text
lemmatizer = WordNetLemmatizer()

def lemmatize(text):
    return lemmatizer.lemmatize(text, pos='v')

def preprocess(text):
    result = []
    for token in simple_preprocess(text):
        if token not in STOPWORDS and len(token) > 3:
            result.append(lemmatize(token))
    return result

# This function reads the files by using read_csv or read_excel depending on the file type
def read_file(file_path, file_type="csv"):
    if file_type == "csv":
        data = pd.read_csv(file_path)
    elif file_type == "xlsx":
        data = pd.read_excel(file_path)
    else:
        raise ValueError("The file type entered is invalid. Use 'csv' or 'xlsx'.")
    return data

# The following functions create and save the topic modelling visualisations applied to the data
def plot_wordcloud(lda_model, topic_id):
    terms = lda_model.show_topic(topic_id, topn=30)
    terms_dict = {term: freq for term, freq in terms}
    file_path = os.path.join("TMdata")
    rc('font', family='Times New Roman', weight='bold')
    # Checks if the folder exists, if it doesn't, it makes the directory
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    wordcloud = WordCloud(background_color='white', width=400, height=400)
    wordcloud.generate_from_frequencies(terms_dict)

    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Topic {topic_id + 1}', fontweight='bold')
    plt.savefig(os.path.join(file_path, f'topic_{topic_id + 1}.png'))

def visualize_lda(lda_model, corpus, dictionary):
    lda_display = gensimvis.prepare(lda_model, corpus, dictionary, sort_topics=False)
    file_name = "LDA.html"
    file_path = os.path.join("TMdata", file_name)
    pyLDAvis.save_html(lda_display, file_path)

def main():
    # File paths for the Twitter data
    file_paths = [    
        ('TwitterData/from_premierleague2022.xlsx', "xlsx"),    
        ('TwitterData/to_premierleague2022.xlsx', "xlsx"),    
        ('TwitterData/mentions_premierleague2022.csv', "csv"),
    ]

    # Applies all the text cleaning and pre-processing functions
    processed_data = []

    for file_path, file_type in file_paths:
        data = read_file(file_path, file_type)
        data['tweet_text'] = data['tweet_text'].astype(str)
        data['tweet_text'] = data['tweet_text'].apply(remove_emoji)
        data['tweet_text'] = data['tweet_text'].apply(remove_url)
        data['tweet_text'] = data['tweet_text'].apply(remove_mentions)
        data['tweet_text'] = data['tweet_text'].apply(remove_hashtags)
        data['processed_tweet'] = data['tweet_text'].apply(preprocess)
        processed_data.extend(data['processed_tweet'])

    # Creates a dictionary for the processed data
    dictionary = gensim.corpora.Dictionary(processed_data)
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

    # Generates a bag-of-words (BoW)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_data]

    # Trains the LDA model
    lda_model = gensim.models.LdaMulticore(
        bow_corpus,
        num_topics=10,
        id2word=dictionary,
        passes=2,
        workers=2,
    )

    # Applies the LDA model to all of the data
    topic_distributions = []
    for bow in bow_corpus:
        topic_distribution = lda_model[bow]
        topic_distributions.append(topic_distribution)

    # Visualises the topics using Word Clouds
    for topic_id in range(lda_model.num_topics):
        plot_wordcloud(lda_model, topic_id)

    # Visualises the topics using pyLDAvis
    visualize_lda(lda_model, bow_corpus, dictionary)

if __name__ == '__main__':
    main()
