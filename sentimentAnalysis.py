import os
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC 
from wordcloud import WordCloud
import seaborn as sns

# The following functions cleans the data by removing emojis, urls, mentions (@) and hashtags (#)
def remove_emoji(string):
    if not isinstance(string, str):
        return ''

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
    for token in nltk.word_tokenize(text):
        if token.isalpha() and len(token) > 1:
            result.append(lemmatize(token.lower()))
    return ' '.join(result)

# The following functions create and save the sentiment analysis visualisations applied to the data
def plot_sentiment_chart(sentiment_counts, file_name):
    sentiment_labels = ['Negative', 'Positive', 'Neutral']
    rc('font', family='Times New Roman', weight='bold')
    plt.bar(sentiment_labels, sentiment_counts, color='lightblue')
    plt.xlabel('Sentiment', fontweight='bold')
    plt.ylabel('Number of Tweets', fontweight='bold')
    plt.title(f'Sentiment Chart for {file_name}', fontweight='bold')
    plt.savefig(f'SAdataSVM/{file_name}_sentiment_chart.png')
    plt.clf()

def plot_pie_chart(sentiment_counts, file_name):
    sentiment_labels = ['Negative', 'Positive', 'Neutral']
    colors = ['red', 'green', 'lightblue']
    rc('font', family='Times New Roman', weight='bold')
    plt.pie(sentiment_counts, labels=sentiment_labels, colors=colors, autopct='%1.1f%%')
    plt.title(f'Sentiment Distribution for {file_name}', fontweight='bold')
    plt.savefig(f'SAdataSVM/{file_name}_pie_chart.png')
    plt.clf()

def plot_time_series(data, file_name):
    data['date'] = pd.to_datetime(data['created_at']).dt.date
    sentiment_over_time = data.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
    sentiment_over_time.plot(kind='line')
    rc('font', family='Times New Roman', weight='bold')
    plt.xlabel('Date', fontweight='bold')
    plt.ylabel('Number of Tweets', fontweight='bold')
    plt.title(f'Sentiment Over Time for {file_name}', fontweight='bold')
    plt.legend(title='Sentiment', labels=['Negative', 'Neutral', 'Positive'])
    plt.savefig(f'SAdataSVM/{file_name}_time_series.png')
    plt.clf()

def plot_word_cloud(data, sentiment, file_name):
    sentiment_data = data[data['sentiment'] == sentiment]
    all_text = ' '.join([text for text in sentiment_data['processed_tweet']])
    rc('font', family='Times New Roman', weight='bold')
    if len(all_text) == 0:
        print(f"Skipping the word cloud as there were no words found for the {sentiment} sentiment in {file_name}.")
        return
    wordcloud = WordCloud(width=700, height=500, random_state=42, max_font_size=110).generate(all_text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'{sentiment} Word Cloud for {file_name}', fontweight='bold')
    plt.savefig(f'SAdataSVM/{file_name}_{sentiment}_word_cloud.png')
    plt.clf()


def plot_confusion_matrix(y_true, y_pred, file_name):
    matrix = confusion_matrix(y_true, y_pred)
    labels = ['Negative', 'Neutral', 'Positive']

    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted', fontweight='bold')
    plt.ylabel('True', fontweight='bold')
    plt.title(f'Confusion Matrix for {file_name}', fontweight='bold')
    plt.savefig(f'SAdataSVM/{file_name}_confusion_matrix.png')
    plt.clf()

# This function reads the files by using read_csv or read_excel depending on the file type
def read_file(file_path, file_type="csv"):
    if file_type == "csv":
        data = pd.read_csv(file_path, encoding='latin1')
    elif file_type == "xlsx":
        data = pd.read_excel(file_path)
    else:
        raise ValueError("The file type entered is invalid. Use 'csv' or 'xlsx'.")
    return data

def main():
    # Reads the training data and applies all the text cleaning and pre-processing functions
    training_data = read_file('trainingData/worldCupTweets.csv')
    training_data.columns = training_data.columns.str.strip()
    training_data['Tweet'] = training_data['Tweet'].apply(remove_emoji).apply(remove_url).apply(remove_mentions).apply(remove_hashtags)
    training_data['cleaned_text'] = training_data['Tweet'].apply(preprocess)

    # Splits the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(training_data['cleaned_text'], training_data['Sentiment'], test_size=0.2, random_state=42)

    # Vectorizes the text
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Trains the Support Vector Machine classifier
    svm_classifier = SVC()
    svm_classifier.fit(X_train_vectorized, y_train)

    # Tests the classifier
    y_pred = svm_classifier.predict(X_test_vectorized)

    # Calculates the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Generates a classification report for the classifier 
    report = classification_report(y_test, y_pred)
    print('Classification report:')
    print(report)

    # Generates a confusion matrix
    plot_confusion_matrix(y_test, y_pred, 'WorldCupTweets')

    # File paths for the twitter data
    file_paths = [    
        ('TwitterData/from_premierleague2022.xlsx', "xlsx"),    
        ('TwitterData/to_premierleague2022.xlsx', "xlsx"),    
        ('TwitterData/mentions_premierleague2022.csv', "csv"),
    ]

    # Creates the directories if they do not exist
    os.makedirs("sentimentAnalysis", exist_ok=True)
    os.makedirs("SAdataSVM", exist_ok=True)

    for file_path, file_type in file_paths:
        # Reads the Twitter data and applies all the text cleaning and pre-processing functions
        data = read_file(file_path, file_type)
        data['tweet_text'] = data['tweet_text'].apply(remove_emoji).apply(remove_url).apply(remove_mentions).apply(remove_hashtags)
        data['processed_tweet'] = data['tweet_text'].apply(preprocess)

        # Analyses the sentiment using the trained SVM classifier
        tweet_vectorized = vectorizer.transform(data['processed_tweet'])
        data['sentiment'] = svm_classifier.predict(tweet_vectorized)

        # Saves the sentiment analysis results to a new CSV file in the sentimentAnalysis folder
        output_file = "sentimentAnalysis/" + file_path.split(".")[0].split("/")[-1] + "_sentiment.csv"
        data.to_csv(output_file, index=False)
        print(f"Sentiment analysis results saved to {output_file}")

        # Plots the sentiment chart
        sentiment_counts = data['sentiment'].value_counts().sort_index().tolist()
        file_name = file_path.split("/")[-1]
        plot_sentiment_chart(sentiment_counts, file_name)

        # Prints sentiment counts
        print(f"Sentiment counts for {file_name}:")
        print(data['sentiment'].value_counts())

        # Calls all other visualisation functions
        plot_pie_chart(sentiment_counts, file_name)
        plot_time_series(data, file_name)
        plot_word_cloud(data, 'positive', file_name)
        plot_word_cloud(data, 'negative', file_name)

if __name__ == '__main__':
    main()
