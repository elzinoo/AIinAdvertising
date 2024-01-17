# Understanding how Artificial Intelligence techniques can support the understanding and development of effective and engaging advertising and promotional Twitter content, in a sporting context

This project applies sentiment analysis and topic modelling to data gathered from the English Premier League's Twitter account. The data involves tweets sent to, from and mentioning the Premier League account in 2022. The topic modelling results are saved in the 'TMdata' folder, and the results from the sentiment analysis are saved in the 'SAdataSVM' folder. The sentiment analysis was also used to create a prototype of an analytical dashboard. The 'sentimentAnalysisNB' file, does the same thing as the sentimentAnalysis file, but it uses a Naive Bayes classifier, instead of a SVM classifier.

## Installation

To install this project, you will need to have Python 3.9 installed on your machine. A virtual environment was used for this project, so you can activate it, and it will have all the necessary packages installed.

To activate the virtual environment on a macOS or Linux, navigate to the directory of the virtual environment and run the following command in your terminal: 
source virtualenv3.9/bin/activate

If you are using a Windows machine, you will need to recreate the virtual environment and install the following packages:

- dash                      2.9.3
- dash-bootstrap-components 1.4.1
- plotly                    5.14.1
- pandas                    1.2.5
- pyLDAvis                  3.3.1
- numpy                     1.24.2
- openpyxl                  3.1.2
- wordcloud                 1.9.1.1
- nltk                      3.8.1
- matplotlib                3.7.1
- scikit-learn              1.2.2
- seaborn                   0.12.2
- gensim                    4.3.1
- ipython                   8.12.0

## Running the Python files

The sentiment analysis files and the topic modelling file can be run normally. The folder already includes the sentiment analysis data; therefore, the dashboard file can also be run normally.

To then view the dashboard in your browser, visit: http://127.0.0.1:8046

The dashboard provides interactive visualisations such as sentiment distribution over time and sentiment chart, word clouds and averages from the data, such as average likes. 
