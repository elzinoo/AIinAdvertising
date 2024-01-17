import base64
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# Creates a new Dash application instance with Bootstrap theme for styling
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# The following parts add styling to the dashboard
box_style = {
    'border': '2px solid #1DA1F2',
    'border-radius': '5px',
    'background-color': '#E2F1FC',
    'padding': '10px',
    'width': '90%',
    'margin': 'auto'
}

dropdown_style = {
    'width': '450px', 
    'height': '60px', 
    'float': 'right', 
    'font-size': '22px', 
    'text-align':'left', 
    'font-family':'Arial', 
    'background-color':'#E2F1FC',
}

dashboard_border_style = {
    'border': '5px solid #FFFFFF',
    'border-radius': '5px',
    'padding': '5px',
    'background-color': '#FFFFFF'
}

# Defines the layout of the dashboard, including components and structure
app.layout = html.Div([
    html.Div([
        dbc.Row(
            [
                dbc.Col(html.Img(
                            src='data:image/png;base64,{}'.format(
                                base64.b64encode(open("images/twitterLogo.png", 'rb').read()).decode()
                            ),
                            style={'width': '50%'}
                        ),
                        width={"size": 1},
                ),
                dbc.Col(html.H1("Sentiment Analysis Dashboard"),
                        style={'color' : '#1DA1F2', 'fontsize' : '55', 'font-family':'Times New Roman'},
                        width={"size": 6},
                        ),
                dbc.Col(dcc.Dropdown(id='file-dropdown', placeholder='select file',
                            options=[
                                {'label': 'From Premier League 2022', 'value': 'from_premierleague2022'},
                                {'label': 'To Premier League 2022', 'value': 'to_premierleague2022'},
                                {'label': 'Mentions Premier League 2022', 'value': 'mentions_premierleague2022'}
                            ],
                            value='from_premierleague2022',
                            style=dropdown_style,
                            ),
                            width={"size": 5},
                        ),
            ],
        ),
        dbc.Row(
            [    
                dbc.Col(
                    html.Div(id='total-tweets', style=box_style),
                    width={"size": 3},
                    style={'text-align': 'center', 'margin-top': '10px'}
                ),
                dbc.Col(
                    html.Div(id='average-likes', style=box_style),
                    width={"size": 3},
                    style={'text-align': 'center', 'margin-top': '10px'}
                ),
                dbc.Col(
                    html.Div(id='average-replies', style=box_style),
                    width={"size": 3},
                    style={'text-align': 'center', 'margin-top': '10px'}
                ),
                dbc.Col(
                    html.Div(id='average-retweets', style=box_style),
                    width={"size": 3},
                    style={'text-align': 'center', 'margin-top': '10px'}
                )
            ], style={'margin-top': '20px'}
        ),
        dbc.Row(
            [
                dbc.Col(children=[
                                html.Div(id='sentiment-chart'),
                            ],
                            width={"size": 4},
                            style={'text-align': 'center'}
                        ),
                dbc.Col(children=[
                                html.Div(id='pie-chart')
                            ],
                            width={"size": 4},
                            style={'text-align': 'center'}
                        ),
                dbc.Col(children=[
                                html.Div(id='negative-word-cloud')
                            ],
                            width={"size": 4},
                            style={'width': ''}
                        ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(children=[
                                html.Div(id='time-series')
                            ],
                            width={"size": 8},
                            style={'text-align': 'center'}
                        ),
                dbc.Col(children=[
                                html.Div(id='positive-word-cloud'),
                            ],
                            width={"size": 4},
                            style={'width': ''}
                        ),
            ]
        ),
    ], style=dashboard_border_style)
])

# Defines a callback that updates the output based on the file selected
@app.callback(
    [Output('sentiment-chart', 'children'),
     Output('pie-chart', 'children'),
     Output('time-series', 'children'),
     Output('positive-word-cloud', 'children'),
     Output('negative-word-cloud', 'children'),
     Output('total-tweets', 'children'),
     Output('average-likes', 'children'),
     Output('average-replies', 'children'),
     Output('average-retweets', 'children')],
    [Input('file-dropdown', 'value')])

# The following function updates the visual data shown based on the selected file
def update_output(value):
    sentiment_data_path = f"sentimentAnalysis/{value}_sentiment.csv"
    df = pd.read_csv(sentiment_data_path)

    total_tweets = len(df)
    total_tweets_text = html.Span([
        html.Strong("Total Number of Tweets:"),html.Br(), f" {total_tweets}"
    ], style={'font-size': '22px', 'font-family': 'Times New Roman', 'color':'darkblue'})

    average_likes = df['metrics_like_count'].mean()
    average_likes_text = html.Span([
        html.Strong("Average Number of Likes:"), html.Br(), f" {average_likes:.0f}"
    ], style={'font-size': '22px', 'font-family': 'Times New Roman', 'color':'darkblue'})

    average_replies = df['metrics_reply_count'].mean()
    average_replies_text = html.Span([
        html.Strong("Average Number of Replies:"), html.Br(), f" {average_replies:.0f}"
    ], style={'font-size': '22px', 'font-family': 'Times New Roman', 'color':'darkblue'})

    average_retweets = df['metrics_retweet_count'].mean()
    average_retweets_text = html.Span([
        html.Strong("Average Number of Retweets:"), html.Br(), f" {average_retweets:.0f}"
    ], style={'font-size': '22px', 'font-family': 'Times New Roman', 'color':'darkblue'})
    sentiment_counts = df['sentiment'].value_counts()

    sentiment_chart = dcc.Graph(
        figure=px.bar(
            sentiment_counts, title='Sentiment Chart', 
            labels=['Negative', 'Positive', 'Neutral'],
            color_discrete_sequence=['#1DA1F2']
        ).update_layout(font={'family': 'Times New Roman'})
    )

    pie_chart = dcc.Graph(
        figure=px.pie(
            df, names='sentiment', title='Sentiment Distribution',
            color_discrete_sequence=['#1DA1F2', 'green', 'red']
        ).update_layout(font={'family': 'Times New Roman'})
    )

    df['date'] = pd.to_datetime(df['created_at']).dt.date
    sentiment_over_time = df.groupby(['date', 'sentiment']).size().reset_index(name='count')
    time_series = dcc.Graph(
        figure=px.line(
            sentiment_over_time, x='date', y='count', color='sentiment', 
            title='Sentiment Over Time',
            color_discrete_sequence=['red', 'green', '#1DA1F2']
        ).update_layout(font={'family': 'Times New Roman'})
    )

    def create_word_cloud_image_element(sentiment, value):
        file_ext = '.csv' if value.startswith('mentions') else '.xlsx'
        word_cloud_path = f"SAdataSVM/{value}{file_ext}_{sentiment}_word_cloud.png"
        encoded = base64.b64encode(open(word_cloud_path, 'rb').read()).decode()
        return html.Img(src=f"data:image/png;base64,{encoded}")

    positive_word_cloud = create_word_cloud_image_element("positive", value)
    negative_word_cloud = create_word_cloud_image_element("negative", value)

    return sentiment_chart, pie_chart, time_series, positive_word_cloud, negative_word_cloud, total_tweets_text, average_likes_text, average_replies_text, average_retweets_text

if __name__ == '__main__':
    app.run_server(debug=True, port=8046)
