! pip install -q findspark pyspark pandas requests dash dash-bootstrap-components plotly pyngrok

import findspark
findspark.init()

import os
import re
import requests
import pandas as pd
import time
import threading
import shutil
from datetime import datetime

# Spark Imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, split, udf, current_timestamp
from pyspark.sql.types import StringType
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression

# Dash and Plotly Imports for Dashboard
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from pyngrok import ngrok

print("STEP 1: All libraries imported successfully.")

# Part 2: Configuration - IMPORTANT: ADD YOUR KEYS HERE
NEWS_API_KEY = "Your_News_API"
if NEWS_API_KEY == "YOUR_API_KEY_HERE":
    print("WARNING: Please replace 'YOUR_API_KEY_HERE' with your actual NewsAPI key.")

NGROK_AUTH_TOKEN = "YOUR_NGROK_AUTH_TOKEN_HERE" # <-- PASTE YOUR NGROK AUTH TOKEN HERE
if NGROK_AUTH_TOKEN == "YOUR_NGROK_AUTH_TOKEN_HERE":
    print("WARNING: Please replace 'YOUR_NGROK_AUTH_TOKEN_HERE' to expose the dashboard publicly.")


spark = SparkSession.builder \
    .appName("NewsSentimentAnalysis") \
    .config("spark.sql.shuffle.partitions", "2") \
    .getOrCreate()

def train_sentiment_model():
    """Loads training data, trains, and returns a sentiment analysis pipeline model."""
    file_path = 'all-data.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: Training data '{file_path}' not found. Please ensure it's in the same directory.")

    # Read data robustly
    df_raw = spark.read.text(file_path)
    split_col = split(df_raw['value'], ',', 2)
    df = df_raw.withColumn('sentiment', split_col.getItem(0)) \
               .withColumn('text', split_col.getItem(1))
    df = df.filter(col("sentiment").isNotNull() & col("text").isNotNull() & (col("sentiment") != "") & (col("sentiment") != "neutral"))

    # Balance the dataset
    df_positive = df.filter(col("sentiment") == "positive")
    df_negative = df.filter(col("sentiment") == "negative")
    min_count = min(df_positive.count(), df_negative.count())

    if min_count == 0:
        raise ValueError("Training data does not contain both 'positive' and 'negative' samples.")

    df_balanced = df_positive.limit(min_count).union(df_negative.limit(min_count))

    # Define and Train the PySpark ML Pipeline
    df_balanced = df_balanced.withColumn("cleaned_text", lower(regexp_replace('text', r'[^\w\s]', '')))
    label_indexer = StringIndexer(inputCol="sentiment", outputCol="label", handleInvalid="skip")
    tokenizer = Tokenizer(inputCol="cleaned_text", outputCol="words")
    hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="rawFeatures", numFeatures=2000)
    idf = IDF(inputCol=hashingTF.getOutputCol(), outputCol="features")
    lr = LogisticRegression(featuresCol="features", labelCol="label")
    pipeline = Pipeline(stages=[label_indexer, tokenizer, hashingTF, idf, lr])

    print("Training the sentiment analysis model...")
    sentiment_model = pipeline.fit(df_balanced)
    print(" Model trained successfully.")
    return sentiment_model

# Train the model
sentiment_model = train_sentiment_model()

# Part 4: Real-time News Producer (Runs in Background)
STREAM_INPUT_DIR = "stream_input"
# Clean up previous stream data before starting
if os.path.exists(STREAM_INPUT_DIR):
    shutil.rmtree(STREAM_INPUT_DIR)
os.makedirs(STREAM_INPUT_DIR, exist_ok=True)


def fetch_and_write_news():
    """Fetches news and writes each headline to a new file in the stream directory."""
    while True:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetching live news...")
        url = f"https://newsapi.org/v2/top-headlines?language=en&category=business&apiKey={NEWS_API_KEY}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            articles = response.json().get("articles", [])
            for i, article in enumerate(articles):
                headline = article.get("title")
                if headline:
                    file_path = os.path.join(STREAM_INPUT_DIR, f"news_{int(time.time())}_{i}.txt")
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(headline)
            print(f"    -> Wrote {len(articles)} new headlines.")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching news: {e}")
        time.sleep(60) # Fetch news every 60 seconds

# Start the producer thread as a daemon
producer_thread = threading.Thread(target=fetch_and_write_news, daemon=True)
producer_thread.start()
print(" News producer started in the background.")

# Part 5: Spark Structured Streaming Pipeline
STREAM_CHECKPOINT_DIR = "stream_checkpoint"
# Clean up previous checkpoint data before starting
if os.path.exists(STREAM_CHECKPOINT_DIR):
    shutil.rmtree(STREAM_CHECKPOINT_DIR)
os.makedirs(STREAM_CHECKPOINT_DIR, exist_ok=True)

# Create the input stream
inputStream = spark.readStream \
    .format("text") \
    .option("path", STREAM_INPUT_DIR) \
    .load()

# Preprocess the incoming text data
processedStream = inputStream.withColumn("cleaned_text", lower(regexp_replace('value', r'[^\w\s]', ''))) \
                             .withColumn("timestamp", current_timestamp())


# Create a prediction pipeline from the trained model's stages
prediction_pipeline = PipelineModel(stages=sentiment_model.stages[1:]) # Exclude label_indexer
predictionsStream = prediction_pipeline.transform(processedStream)

# Convert prediction labels (0.0, 1.0) back to strings ("positive", "negative")
label_converter = {float(i): label for i, label in enumerate(sentiment_model.stages[0].labels)}
converter_udf = udf(lambda p: label_converter.get(p, "unknown"), StringType())
finalStream = predictionsStream.withColumn("sentiment", converter_udf(col("prediction")))

# Write the results to an in-memory table
query = finalStream \
    .select("timestamp", "value", "sentiment") \
    .writeStream \
    .format("memory") \
    .queryName("sentiment_results") \
    .outputMode("append") \
    .option("checkpointLocation", STREAM_CHECKPOINT_DIR) \
    .start()

print("Spark Structured Streaming pipeline is running and writing to 'sentiment_results' table.")

!pip install wordcloud

# Part 6: Real-time Dashboard with Dash (Enhanced Version)
from dash import dash_table
from wordcloud import WordCloud
import io
import base64

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

# --- Helper function for Word Cloud ---
def plot_wordcloud(data):
    d = {a: x for a, x in data.values}
    wc = WordCloud(background_color='black', width=480, height=360)
    wc.fit_words(d)
    return wc.to_image()

# --- App Layout ---
app.layout = dbc.Container([
    # Title
    dbc.Row(dbc.Col(html.H1("Real-Time News Sentiment Dashboard", className="text-center text-primary mb-4"), width=12)),

    # KPI Cards
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([html.H4("Total Headlines", className="card-title"), html.P(id="total-headlines-card", className="card-text fs-3")]), color="primary", inverse=True), width=4),
        dbc.Col(dbc.Card(dbc.CardBody([html.H4("Positive Sentiment %", className="card-title"), html.P(id="positive-pct-card", className="card-text fs-3")]), color="success", inverse=True), width=4),
        dbc.Col(dbc.Card(dbc.CardBody([html.H4("Negative Sentiment %", className="card-title"), html.P(id="negative-pct-card", className="card-text fs-3")]), color="danger", inverse=True), width=4),
    ], className="mb-4"),

    # Main Charts
    dbc.Row([
        dbc.Col(dcc.Graph(id='sentiment-pie-chart'), width=12, lg=6),
        dbc.Col(dcc.Graph(id='sentiment-time-series'), width=12, lg=6),
    ], className="mb-4"),

    # Word Clouds
    dbc.Row([
        dbc.Col(html.Img(id='positive-wordcloud'), width=12, lg=6),
        dbc.Col(html.Img(id='negative-wordcloud'), width=12, lg=6),
    ], className="mb-4"),

    # Interactive Data Table
    dbc.Row(dbc.Col(html.H3("Latest Headlines", className="text-center text-light mt-4 mb-2"), width=12)),
    dbc.Row(dbc.Col(
        dash_table.DataTable(
            id='latest-headlines-table',
            style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
            style_cell={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white', 'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto'},
            style_table={'overflowX': 'auto'}
        ), width=12
    )),

    dcc.Interval(id='interval-component', interval=5*1000, n_intervals=0)
], fluid=True)


@app.callback(
    [Output('sentiment-pie-chart', 'figure'),
     Output('sentiment-time-series', 'figure'),
     Output('latest-headlines-table', 'data'),
     Output('latest-headlines-table', 'columns'),
     Output('total-headlines-card', 'children'),
     Output('positive-pct-card', 'children'),
     Output('negative-pct-card', 'children'),
     Output('positive-wordcloud', 'src'),
     Output('negative-wordcloud', 'src')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    try:
        df_results = spark.sql("SELECT * FROM sentiment_results").toPandas()
    except Exception as e:
        df_results = pd.DataFrame(columns=['timestamp', 'value', 'sentiment'])

    if df_results.empty:
        empty_pie = go.Figure(go.Pie(labels=['Positive', 'Negative'], values=[0, 0], hole=.3)).update_layout(title_text='Sentiment Distribution', template='plotly_dark')
        empty_ts = go.Figure().update_layout(title_text='Sentiment Over Time (Waiting for data...)', template='plotly_dark')
        return empty_pie, empty_ts, [], [], "0", "0.0%", "0.0%", None, None

    # --- KPI Card Calculations ---
    total_headlines = len(df_results)
    positive_count = len(df_results[df_results['sentiment'] == 'positive'])
    negative_count = len(df_results[df_results['sentiment'] == 'negative'])
    positive_pct = f"{(positive_count / total_headlines) * 100:.1f}%" if total_headlines > 0 else "0.0%"
    negative_pct = f"{(negative_count / total_headlines) * 100:.1f}%" if total_headlines > 0 else "0.0%"

    # --- Sentiment Pie Chart ---
    sentiment_counts = df_results['sentiment'].value_counts()
    pie_fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title='Sentiment Distribution', hole=.3, color=sentiment_counts.index, color_discrete_map={'positive':'green', 'negative':'red'})
    pie_fig.update_layout(template='plotly_dark', legend_title_text='Sentiment')

    # --- Time Series Bar Chart with Trend Line ---
    df_results['timestamp'] = pd.to_datetime(df_results['timestamp'])
    time_series_data = df_results.set_index('timestamp').resample('1min').apply({'sentiment': lambda x: (x == 'positive').sum() - (x == 'negative').sum()}).rename(columns={'sentiment': 'net_sentiment'}).reset_index()
    time_series_data['trend'] = time_series_data['net_sentiment'].rolling(3, min_periods=1).mean() # 3-minute moving average

    ts_fig = px.bar(time_series_data, x='timestamp', y='net_sentiment', title='Net Sentiment Score Over Time (Positive - Negative)', color='net_sentiment', color_continuous_scale=px.colors.diverging.RdYlGn)
    ts_fig.add_trace(go.Scatter(x=time_series_data['timestamp'], y=time_series_data['trend'], mode='lines', name='Trend', line=dict(color='yellow', width=2)))
    ts_fig.update_layout(template='plotly_dark', xaxis_title='Time', yaxis_title='Net Sentiment Score')

    # --- Word Clouds ---
    pos_text = ' '.join(df_results[df_results['sentiment'] == 'positive']['value'].dropna())
    neg_text = ' '.join(df_results[df_results['sentiment'] == 'negative']['value'].dropna())

    pos_wordcloud_img = WordCloud(background_color='#111111', colormap='Greens', width=480, height=360).generate(pos_text if pos_text else 'No positive words').to_image()
    neg_wordcloud_img = WordCloud(background_color='#111111', colormap='Reds', width=480, height=360).generate(neg_text if neg_text else 'No negative words').to_image()

    img_stream_pos = io.BytesIO()
    img_stream_neg = io.BytesIO()
    pos_wordcloud_img.save(img_stream_pos, format='PNG')
    neg_wordcloud_img.save(img_stream_neg, format='PNG')

    encoded_pos_img = base64.b64encode(img_stream_pos.getvalue()).decode()
    encoded_neg_img = base64.b64encode(img_stream_neg.getvalue()).decode()

    pos_wc_src = f"data:image/png;base64,{encoded_pos_img}"
    neg_wc_src = f"data:image/png;base64,{encoded_neg_img}"

    # --- Interactive Data Table ---
    latest_headlines = df_results.sort_values(by='timestamp', ascending=False).head(10)
    table_data = latest_headlines[['value', 'sentiment']].to_dict('records')
    table_columns = [{"name": "Headline", "id": "value"}, {"name": "Sentiment", "id": "sentiment"}]

    return pie_fig, ts_fig, table_data, table_columns, f"{total_headlines}", positive_pct, negative_pct, pos_wc_src, neg_wc_src


# Main execution block
if __name__ == '__main__':
    public_url_obj = None
    if NGROK_AUTH_TOKEN != "YOUR_NGROK_AUTH_TOKEN_HERE":
        print("--- Attempting robust ngrok cleanup ---")
        ngrok.kill()
        time.sleep(2)
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)

        print("Starting new ngrok tunnel...")
        public_url_obj = ngrok.connect(8050)
        print(f"Dashboard is publicly accessible at: {public_url_obj.public_url}")
    else:
        print("Dashboard is ready for local access.")
        print("Navigate to http://1227.0.0.1:8050/ in your web browser.")

    try:
        app.run(debug=False)
    finally:
        if public_url_obj:
            print(f"Disconnecting ngrok tunnel: {public_url_obj.public_url}")
            ngrok.disconnect(public_url_obj.public_url)
        ngrok.kill()
        print("Ngrok processes terminated.")

