# TITLE: Stocks News Sentiment Analysis with Portfolio Optimization and Future Returns Prediction
# Name: Subir Singh
# Year: Oct 2023
import streamlit as st
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime, timedelta

# Replace with your own News API key
NEWS_API_KEY = '5843e8b1715a4c1fb6628befb47ca1e8'


def get_stock_news(stock_symbol, num_articles=10):
    url = f'https://newsapi.org/v2/everything?q={stock_symbol}&apiKey={NEWS_API_KEY}&sortBy=publishedAt&pageSize={num_articles}'
    response = requests.get(url)
    news_data = response.json()
    articles = news_data['articles']
    return articles


def analyze_sentiment(article_text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(article_text)
    compound_score = sentiment_scores['compound']

    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def get_stock_data(stock_symbol, start_date, end_date):
    try:
        stock = yf.Ticker(stock_symbol)
        df = stock.history(start=start_date, end=end_date)
        if df.empty:
            st.warning(f"No data found for {stock_symbol} in the specified date range.")
            return None
        return df
    except Exception as e:
        st.error(f"Error fetching data for {stock_symbol}: {str(e)}")
        return None


def calculate_portfolio_performance(weights, returns, cov_matrix):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return portfolio_return, portfolio_stddev


def minimize_portfolio_volatility(weights, returns, cov_matrix):
    return -calculate_portfolio_performance(weights, returns, cov_matrix)[0]


def predict_future_returns(returns, portfolio_weights, days=30):
    # Calculate mean daily return
    mean_return = returns.mean()

    # Calculate projected future returns
    future_returns = mean_return * portfolio_weights * days
    return future_returns


def main():
    st.title("Stocks News Sentiment Analysis with Portfolio Optimization and Future Returns Prediction")
    st.subheader("By Subir Singh, Oct 2023")

    st.sidebar.header("User Input")
    stock_symbols_portfolio = st.sidebar.text_input("Enter stock symbols for portfolio optimization (separated by spaces)", "AAPL GOOGL MSFT").split()
    start_date = st.sidebar.text_input("Enter start date for historical data (YYYY-MM-DD)", "2022-01-01")
    end_date = st.sidebar.text_input("Enter end date for historical data (YYYY-MM-DD)", "2023-01-01")

    # ... (rest of the code remains unchanged)

    # Sentiment Analysis
    st.sidebar.header("Sentiment Analysis")
    stock_symbols_sentiment = st.sidebar.text_input("Enter stock symbols for sentiment analysis (separated by spaces)", "AAPL GOOGL MSFT").split()

    st.subheader("Performing Sentiment Analysis...")
    for stock_symbol in stock_symbols_sentiment:
        news_articles = get_stock_news(stock_symbol, num_articles=10)

        if not news_articles:
            st.warning(f"No news articles found for {stock_symbol}.")
            continue

        st.write(f"\nLatest 10 news articles related to {stock_symbol}:")

        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for i, article in enumerate(news_articles, 1):
            st.write(f"\nArticle {i}:")
            st.write(f"Title: {article['title']}")
            st.write(f"Source: {article['source']['name']}")
            st.write(f"Published At: {article['publishedAt']}")

            sentiment = analyze_sentiment(article['title'])
            st.write(f"Sentiment: {sentiment}")

            # Print the news link
            st.write(f"News Link: {article['url']}")

            # Update sentiment counters
            if sentiment == "Positive":
                positive_count += 1
            elif sentiment == "Negative":
                negative_count += 1
            else:
                neutral_count += 1

        # Print sentiment summary for the current stock symbol
        st.write("\nSentiment Summary:")
        st.write(f"Total Positive: {positive_count}")
        st.write(f"Total Negative: {negative_count}")
        st.write(f"Total Neutral: {neutral_count}")

        # ... (rest of the code remains unchanged)

if __name__ == "__main__":
    main()
