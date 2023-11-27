# Import necessary libraries
import streamlit as st
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime, timedelta

# Title and introduction
st.title("Stocks News Sentiment Analysis with Portfolio Optimization and Future Returns Prediction")
st.write("This app performs sentiment analysis on stock news, optimizes a portfolio, and predicts future returns.")

# Input for News API key
NEWS_API_KEY = st.text_input("Enter your News API key:")

def get_stock_news(stock_symbol, num_articles=10):
    url = f'https://newsapi.org/v2/everything?q={stock_symbol}&apiKey={NEWS_API_KEY}&sortBy=publishedAt&pageSize={num_articles}'
    response = requests.get(url)
    news_data = response.json()
    articles = news_data['articles']
    return articles

# Input for stock symbols
stock_symbols_portfolio = st.text_input("Enter a list of stock symbols for portfolio optimization separated by spaces (e.g., INFY.NS ITC.NS WIPRO.NS):").split()

# Input for historical data range
start_date = st.text_input("Enter start date for historical data (YYYY-MM-DD):")
end_date = st.text_input("Enter end date for historical data (YYYY-MM-DD):")

# Initialize counters for positive, negative, and neutral sentiments
total_positive_count = 0
total_negative_count = 0
total_neutral_count = 0

stock_data = {}

# Portfolio Optimization
st.header("Portfolio Optimization")
st.write("Performing Portfolio Optimization...")

for stock_symbol in stock_symbols_portfolio:
    stock_df = get_stock_data(stock_symbol, start_date, end_date)
    if stock_df is not None:
        stock_data[stock_symbol] = stock_df

if not stock_data:
    st.warning("No stock data available for portfolio optimization.")
else:
    # Calculate returns for selected stocks
    stock_returns = pd.concat([df['Close'].pct_change().dropna() for df in stock_data.values()], axis=1)
    stock_returns.columns = stock_data.keys()

    # Calculate covariance matrix
    cov_matrix = stock_returns.cov()

    # Calculate expected returns for selected stocks
    expected_returns = stock_returns.mean()

    # Define initial weights for the portfolio optimization
    num_assets = len(stock_symbols_portfolio)
    initial_weights = [1 / num_assets] * num_assets

    # Perform portfolio optimization to find optimal weights
    bounds = [(0, 1) for _ in range(num_assets)]
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    optimized_weights = minimize(minimize_portfolio_volatility, initial_weights, args=(expected_returns, cov_matrix),
                                 method='SLSQP', bounds=bounds, constraints=constraints)

    # Extract optimal weights
    optimal_weights = optimized_weights.x

    # Calculate portfolio performance with optimal weights
    portfolio_return, portfolio_stddev = calculate_portfolio_performance(optimal_weights, expected_returns, cov_matrix)

    # Print portfolio optimization results
    st.subheader("Portfolio Optimization Results:")
    st.write(f"Optimal Weights: {optimal_weights}")
    st.write(f"Expected Portfolio Return: {portfolio_return:.4f}")
    st.write(f"Portfolio Standard Deviation: {portfolio_stddev:.4f}")

    # Compare with Nifty Benchmark
    nifty_data = get_stock_data(nifty_symbol, start_date, end_date)
    if nifty_data is not None:
        nifty_returns = nifty_data['Close'].pct_change().dropna()
        nifty_return = np.mean(nifty_returns) * 252
        nifty_stddev = np.std(nifty_returns) * np.sqrt(252)

        st.subheader("Nifty Benchmark Results:")
        st.write(f"Expected Nifty Return: {nifty_return:.4f}")
        st.write(f"Nifty Standard Deviation: {nifty_stddev:.4f}")

# Sentiment Analysis
st.header("Sentiment Analysis")
stock_symbols_sentiment = st.text_input("Enter a list of stock symbols for sentiment analysis separated by spaces (e.g., INFY ITC WIPRO):").split()

sentiment_data = {}

st.write("Performing Sentiment Analysis...")

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
    st.subheader("Sentiment Summary:")
    st.write(f"Total Positive: {positive_count}")
    st.write(f"Total Negative: {negative_count}")
    st.write(f"Total Neutral: {neutral_count}")

    sentiment_data[stock_symbol] = {
        "Positive": positive_count,
        "Negative": negative_count,
        "Neutral": neutral_count
    }

    # Update the total sentiment counters
    total_positive_count += positive_count
    total_negative_count += negative_count
    total_neutral_count += neutral_count

# Combined Results
st.header("Combined Portfolio Optimization and Sentiment Analysis Results:")
st.write(f"Total Positive Sentiment: {total_positive_count}")
st.write(f"Total Negative Sentiment: {total_negative_count}")
st.write(f"Total Neutral Sentiment: {total_neutral_count}")

# Future Returns Prediction
days_to_predict = st.number_input("Enter the number of days for future returns prediction:", min_value=1, step=1, value=30)
portfolio_returns_prediction = predict_future_returns(stock_returns, optimal_weights, days_to_predict)

st.subheader("Projected Future Returns for Portfolio:")
st.write(f"{days_to_predict} days: {portfolio_returns_prediction.values[0]:.4f}")
