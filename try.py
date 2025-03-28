import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from geopy.geocoders import Nominatim
from sklearn.linear_model import LinearRegression
import numpy as np
from config import SERPAPI_KEY

# Function to scrape reviews from the web
def scrape_reviews(keyword, location):
    import os
    api_key = os.getenv("SERPAPI_KEY") or SERPAPI_KEY
    if not api_key:
        st.error("API key is missing. Please configure your SerpAPI key.")
        return []

    query = f"{keyword} {location} reviews"
    url = f"https://serpapi.com/search.json?q={query}&hl=en&gl=ke&api_key={api_key}"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            reviews = [
                item.get("snippet", "No snippet available")
                for item in data.get("organic_results", [])
                if "snippet" in item
            ]
            if not reviews:
                st.warning("No reviews were found for the given query.")
            return reviews
        elif response.status_code == 429:
            st.error("API quota exceeded. Please check your SerpAPI account.")
        else:
            st.error(f"Failed to fetch reviews: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {e}")
    return []

# Analyze reviews to predict business viability
def analyze_reviews(reviews):
    # Using pre-trained sentiment analysis model (VADER)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    nltk.download('vader_lexicon')

    sid = SentimentIntensityAnalyzer()

    sentiment_scores = [sid.polarity_scores(review)['compound'] for review in reviews]
    average_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0

    return average_sentiment

# Helper function to suggest improvement steps
def suggest_improvements(score):
    if score > 0.8:
        return "The business has a high viability. Focus on maintaining quality and engaging with customers."
    elif 0.5 <= score <= 0.8:
        return "The business shows potential but consider improving marketing strategies or addressing customer pain points."
    else:
        return "The business has low viability. Conduct more research or identify underserved niches in the market."

# Function to display a word cloud
def display_wordcloud(reviews):
    text = " ".join(reviews)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Function to visualize business location on a map
def display_location_on_map(location):
    geolocator = Nominatim(user_agent="business-viability-app")
    location_data = geolocator.geocode(location)

    if location_data:
        st.map(pd.DataFrame({
            'lat': [location_data.latitude],
            'lon': [location_data.longitude]
        }))
    else:
        st.warning("Location not found. Please try another location.")

# UI Layout
st.title("Business Service Viability Predictor")
st.sidebar.title("Input Parameters")

# Sidebar inputs
keyword = st.sidebar.text_input("Keyword for Business (e.g., Restaurant, IT Services)", "Restaurant")
location = st.sidebar.text_input("Location (e.g., Nairobi, Mombasa)", "Nairobi")

# Analyze button
if st.sidebar.button("Analyze Potential"):
    with st.spinner("Fetching reviews and analyzing..."):
        reviews = scrape_reviews(keyword, location)
        if reviews:
            score = analyze_reviews(reviews)

            # Display results
            st.subheader(f"Predicted Viability for {keyword} in {location}")
            st.write(f"Sentiment Score: {score:.2f} (1: Highly Viable, 0: Not Viable)")

            # Provide suggestions
            suggestions = suggest_improvements(score)
            st.subheader("Suggestions")
            st.write(suggestions)

            # Display sample reviews
            st.subheader("Sample Reviews")
            for review in reviews[:5]:
                st.write(f"- {review}")

            # Display word cloud
            st.subheader("Word Cloud of Reviews")
            display_wordcloud(reviews)

            # Display location on map
            st.subheader("Business Location on Map")
            display_location_on_map(location)
        else:
            st.write("No reviews found. Consider expanding the search criteria.")
