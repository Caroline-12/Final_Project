import streamlit as st
import requests
from msal import ConfidentialClientApplication
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from geopy.geocoders import Nominatim
from config import SERPAPI_KEY

nltk.download("vader_lexicon")

# Power BI Configuration
CLIENT_ID = "368a3554-e68e-4e0c-ad1a-0bbd6f33b715"
CLIENT_SECRET = "bd9ed252-ab11-4894-8fb0-61a37fb1dfc6"
TENANT_ID = "06151e3a-25a4-411b-a0e0-ccc6e1fad02a"
WORKSPACE_ID = "fad97a95-c663-4cd1-83b3-3b712c3f407e"
REPORT_ID = "YOUR_REPORT_ID"

AUTHORITY_URL = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPE = ["https://analysis.windows.net/powerbi/api/.default"]
POWER_BI_API_URL = (
    f"https://api.powerbi.com/v1.0/myorg/groups/{WORKSPACE_ID}/reports/{REPORT_ID}"
)

# Authenticate with Power BI
def get_access_token():
    app = ConfidentialClientApplication(
        CLIENT_ID, authority=AUTHORITY_URL, client_credential=CLIENT_SECRET
    )
    result = app.acquire_token_for_client(scopes=SCOPE)
    return result.get("access_token")


def get_powerbi_report_embed_url(token):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(POWER_BI_API_URL, headers=headers)
    if response.status_code == 200:
        return response.json().get("embedUrl")
    else:
        st.error(f"Failed to fetch Power BI report: {response.text}")
        return None


# Scrape reviews using SerpAPI
def scrape_reviews(keyword, location, api_key):
    url = f"https://serpapi.com/search.json?q={keyword} {location} reviews&hl=en&gl=ke&api_key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return [item.get("snippet", "No snippet") for item in response.json().get("organic_results", [])]
    return []


# Sentiment Analysis
def analyze_reviews(reviews):
    sid = SentimentIntensityAnalyzer()
    scores = [sid.polarity_scores(r)["compound"] for r in reviews]
    return np.mean(scores) if scores else 0


# Word Cloud Display
def display_wordcloud(reviews):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(reviews))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)


# Location on Map
def display_location_on_map(location):
    geolocator = Nominatim(user_agent="business-viability-app")
    loc_data = geolocator.geocode(location)
    if loc_data:
        st.map(pd.DataFrame({"lat": [loc_data.latitude], "lon": [loc_data.longitude]}))


# Suggestions based on sentiment
def suggest_improvements(score):
    if score > 0.8:
        return "High viability. Focus on quality and customer engagement."
    elif 0.5 <= score <= 0.8:
        return "Potential viability. Improve marketing and address customer concerns."
    return "Low viability. Research underserved niches."


# Streamlit UI
st.title("Business Viability Analyzer with Power BI Integration")
st.sidebar.title("Business Parameters")

business_type = st.sidebar.text_input("Nature of Business", "Restaurant")
location = st.sidebar.text_input("Preferred Location", "Nairobi")
SERPAPI_KEY = st.sidebar.text_input("SerpAPI Key", "")

if st.sidebar.button("Analyze Business Potential"):
    st.info("Fetching data and generating insights...")
    reviews = scrape_reviews(business_type, location, SERPAPI_KEY)
    if reviews:
        score = analyze_reviews(reviews)
        st.subheader(f"Viability Analysis for {business_type} in {location}")
        st.write(f"Sentiment Score: {score:.2f} (1: Highly Viable, 0: Not Viable)")
        st.write(suggest_improvements(score))

        st.subheader("Sample Reviews")
        for r in reviews[:5]:
            st.write(f"- {r}")

        st.subheader("Word Cloud Analysis")
        display_wordcloud(reviews)

        st.subheader("Location Visualization")
        display_location_on_map(location)

        # Power BI Report Embedding
        st.subheader("Detailed Power BI Report")
        token = get_access_token()
        if token:
            embed_url = get_powerbi_report_embed_url(token)
            if embed_url:
                st.markdown(
                    f"""
                    <iframe 
                        width="1000" 
                        height="600" 
                        src="{embed_url}" 
                        frameborder="0" 
                        allowFullScreen="true">
                    </iframe>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.warning("No reviews found. Try adjusting the search criteria.")
