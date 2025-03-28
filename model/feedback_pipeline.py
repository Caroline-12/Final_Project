import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from geotext import GeoText

nltk.download('punkt')
nltk.download('stopwords')

# Step 1: Search the web
def search_web(keyword, num_results=10):
    API_KEY = "AIzaSyC5vlurQ8RnW-qGJczA6LOjGOCxvYpl9JA"
    SEARCH_ENGINE_ID = "823ec56f4e9c64598"
    search_url = f"https://www.googleapis.com/customsearch/v1?q={keyword}&key={API_KEY}&cx={SEARCH_ENGINE_ID}&num={num_results}"
    response = requests.get(search_url)
    if response.status_code != 200:
        return []
    results = response.json().get("items", [])
    return [result.get("link") for result in results]

# Step 2: Scrape feedback
def scrape_feedback(urls):
    feedback_list = []
    for url in urls:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")
            paragraphs = soup.find_all("p")
            feedback_list.extend([p.text.strip() for p in paragraphs if len(p.text.strip()) > 50])
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            continue
    return feedback_list

# Step 3: Preprocess feedback
def preprocess_feedback(feedback_list):
    stop_words = set(stopwords.words("english"))
    cleaned_feedback = []
    for feedback in feedback_list:
        tokens = word_tokenize(feedback.lower())
        tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
        cleaned_feedback.append(" ".join(tokens))
    return cleaned_feedback

# Step 4: Sentiment analysis
def analyze_sentiment(feedback_list):
    sentiments = []
    for feedback in feedback_list:
        blob = TextBlob(feedback)
        sentiments.append(blob.sentiment.polarity)
    return sentiments

# Step 5: Aspect-based sentiment analysis
def aspect_based_sentiment(feedback_list, aspects):
    aspect_sentiments = {aspect: [] for aspect in aspects}

    for feedback in feedback_list:
        for aspect in aspects:
            if aspect in feedback:
                blob = TextBlob(feedback)
                aspect_sentiments[aspect].append(blob.sentiment.polarity)

    # Calculate the average sentiment for each aspect
    avg_aspect_sentiments = {
        aspect: sum(sentiments) / len(sentiments) if sentiments else 0
        for aspect, sentiments in aspect_sentiments.items()
    }
    return avg_aspect_sentiments

# Step 6: Analyze locations
def analyze_locations(feedback_list, sentiments):
    location_data = {}
    for feedback, sentiment in zip(feedback_list, sentiments):
        places = GeoText(feedback)
        for location in places.cities:
            if location not in location_data:
                location_data[location] = {"positive": 0, "neutral": 0, "negative": 0}
            
            if sentiment > 0:
                location_data[location]["positive"] += 1
            elif sentiment == 0:
                location_data[location]["neutral"] += 1
            else:
                location_data[location]["negative"] += 1
    return location_data

# Step 7: Extract topics
def extract_topics(feedback_list, num_topics=5):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(feedback_list)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
    lda.fit(dtm)
    topics = []
    for index, topic in enumerate(lda.components_):
        topic_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
        topics.append(f"Topic {index + 1}: " + ", ".join(topic_words))
    return topics


# Full analysis pipeline
def feedback_pipeline(keyword, aspects, num_results=10):
    urls = search_web(keyword, num_results)
    feedback = scrape_feedback(urls)
    if not feedback:
        return {"error": "No feedback found."}

    processed_feedback = preprocess_feedback(feedback)
    sentiments = analyze_sentiment(processed_feedback)
    
    # Sentiment breakdown (Positive, Neutral, Negative)
    sentiment_labels = {
        "Positive": len([s for s in sentiments if s > 0]),
        "Neutral": len([s for s in sentiments if s == 0]),
        "Negative": len([s for s in sentiments if s < 0]),
    }
    
    # Extract topics
    topics = extract_topics(processed_feedback)
    
    # Aspect-based Sentiment Analysis
    aspect_based_sentiments = aspect_based_sentiment(processed_feedback, aspects)
    
    # Location sentiment analysis
    location_sentiments = analyze_locations(feedback, sentiments)
    
    # Sort locations by most positive sentiment
    ranked_locations = sorted(
        location_sentiments.items(),
        key=lambda item: item[1]["positive"],
        reverse=True
    )

    return {
        "sentiments": sentiment_labels,
        "topics": topics,
        "aspect_based_sentiments": aspect_based_sentiments,
        "locations": ranked_locations,
    }
