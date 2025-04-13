import streamlit as st
import requests
import re
import datetime
from googlesearch import search  # pip install google-search-python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

st.set_page_config(page_title="Real-time Fake News & Fact Checker", layout="wide")
st.title("🌐 Real-time Fake News & Fact Checker")
st.write("Analyze text or URLs and attempt to verify public interest claims with real-time information.")
st.info("This tool uses a basic fake news detection model and Google search to check facts. Accuracy of fact-checking depends on search results.")

# ----------------------- Load Pre-trained Model -----------------------
# A basic model for fake news classification.
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

# Sample data to train a basic fake news detection model
internal_X_train = [
    "Breaking news: Local school announces holiday extension.",
    "Shocking report reveals alien sightings in remote areas.",
    "Study shows that regular exercise improves mental health.",
    "Controversial bill passed by the parliament amidst protests.",
    "Scientists discover a new species of deep-sea creature.",
    "The Prime Minister of India is Narendra Modi.",
    "The capital of France is Paris.",
    "Elephants are the largest land animals.",
    "Water boils at 100 degrees Celsius.",
    "The Earth is flat.",
    "India won the Cricket World Cup in 2011.",
    f"The current year is {datetime.datetime.now().year}.",
    "Vizianagaram is the capital of Andhra Pradesh."
]
internal_y_train = ["REAL", "FAKE", "REAL", "REAL", "REAL", "REAL", "REAL", "REAL", "REAL", "FAKE", "REAL", "REAL", "FAKE"]

# Vectorizing training data and training the model
X_train_vec = vectorizer.fit_transform(internal_X_train)
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_vec, internal_y_train)

# ----------------------- Prediction -----------------------
def predict_news(text):
    input_vec = vectorizer.transform([text])
    prediction = model.predict(input_vec)[0]
    confidence = model.decision_function(input_vec)[0]
    return prediction, abs(round(confidence, 2))

# ----------------------- URL Text Extraction (Without BeautifulSoup) -----------------------
def fetch_news_from_url(url):
    try:
        page = requests.get(url, timeout=10)
        text = page.text
        # Remove HTML tags
        clean_text = re.sub('<[^<]+?>', '', text)
        return clean_text[:5000]  # Limit text length
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching URL: {e}")
        return None
    except Exception as e:
        st.error(f"Error processing URL content: {e}")
        return None

# ----------------------- Google Search for Real-time Fact Checking -----------------------
def google_search(query):
    try:
        search_results = search(query, num_results=3)
        return search_results
    except Exception as e:
        st.error(f"Error during Google search: {e}")
        return []

# ----------------------- User Input -----------------------
user_input = st.text_area("Enter text or a news URL for analysis:")

# ----------------------- Analyze -----------------------
if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter text or a URL.")
    else:
        news_text = user_input
        if user_input.startswith("http://") or user_input.startswith("https://"):
            with st.spinner(f"Fetching content from {user_input}..."):
                news_text = fetch_news_from_url(user_input)
            if not news_text:
                st.warning("Could not retrieve content from the URL. Analyzing the URL itself.")
                news_text = user_input

        with st.spinner("Analyzing..."):
            # Predict if the news is real or fake using the pre-trained model
            prediction, conf = predict_news(news_text)

            if prediction == "FAKE":
                st.error(f"⚠ Likely FAKE with confidence: {conf:.2f}")
                # Search Google for fact-checking if the news is predicted to be fake
                st.subheader("Real-time Fact Check from Google Search:")
                search_results = google_search(f"fact check {news_text}")
                if search_results:
                    for idx, result in enumerate(search_results):
                        st.write(f"{idx+1}. {result}")
                else:
                    st.write("No relevant search results found.")
            else:
                st.success(f"✅ Likely REAL with confidence: {conf:.2f}")

        st.write(f"Confidence Score: {conf:.2f}")
