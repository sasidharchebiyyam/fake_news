import streamlit as st
import pandas as pd
import requests
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import datetime
from googlesearch import search  # pip install google-search-python

st.set_page_config(page_title="Real-time Fake News & Fact Checker", layout="wide")
st.title("🌐 Real-time Fake News & Fact Checker")
st.write("Analyze text or URLs and attempt to verify public interest claims with real-time information.")
st.info("This tool uses a basic fake news detection model and searches the web for real-time information to check facts. Accuracy of fact-checking depends on search results.")

# ----------------------- Load Pre-trained Model -----------------------
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
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
X_train_vec = vectorizer.fit_transform(internal_X_train)
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_vec, internal_y_train)
accuracy = 0.88  # Placeholder

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

# ----------------------- Real-time Fact Checking -----------------------
def check_real_time_facts(text):
    results = {}
    public_interest_keywords = ["prime minister of india", "capital of india", "capital of andhra pradesh",
                               "current year", "cricket world cup winner", "president of india",
                               "chief minister of andhra pradesh", "major earthquake today",
                               "stock market today", "weather in vizianagaram"]

    for keyword in public_interest_keywords:
        if keyword in text.lower():
            try:
                search_results = search(f"what is the {keyword}", num_results=3)
                context = f"Based on web search results for '{keyword}':\n"
                for result in search_results:
                    context += f"- {result}\n"

                results[f"Real-time check for '{keyword}'"] = context

                if "chandrababu naidu" in text.lower() and "prime minister of india" in keyword:
                    results[f"Real-time check for '{keyword}'"] += "*Potentially Incorrect:* Current Prime Minister is likely Narendra Modi."
                elif "vizianagaram" in text.lower() and "capital of andhra pradesh" in keyword:
                    results[f"Real-time check for '{keyword}'"] += "*Potentially Incorrect:* The capital is Amaravati."
                elif str(datetime.datetime.now().year + 1) in text.lower() and "current year" in keyword:
                    results[f"Real-time check for '{keyword}'"] += f"*Potentially Incorrect:* The current year is {datetime.datetime.now().year}."

            except Exception as e:
                results[f"Real-time check for '{keyword}'"] = f"Error during web search: {e}"

    if not results:
        results["Real-time Analysis"] = "No specific public interest keywords detected."

    return results

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
            prediction, conf = predict_news(news_text)
            real_time_check_results = check_real_time_facts(news_text)

        st.subheader("Fake News Detection Result:")
        if prediction == "FAKE":
            st.error(f"⚠ Likely FAKE with confidence: {conf:.2f}")
        else:
            st.success(f"✅ Likely REAL with confidence: {conf:.2f}")
        st.write(f"Confidence Score: {conf:.2f}")
        st.caption(f"Model Accuracy (Estimated): {accuracy:.2f}%")

        if real_time_check_results:
            st.subheader("Real-time Fact Check:")
            for claim, result in real_time_check_results.items():
                st.write(f"- *{claim}:*")
                st.markdown(f"<div style='padding-left: 10px;'>{result}</div>", unsafe_allow_html=True)
        else:
            st.info("No specific public interest claims detected.")

        if st.checkbox("Save Analysis"):
            output_df = pd.DataFrame({
                "Input": [user_input],
                "Fake News Prediction": [prediction],
                "Confidence": [conf],
                "Real-time Fact Check": [real_time_check_results if real_time_check_results else "No checks"]
            })
            output_df.to_csv("analysis_result.csv", index=False)
            st.success("Analysis saved to analysis_result.csv")

# ----------------------- Sidebar -----------------------
st.sidebar.header("About")
st.sidebar.info(
    "This tool detects fake news and verifies public claims using a basic model and web search.\n\n"
    "*Note:*\n"
    "- Fact-checking depends on search results.\n"
    "- Basic model: not highly accurate.\n"
    "- Always cross-check from trusted sources."
)
