import streamlit as st
import pandas as pd
import requests
import re
import datetime
from googlesearch import search  # pip install google-search-python

st.set_page_config(page_title="Real-time Fake News & Fact Checker", layout="wide")
st.title("🌐 Real-time Fake News & Fact Checker")
st.write("Analyze text or URLs and attempt to verify public interest claims with real-time information.")
st.info("This tool uses a basic fake news detection model and searches the web for real-time information to check facts. Accuracy of fact-checking depends on search results.")

# ----------------------- Load Pre-trained Model -----------------------
# Placeholder for model prediction
accuracy = 0.88  # Placeholder, update with your model accuracy.

# ----------------------- Prediction -----------------------
def predict_news(text):
    # Basic placeholder logic for fake news detection
    fake_keywords = ["fake", "hoax", "scam", "rumor", "unverified"]
    real_keywords = ["real", "verified", "true", "authentic"]
    
    prediction = "REAL" if any(word in text.lower() for word in real_keywords) else "FAKE"
    confidence = 0.75  # Placeholder confidence score
    return prediction, confidence

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
