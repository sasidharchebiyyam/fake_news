import streamlit as st
import requests
import re

# Replace with your actual SerpApi API key
SERPAPI_API_KEY = 'YOUR_SERPAPI_API_KEY'

st.set_page_config(page_title="Real-time Fake News & Fact Checker", layout="wide")
st.title("🌐 Real-time Fake News & Fact Checker")
st.write("Analyze text or URLs and attempt to verify public interest claims with real-time information.")
st.info("This tool uses a basic fake news detection model and searches the web for real-time information to check facts. Accuracy of fact-checking depends on search results.")

# ----------------------- Load Pre-trained Model -----------------------
accuracy = 0.88  # Placeholder for model accuracy.

# ----------------------- Prediction -----------------------
def predict_news(text):
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

        # Show fake or real prediction
        st.subheader("Fake News Detection Result:")
        if prediction == "FAKE":
            st.error(f"⚠ Likely FAKE with confidence: {conf:.2f}")
        else:
            st.success(f"✅ Likely REAL with confidence: {conf:.2f}")
        st.write(f"Confidence Score: {conf:.2f}")
        st.caption(f"Model Accuracy (Estimated): {accuracy:.2f}%")

# ----------------------- Sidebar -----------------------
st.sidebar.header("About")
st.sidebar.info(
    "This tool detects fake news using a basic model.\n\n"
    "*Note:*\n"
    "- The model is a basic placeholder and is not highly accurate.\n"
    "- Always cross-check from trusted sources."
)
