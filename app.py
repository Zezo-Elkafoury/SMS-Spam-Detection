import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

# Ensure required NLTK data is available
nltk.download('punkt_tab')
nltk.download('stopwords')

# Preprocessing function
ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    text = [i for i in text if i.isalnum()]
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    text = [ps.stem(i) for i in text]

    return " ".join(text)

# Page config
st.set_page_config(page_title="Spam Classifier", page_icon="🚫", layout="centered")

# Custom styling
st.markdown("""
    <style>
        .main {
            background-color: #f9f9f9;
        }
        .stTextArea > div > textarea {
            font-size: 16px;
            line-height: 1.6;
        }
        .result-box {
            padding: 1rem;
            border-radius: 10px;
            margin-top: 1rem;
            text-align: center;
            font-size: 1.5rem;
            font-weight: bold;
        }
        .spam {
            background-color: #ffdddd;
            color: #a70000;
        }
        .not-spam {
            background-color: #ddffdd;
            color: #006600;
        }
    </style>
""", unsafe_allow_html=True)

# App title
st.title("📩 Spam SMS Classifier")
st.subheader("Detect if a message is spam or legitimate in seconds")

# Input field
input_sms = st.text_area("✍️ Enter your message:")

# Predict button
if st.button('🔍 Analyze Message'):

    # Preprocess and predict
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    # Show result
    if result == 1:
        st.markdown('<div class="result-box spam">🚫 This message is <b>SPAM</b></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-box not-spam">✅ This message is <b>NOT SPAM</b></div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Built with ❤️ by Ziad Elkafoury · Stay safe from scams!")

