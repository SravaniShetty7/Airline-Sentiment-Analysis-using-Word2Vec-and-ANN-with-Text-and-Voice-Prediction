import streamlit as st
import os
import re
import html
import pickle
import numpy as np
import pandas as pd
import speech_recognition as sr

from gensim.models import Word2Vec
from tensorflow.keras.models import load_model

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Airline Sentiment AI",
    page_icon="✈️",
    layout="wide"
)


# ============================================================
# NLTK
# ============================================================

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# ============================================================
# PATHS
# ============================================================

BASE_PATH = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_PATH,
    "models",
    "sentiment_ann.keras"
)

WORD2VEC_PATH = os.path.join(
    BASE_PATH,
    "models",
    "word2vec.model"
)

ENCODER_PATH = os.path.join(
    BASE_PATH,
    "models",
    "label_encoder.pkl"
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"sentiment_ann.keras not found:\n{MODEL_PATH}"
        )

    if not os.path.exists(WORD2VEC_PATH):
        raise FileNotFoundError(
            f"word2vec.model not found:\n{WORD2VEC_PATH}"
        )

    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(
            f"label_encoder.pkl not found:\n{ENCODER_PATH}"
        )

    ann_model = load_model(
        MODEL_PATH
    )

    word2vec_model = Word2Vec.load(
        WORD2VEC_PATH
    )

    with open(
        ENCODER_PATH,
        "rb"
    ) as f:

        label_encoder = pickle.load(f)

    return (
        ann_model,
        word2vec_model,
        label_encoder
    )


try:

    (
        ann_model,
        word2vec_model,
        label_encoder
    ) = load_models()

except Exception as e:

    st.error(
        "❌ Model loading failed"
    )

    st.code(
        str(e)
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("✈️ AirSentiment")

    st.write(
        "AI-powered airline customer "
        "sentiment analysis"
    )

    st.divider()

    st.subheader("🤖 Model")

    st.write("Word2Vec + ANN")

    st.write(
        "Input Features: **100**"
    )

    st.write(
        "Classes: **3**"
    )

    st.divider()

    st.subheader("🎯 Classes")

    st.write("😊 Positive")
    st.write("😐 Neutral")
    st.write("😞 Negative")

    st.divider()

    st.subheader("⚙️ Best ANN Settings")

    st.write(
        "Learning Rate: **0.0005**"
    )

    st.write(
        "Batch Size: **64**"
    )

    st.write(
        "Epochs: **30**"
    )

    st.divider()

    st.success(
        "Models loaded successfully"
    )


# ============================================================
# PREPROCESSING
# ============================================================

def expand_contractions(text):

    contractions = {

        "can't": "cannot",
        "won't": "will not",
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",

        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",

        "haven't": "have not",
        "hasn't": "has not",
        "hadn't": "had not",

        "wouldn't": "would not",
        "couldn't": "could not",
        "shouldn't": "should not",

        "i'm": "i am",
        "you're": "you are",
        "we're": "we are",
        "they're": "they are",

        "it's": "it is",
        "that's": "that is",
        "there's": "there is"
    }

    for contraction, replacement in contractions.items():

        text = re.sub(
            r"\b" +
            re.escape(contraction) +
            r"\b",
            replacement,
            text
        )

    return text


def preprocess_text(text):

    text = str(text)

    text = text.lower()

    text = expand_contractions(
        text
    )

    text = html.unescape(
        text
    )

    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    tokens = text.split()

    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return tokens


# ============================================================
# WORD2VEC VECTOR
# ============================================================

def document_vector(tokens):

    vectors = []

    for word in tokens:

        if word in word2vec_model.wv:

            vectors.append(
                word2vec_model.wv[word]
            )

    if not vectors:

        return np.zeros(
            word2vec_model.vector_size,
            dtype=np.float32
        )

    return np.mean(
        vectors,
        axis=0
    ).astype(
        np.float32
    )


# ============================================================
# PREDICTION
# ============================================================

def predict_sentiment(text):

    tokens = preprocess_text(
        text
    )

    vector = document_vector(
        tokens
    )

    vector = vector.reshape(
        1,
        -1
    )

    probabilities = ann_model.predict(
        vector,
        verbose=0
    )[0]

    index = np.argmax(
        probabilities
    )

    sentiment = (
        label_encoder.inverse_transform(
            [index]
        )[0]
    )

    confidence = probabilities[index]

    return (
        sentiment,
        confidence,
        probabilities
    )


# ============================================================
# RESULT DISPLAY
# ============================================================

def show_result(
    sentiment,
    confidence,
    probabilities
):

    st.subheader(
        "🎯 Prediction"
    )

    if sentiment == "Positive":

        st.success(
            f"😊 POSITIVE   |   "
            f"{confidence * 100:.2f}%"
        )

    elif sentiment == "Negative":

        st.error(
            f"😞 NEGATIVE   |   "
            f"{confidence * 100:.2f}%"
        )

    else:

        st.info(
            f"😐 NEUTRAL   |   "
            f"{confidence * 100:.2f}%"
        )

    st.subheader(
        "📊 Sentiment Scores"
    )

    chart_data = pd.DataFrame(
        {
            "Sentiment":
                label_encoder.classes_,

            "Score":
                probabilities * 100
        }
    )

    chart_data = chart_data.sort_values(
        "Score",
        ascending=False
    )

    st.bar_chart(
        chart_data.set_index(
            "Sentiment"
        ),
        height=300
    )

    st.subheader(
        "📌 Scores"
    )

    columns = st.columns(3)

    for i, label in enumerate(
        label_encoder.classes_
    ):

        score = probabilities[i] * 100

        if label == "Positive":
            columns[0].metric(
                "😊 Positive",
                f"{score:.2f}%"
            )

        elif label == "Neutral":
            columns[1].metric(
                "😐 Neutral",
                f"{score:.2f}%"
            )

        elif label == "Negative":
            columns[2].metric(
                "😞 Negative",
                f"{score:.2f}%"
            )


# ============================================================
# HEADER
# ============================================================

st.title(
    "✈️ Airline Sentiment AI"
)

st.write(
    "Analyze airline customer reviews using "
    "**Word2Vec + Artificial Neural Network**."
)

st.caption(
    "📝 Text Analysis   •   🎙️ Voice Analysis"
)

st.divider()


# ============================================================
# TEXT ANALYSIS
# ============================================================

st.header(
    "📝 Text Sentiment Analysis"
)

text_input = st.text_area(
    "Enter customer review",
    placeholder=(
        "Example: The flight was excellent "
        "and the staff were very friendly."
    ),
    height=140
)

if st.button(
    "🔍 Analyze Text",
    use_container_width=True
):

    if not text_input.strip():

        st.warning(
            "Please enter a review."
        )

    else:

        (
            sentiment,
            confidence,
            probabilities
        ) = predict_sentiment(
            text_input
        )

        show_result(
            sentiment,
            confidence,
            probabilities
        )


# ============================================================
# VOICE SECTION
# ============================================================

st.divider()

st.header(
    "🎙️ Voice Sentiment Analysis"
)

st.write(
    "Record your airline review below."
)

st.info(
    "🎙️ Speak clearly after clicking the microphone. "
    "After recording, press ▶️ to listen to your voice."
)


# ============================================================
# MICROPHONE RECORDING
# ============================================================

audio_value = st.audio_input(
    "🎙️ Record your voice",
    key="airline_voice"
)


if audio_value is not None:

    # --------------------------------------------------------
    # GET AUDIO
    # --------------------------------------------------------

    audio_bytes = audio_value.getvalue()

    # --------------------------------------------------------
    # AUDIO INFORMATION
    # --------------------------------------------------------

    st.subheader(
        "🔊 Your Recorded Voice"
    )

    st.write(
        f"Recording received: "
        f"**{len(audio_bytes):,} bytes**"
    )

    # --------------------------------------------------------
    # PLAY AUDIO
    # --------------------------------------------------------

    st.audio(
        audio_bytes,
        format="audio/wav"
    )

    st.success(
        "✅ Recording captured."
    )

    # --------------------------------------------------------
    # SPEECH TO TEXT
    # --------------------------------------------------------

    temp_file = os.path.join(
        BASE_PATH,
        "voice_recording.wav"
    )

    try:

        with open(
            temp_file,
            "wb"
        ) as f:

            f.write(
                audio_bytes
            )

        recognizer = sr.Recognizer()

        with sr.AudioFile(
            temp_file
        ) as source:

            audio_data = (
                recognizer.record(
                    source
                )
            )

        st.info(
            "🔄 Converting voice to text..."
        )

        recognized_text = (
            recognizer.recognize_google(
                audio_data
            )
        )

        st.success(
            "✅ Speech converted successfully."
        )

        # ----------------------------------------------------
        # SHOW RECOGNIZED TEXT
        # ----------------------------------------------------

        st.subheader(
            "📝 Recognized Text"
        )

        st.write(
            recognized_text
        )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        (
            sentiment,
            confidence,
            probabilities
        ) = predict_sentiment(
            recognized_text
        )

        show_result(
            sentiment,
            confidence,
            probabilities
        )

    except sr.UnknownValueError:

        st.error(
            "❌ I could not understand the speech."
        )

    except sr.RequestError:

        st.error(
            "❌ Google speech recognition "
            "service is unavailable."
        )

    except Exception as e:

        st.error(
            f"❌ Voice processing error:\n{e}"
        )

    finally:

        if os.path.exists(
            temp_file
        ):

            try:
                os.remove(
                    temp_file
                )
            except:
                pass


# ============================================================
# MICROPHONE TEST
# ============================================================

st.divider()

with st.expander(
    "🎙️ Microphone Troubleshooting"
):

    st.write(
        "If you cannot hear the recording, "
        "check the following:"
    )

    st.write(
        "1. Make sure microphone permission is enabled."
    )

    st.write(
        "2. Make sure the correct microphone is selected."
    )

    st.write(
        "3. After recording, press ▶️ on "
        "the audio player."
    )

    st.write(
        "4. Check your computer/browser volume."
    )

    st.write(
        "5. Make sure the browser tab is not muted."
    )

    st.write(
        "6. In Chrome, allow microphone access "
        "for localhost."
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

with st.expander(
    "ℹ️ Model Information"
):

    st.write(
        "**Model:** Artificial Neural Network"
    )

    st.write(
        "**Word Embedding:** Word2Vec"
    )

    st.write(
        "**Document Vector:** Mean Word2Vec"
    )

    st.write(
        "**Input Features:** 100"
    )

    st.write(
        "**Learning Rate:** 0.0005"
    )

    st.write(
        "**Batch Size:** 64"
    )

    st.write(
        "**Epochs:** 30"
    )

    st.write(
        "**Classes:** Negative, Neutral, Positive"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "✈️ Airline Sentiment AI | "
    "Word2Vec + ANN | Text & Voice"
)