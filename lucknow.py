import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
import plotly.graph_objects as go
import plotly.express as px
import mysql.connector
from mysql.connector import Error
from datetime import datetime
import time
import tempfile
import speech_recognition as sr
from pydub import AudioSegment
from streamlit_mic_recorder import mic_recorder

# ---------------- DATABASE CONFIG ----------------


DB_CONFIG = {
    "host": st.secrets["host"],
    "user": st.secrets["user"],
    "password": st.secrets["password"],
    "database": st.secrets["database"],
    "port": int(st.secrets["port"])
}


# ---------------- DATABASE FUNCTIONS ----------------
def create_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        if conn.is_connected():
            return conn
    except Error as e:
        st.error(f"MySQL connection error: {e}")
    return None


def init_db():
    try:
        conn = mysql.connector.connect(
            host=DB_CONFIG["host"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"]
        )
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
        cursor.execute(f"USE {DB_CONFIG['database']}")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS consultations (
                id INT AUTO_INCREMENT PRIMARY KEY,
                serial_no VARCHAR(100),
                comment TEXT,
                sentiment VARCHAR(20),
                confidence FLOAT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    except Error as e:
        st.error(f"Database init error: {e}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()


def insert_record(serial_no, comment, sentiment, confidence):
    conn = create_connection()
    if conn:
        try:
            cursor = conn.cursor()
            sql = """INSERT INTO consultations 
                     (serial_no, comment, sentiment, confidence, timestamp) 
                     VALUES (%s, %s, %s, %s, %s)"""
            cursor.execute(sql, (serial_no, comment, sentiment, confidence, datetime.now()))
            conn.commit()
        except Error as e:
            st.error(f"Insert failed: {e}")
        finally:
            cursor.close()
            conn.close()


def fetch_all_records():
    conn = create_connection()
    if not conn:
        return pd.DataFrame()
    try:
        query = "SELECT * FROM consultations ORDER BY timestamp DESC"
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def fetch_by_serial(serial_no):
    conn = create_connection()
    if not conn:
        return pd.DataFrame()
    try:
        query = f"SELECT * FROM consultations WHERE serial_no='{serial_no}'"
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Error fetching record: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

# ---------------- SENTIMENT ANALYZER ----------------
def analyze_sentiment(text):
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    if polarity > 0.05:
        sentiment = "Positive"
    elif polarity < -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment, round(abs(polarity), 2)

# ---------------- STREAMLIT SETUP ----------------
st.set_page_config(page_title="Sentiment Analyzer", layout="wide")
init_db()

# ---------------- CSS ----------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&family=Cinzel+Decorative:wght@700&family=Dancing+Script:wght@700&display=swap" rel="stylesheet">

<style>
.stApp { background-color: #faf6f2; font-family: 'Poppins', sans-serif !important; }
[data-testid="stTabs"] button {
    background-color: #4A2C2A !important;
    color: white !important;
    border-radius: 10px 10px 0 0 !important;
    font-weight: 600 !important;
    font-size: 18px !important;
    margin-right: 5px !important;
}
[data-testid="stTabs"] button[aria-selected="true"] { border-bottom: 4px solid #E0B37D !important; }

.title { font-family: 'Cinzel Decorative', serif; font-size: 70px; font-weight: 700; color: #4A2C2A; text-align: center; }
.subtitle { text-align:center; color:#593C39; font-size:28px; }

@keyframes moveText { 0% { transform: translateX(100%); } 100% { transform: translateX(-100%); } }
.moving-text { width: 100%; overflow: hidden; white-space: nowrap; color: #4A2C2A; font-size: 22px; margin-top: 10px; }
.moving-text span { display: inline-block; padding-left: 100%; animation: moveText 20s linear infinite; }

.carousel-container { position: relative; width: 90%; height: 450px; margin: 0 auto; overflow: hidden; border-radius: 16px; box-shadow: 0 6px 20px rgba(0,0,0,0.1); }
.carousel-image { width: 100%; height: 450px; object-fit: cover; border-radius: 16px; position: absolute; animation: fade 15s infinite; }
.carousel-image:nth-child(1){animation-delay:0s;} .carousel-image:nth-child(2){animation-delay:5s;} .carousel-image:nth-child(3){animation-delay:10s;}
@keyframes fade { 0%{opacity:0;}10%{opacity:1;}33%{opacity:1;}43%{opacity:0;}100%{opacity:0;} }

.info-card { background:white; padding:30px; border-radius:16px; box-shadow:4px 4px 12px rgba(0,0,0,0.08); border:2px solid #e3c49b; text-align:center; }

.metric-card { background:white; border-radius:14px; padding:24px; box-shadow:0 6px 18px rgba(0,0,0,0.06); border:2px solid #e9d5bf; text-align:center; }
.metric-card h4 { margin:0; font-family:'Dancing Script', cursive; color:#4b2b28; font-size:28px; }
.metric-value { font-size:36px; font-weight:800; color:#4b2b28; font-family:'Poppins'; }
</style>
""", unsafe_allow_html=True)

# ---------------- TABS ----------------
tabs = st.tabs(["🏠 Home", "📊 Insights", "📈 Reports"])

# ---------------- HOME ----------------
with tabs[0]:
    st.markdown("""
    <h1 style='text-align:center; color:#4A2C2A; font-size:70px; font-family:"Cinzel Decorative", serif !important; font-weight:700;'>
    Sentiment Analyzer Dashboard
    </h1>
    """, unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Analyze feedback with AI-powered insights</p>", unsafe_allow_html=True)
    st.markdown("<div class='moving-text'><span>Decoding Feedback • Empowering Better Governance • Enhancing Customer Experience</span></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="carousel-container">
        <img src="https://images.unsplash.com/photo-1522202176988-66273c2fd55f" class="carousel-image">
        <img src="https://6663efc8bb.clvaw-cdnwnd.com/dd9f2c9a65e08d0763028ae8eeadaab9/200001508-82a0d82a0f/Business%20Analytices%202.jpg?ph=6663efc8bb" class="carousel-image">
        <img src="https://d1krbhyfejrtpz.cloudfront.net/blog/wp-content/uploads/2023/03/01184121/How-to-Develop-a-Custom-Sentiment-Analysis-Tool.jpg" class="carousel-image">
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.markdown("<div class='info-card'><h3>🧠 What is it?</h3><p>An AI-powered dashboard to analyze sentiments.</p></div>", unsafe_allow_html=True)
    c2.markdown("<div class='info-card'><h3>📊 How does it work?</h3><p>Input comments or upload files — AI detects sentiment instantly.</p></div>", unsafe_allow_html=True)
    c3.markdown("<div class='info-card'><h3>🎯 Why use it?</h3><p>Helps identify improvement areas and customer mood trends.</p></div>", unsafe_allow_html=True)

# ---------------- CHATBOT ----------------ii
# ---------------- CHATBOT OVERLAY (Fixed Floating Overlay on Home) ----------------
import streamlit as st

# Initialize chatbot visibility
if "chat_visible" not in st.session_state:
    st.session_state.chat_visible = False

# Custom CSS for floating chat button + overlay panel
st.markdown("""
<style>
/* Floating Chat Button */
.chat-button {
    position: fixed;
    bottom: 25px;
    right: 25px;
    background-color: #7b4b36;
    color: white;
    font-size: 26px;
    border-radius: 50%;
    width: 60px;
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    cursor: pointer;
    transition: all 0.3s ease;
    z-index: 9999;
}
.chat-button:hover {
    background-color: #9b604a;
    transform: scale(1.05);
}

/* Chat Window */
.chat-overlay {
    position: fixed;
    bottom: 100px;
    right: 35px;
    width: 340px;
    background: #ffffff;
    border: 2px solid #c19ae6;
    border-radius: 18px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.25);
    z-index: 10000;
    animation: fadeIn 0.4s ease-in-out;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}
.chat-header {
    background-color: #a88bff;
    color: white;
    padding: 10px 16px;
    border-radius: 18px 18px 0 0;
    font-weight: bold;
    text-align: center;
}
.chat-body {
    padding: 12px 16px;
    max-height: 230px;
    overflow-y: auto;
    font-size: 15px;
    color: #4b2b28;
}
.chat-input {
    display: flex;
    padding: 8px 10px 12px 10px;
    border-top: 1px solid #eee;
}
.chat-input input {
    flex: 1;
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 6px 10px;
    outline: none;
}
.chat-input button {
    background: #a88bff;
    color: white;
    border: none;
    margin-left: 8px;
    padding: 8px 12px;
    border-radius: 10px;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

# Floating button (always visible)
st.markdown(f"""
    <div class="chat-button" onclick="window.parent.postMessage({{'type':'toggleChat'}}, '*')">💬</div>
    <script>
    window.addEventListener('message', (event) => {{
        if (event.data.type === 'toggleChat') {{
            const iframe = window.parent.document.querySelector('iframe');
            if (iframe) iframe.contentWindow.postMessage({{'chat_toggle': true}}, '*');
        }}
    }});
    </script>
""", unsafe_allow_html=True)

# Streamlit toggle logic
if st.button("💬 Toggle Sentify Bot (Local Test)"):
    st.session_state.chat_visible = not st.session_state.chat_visible
    st.rerun()

# Display chatbot overlay
if st.session_state.chat_visible:
    st.markdown("""
    <div class="chat-overlay">
        <div class="chat-header">🤖 Sentify Bot</div>
        <div class="chat-body">
            <p><b>HIII HOW MAY I HELP YOU…</b></p>
            <p>💡 Try asking: "How to upload data?" or "Where is the insights tab?"</p>
        </div>
        <div class="chat-input">
            <input type="text" placeholder="Type your message..." disabled />
            <button>📤</button>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- INSIGHTS ----------------
with tabs[1]:
    st.markdown("""
    <h2 style='text-align:center; color:#4A2C2A; font-size:80px; font-family:"Cinzel Decorative", serif !; font-weight:700;'>
    SENTIMENT INSIGHTS
    </h2>
    """, unsafe_allow_html=True)
    df_all = fetch_all_records()
    total = len(df_all)
    pos = len(df_all[df_all["sentiment"] == "Positive"])
    neu = len(df_all[df_all["sentiment"] == "Neutral"])
    neg = len(df_all[df_all["sentiment"] == "Negative"])

    c1, c2, c3, c4 = st.columns(4)
    for c, l, v in zip([c1, c2, c3, c4], ["Total", "Positive", "Neutral", "Negative"], [total, pos, neu, neg]):
        c.markdown(f"<div class='metric-card'><h4>{l}</h4><div class='metric-value'>{v}</div></div>", unsafe_allow_html=True)

    st.write("---")

    serial_no = st.text_input("Serial Number")
    speech_text = st.session_state.get("speech_text", "")
    colA, colB = st.columns([3, 1])
    comment = colA.text_area("Enter comment...", value=speech_text, key="comment_box", height=120)

    # 🎙️ Browser-based speech input (works on Streamlit Cloud)
if colB.button("🎙️ Speak Now"):
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("🎧 Listening...")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            audio = recognizer.listen(source, phrase_time_limit=10)
        text_from_speech = recognizer.recognize_google(audio, language="en-IN")
        st.session_state["speech_text"] = text_from_speech
        st.success("✅ Recognized and filled in the text box.")
        st.rerun()
    except Exception as e:
        st.error("🎤 Microphone not supported in this environment.")


    if st.button("Analyze Sentiment"):
        comment = st.session_state.get("speech_text", comment)
        if comment.strip():
            sentiment, confidence = analyze_sentiment(comment)
            insert_record(serial_no or "", comment, sentiment, confidence)
            st.success(f"Sentiment: {sentiment} | Confidence: {confidence}")
            st.session_state["speech_text"] = ""
        else:
            st.warning("Please enter or record a comment.")

    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if uploaded:
        df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
        df["Sentiment"], df["Confidence"] = zip(*df["comment"].astype(str).map(analyze_sentiment))
        for _, r in df.iterrows():
            insert_record(str(r.get("serial_no", "")), r["comment"], r["Sentiment"], r["Confidence"])
        st.success(f"✅ {len(df)} records analyzed & saved!")

    st.write("---")

    if not df_all.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("☁️ Word Cloud")
            text = " ".join(df_all["comment"].astype(str))
            wc = WordCloud(width=600, height=400, background_color="white").generate(text)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

        with col2:
            st.subheader("📊 Average Confidence by Sentiment")
            avg_conf = df_all.groupby("sentiment")["confidence"].mean().reset_index()
            fig, ax = plt.subplots()
            sns.barplot(data=avg_conf, x="sentiment", y="confidence", palette=["green", "orange", "red"], ax=ax)
            ax.set_ylim(0, 1)
            st.pyplot(fig)

        col_g1, col_g2 = st.columns(2)
        latest = df_all.iloc[0]
        val, sentiment = latest["confidence"], latest["sentiment"]
        color_map = {"Positive": "green", "Neutral": "orange", "Negative": "red"}
        with col_g1:
            st.subheader("🧭 Sentiment Gauge")
            fig_g = go.Figure(go.Indicator(mode="gauge+number", value=val * 100, title={'text': sentiment},
                        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color_map.get(sentiment, "gray")}}))
            st.plotly_chart(fig_g, use_container_width=True)
        with col_g2:
            st.subheader("🍰 Sentiment Distribution")
            pie_df = df_all["sentiment"].value_counts().reset_index()
            pie_df.columns = ["Sentiment", "Count"]
            fig_pie = px.pie(pie_df, names="Sentiment", values="Count",
                             color="Sentiment",
                             color_discrete_map=color_map)
            st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("📈 Sentiment Timeline")
        df_all["date"] = pd.to_datetime(df_all["timestamp"]).dt.date
        trend = df_all.groupby(["date", "sentiment"]).size().reset_index(name="count")
        fig_tl = px.line(trend, x="date", y="count", color="sentiment", color_discrete_map=color_map)
        st.plotly_chart(fig_tl, use_container_width=True)

        st.dataframe(df_all[["serial_no", "comment", "sentiment", "confidence", "timestamp"]])
    else:
        st.info("No data yet.")

# ---------------- REPORTS ----------------
with tabs[2]:
    st.markdown("""
    <h2 style='text-align:center; color:#4A2C2A; font-size:72px ; font-family:"Cinzel Decorative", serif; font-weight:700;'>
    Reports Dashboard
    </h2>
    """, unsafe_allow_html=True)
    df = fetch_all_records()

    if not df.empty:
        col1, col2 = st.columns([2, 1])
        with col1:
            serial_search = st.text_input("🔍 Enter Serial Number:")
        with col2:
            sentiment_filter = st.selectbox("🧭 Filter by Sentiment:", ["All", "Positive", "Neutral", "Negative"])

        if serial_search:
            df = df[df["serial_no"].str.contains(serial_search, case=False, na=False)]

        if sentiment_filter != "All":
            df = df[df["sentiment"] == sentiment_filter]

        st.dataframe(df)
    else:
        st.info("No records available.")

    st.write("---")
    st.markdown("<p style='text-align:center; color:#9c6d33;'>© 2025 Sentiment Analyzer | Designed with ❤️</p>", unsafe_allow_html=True)


