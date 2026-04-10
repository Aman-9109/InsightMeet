import streamlit as st
import os
import sys
import matplotlib.pyplot as plt



@st.cache_resource
def load_transcriber():
    return Transcriber(model_size="base")

@st.cache_resource
def load_summarizer():
    return MeetingSummarizer()

@st.cache_resource
def load_sentiment():
    return SentimentAnalyzer()

@st.cache_resource
def load_topic_modeler():
    return TopicModeler(num_topics=2, num_keywords=5)

st.sidebar.title("⚙ Settings")
st.sidebar.write("Model Size: base")
st.sidebar.write("Topics: 2")
st.sidebar.write("Keywords per Topic: 5")


current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)


from src.transcription import Transcriber
from src.preprocessing import TextPreprocessor
from src.chunking import TextChunker
from src.summarization import MeetingSummarizer
from src.sentiment import SentimentAnalyzer
from src.action_items import ActionItemExtractor
from src.topic_modeling import TopicModeler


st.set_page_config(page_title="Meeting Insights Pipeline", layout="wide")

st.title("📊 Meeting Insights Pipeline")
st.markdown("---")

uploaded_file = st.file_uploader("Upload Meeting Audio", type=["wav", "mp3"])

if uploaded_file is not None:

    os.makedirs("../data/raw", exist_ok=True)
    file_path = os.path.join("../data/raw", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("File uploaded successfully ✅")
    st.audio(file_path)

    st.header("🎤 Transcription")

    with st.spinner("Transcribing audio..."):
        transcriber = load_transcriber()
        transcript = transcriber.transcribe(file_path)

    st.text_area("Transcript", transcript, height=200)

 
    processor = TextPreprocessor()
    cleaned_text = processor.full_preprocess(transcript)

    chunker = TextChunker(max_words=200)
    chunks = chunker.create_chunks(cleaned_text)

  
    st.header("📄 Meeting Summary")

    with st.spinner("Generating summary..."):
        
        summarizer = load_summarizer()
        chunk_summaries = summarizer.summarize_chunks(chunks)
        final_summary = summarizer.refine_summary(chunk_summaries)

    st.success(final_summary)

   
    st.header("😊 Sentiment Analysis")

    analyzer = load_sentiment()
    sentiment_results = analyzer.analyze_chunks(chunks)
    overall = analyzer.aggregate_sentiment(sentiment_results)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Overall Sentiment", overall["overall_sentiment"])

    with col2:
        st.metric("Total Chunks", len(chunks))

    # Bar Chart
    positive = overall["positive_chunks"]
    negative = overall["negative_chunks"]

    fig = plt.figure()
    plt.bar(["Positive", "Negative"], [positive, negative])
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Chunks")

    st.pyplot(fig)

 
    st.header("📌 Action Items & Decisions")

    extractor = ActionItemExtractor()
    action_data = extractor.structured_output(transcript)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Action Items")
        if action_data["action_items"]:
            for item in action_data["action_items"]:
                st.write("•", item)
        else:
            st.info("No action items detected")

    with col2:
        st.subheader("Decisions")
        if action_data["decisions"]:
            for decision in action_data["decisions"]:
                st.write("•", decision)
        else:
            st.info("No decisions detected")


    st.header("📊 Topics & Keywords")

    topic_modeler = load_topic_modeler()
    topic_results = topic_modeler.structured_output(cleaned_text)

    st.subheader("Top Keywords")
    st.write(", ".join(topic_results["keywords"]))

    st.subheader("Identified Topics")

    for idx, topic in enumerate(topic_results["topics"]):
        st.write(f"Topic {idx+1}: {', '.join(topic)}")
