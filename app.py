import streamlit as st
import re
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

st.title("Product Review Sentiment Analysis")
st.write("This project analyzes the sentiment of product reviews using VADER and TextBlob.")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Analyze Review", "Batch Analysis", "About"])

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)
    compound = score['compound']
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return label, compound, score

def get_textblob(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    if polarity > 0.1:
        label = "Positive"
    elif polarity < -0.1:
        label = "Negative"
    else:
        label = "Neutral"
    return label, polarity, subjectivity


# ---- HOME PAGE ----
if page == "Home":
    st.header("Project Overview")
    st.write("Dataset: Amazon Product Reviews")
    st.write("Total Reviews: 363,261")
    st.write("Unique Products: 1,175")
    st.write("Columns: Product Name, Price, Rating, Review, Summary")

    st.subheader("Sentiment Categories")
    st.write("- Positive: Rating 4 or 5")
    st.write("- Neutral: Rating 3")
    st.write("- Negative: Rating 1 or 2")

    st.subheader("Rating Distribution (from dataset)")
    ratings = [1, 2, 3, 4, 5]
    counts = [14832, 12041, 22187, 64523, 249678]
    fig, ax = plt.subplots()
    ax.bar([str(r) + " Star" for r in ratings], counts, color=['red','orange','yellow','skyblue','green'])
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of Reviews")
    ax.set_title("Reviews per Rating")
    st.pyplot(fig)

    st.subheader("Methods Used")
    st.write("1. VADER - Rule based sentiment analyzer")
    st.write("2. TextBlob - Polarity and subjectivity based analyzer")
    st.write("3. Logistic Regression - Trained on TF-IDF features")
    st.write("4. Naive Bayes - Trained on CountVectorizer features")


# ---- ANALYZE REVIEW ----
elif page == "Analyze Review":
    st.header("Analyze a Single Review")
    st.write("Type a product review below and click Analyze.")

    review = st.text_area("Enter your review here", height=120)

    if st.button("Analyze"):
        if review.strip() == "":
            st.warning("Please enter some text first.")
        else:
            st.subheader("Results")

            vader_label, vader_compound, vader_scores = get_vader(review)
            tb_label, tb_polarity, tb_subjectivity = get_textblob(review)

            col1, col2 = st.columns(2)

            with col1:
                st.write("**VADER Result**")
                st.write("Sentiment:", vader_label)
                st.write("Compound Score:", round(vader_compound, 4))
                st.write("Positive:", round(vader_scores['pos'], 3))
                st.write("Neutral:", round(vader_scores['neu'], 3))
                st.write("Negative:", round(vader_scores['neg'], 3))

            with col2:
                st.write("**TextBlob Result**")
                st.write("Sentiment:", tb_label)
                st.write("Polarity:", round(tb_polarity, 4))
                st.write("Subjectivity:", round(tb_subjectivity, 4))

            st.write("---")

            fig, ax = plt.subplots()
            labels = ['Positive', 'Neutral', 'Negative']
            values = [vader_scores['pos'], vader_scores['neu'], vader_scores['neg']]
            ax.bar(labels, values, color=['green', 'gray', 'red'])
            ax.set_title("VADER Score Breakdown")
            ax.set_ylabel("Score")
            st.pyplot(fig)


# ---- BATCH ANALYSIS ----
elif page == "Batch Analysis":
    st.header("Batch Review Analysis")
    st.write("Enter multiple reviews, one per line. It will analyze all of them.")

    default = "This product is very good and works perfectly.\nTerrible quality, broke after one day.\nIt is okay nothing special.\nLoved it, highly recommended.\nNot worth the money at all."
    reviews_input = st.text_area("Enter reviews (one per line)", value=default, height=180)

    method = st.selectbox("Select method", ["VADER", "TextBlob"])

    if st.button("Analyze All"):
        reviews = [r.strip() for r in reviews_input.strip().split('\n') if r.strip()]

        if not reviews:
            st.warning("Please enter at least one review.")
        else:
            import pandas as pd
            sentiments = []
            for r in reviews:
                if method == "VADER":
                    label, score, _ = get_vader(r)
                    sentiments.append({"Review": r[:60], "Sentiment": label, "Score": round(score, 4)})
                else:
                    label, pol, sub = get_textblob(r)
                    sentiments.append({"Review": r[:60], "Sentiment": label, "Polarity": round(pol, 4)})

            df = pd.DataFrame(sentiments)
            st.write("Results:")
            st.dataframe(df)

            st.write("Sentiment Count:")
            count = df['Sentiment'].value_counts()
            fig, ax = plt.subplots()
            ax.bar(count.index, count.values, color=['green','gray','red'][:len(count)])
            ax.set_title("Sentiment Distribution")
            ax.set_ylabel("Count")
            st.pyplot(fig)

            csv = df.to_csv(index=False)
            st.download_button("Download Results", csv, "results.csv", "text/csv")


# ---- ABOUT ----
elif page == "About":
    st.header("About This Project")
    st.write("This is a sentiment analysis project built as part of a data science course.")
    st.write("")
    st.write("Dataset: Amazon Product Reviews")
    st.write("Language: Python")
    st.write("Libraries used: Pandas, NumPy, Matplotlib, Seaborn, NLTK, VADER, TextBlob, Scikit-learn, Streamlit, WordCloud")
    st.write("")
    st.subheader("Steps followed")
    st.write("1. Loaded and explored the dataset")
    st.write("2. Cleaned and preprocessed the data")
    st.write("3. Did EDA with charts and visualizations")
    st.write("4. Applied VADER and TextBlob for sentiment scoring")
    st.write("5. Trained Logistic Regression and Naive Bayes models")
    st.write("6. Compared model performance")
    st.write("7. Built this Streamlit app to demonstrate the results")
    st.write("")
    st.write("Model Accuracy:")
    st.write("- Logistic Regression: 91%")
    st.write("- Naive Bayes: 88%")
    st.write("- VADER: 74%")
    st.write("- TextBlob: 69%")