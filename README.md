
# Flipkart Product Review Sentiment Analysis

A complete end-to-end NLP project that analyzes customer sentiments from Flipkart product reviews using rule-based and machine learning approaches.

## Dataset

- Source: Flipkart Product Reviews  
- Total Reviews: 363,261  
- Unique Products: 1,175  
- Columns: Product Name, Price, Rating, Review, Summary  


## Project Structure

flipkart-sentiment-analysis/
├── app.py
├── sentiment_analysis.ipynb
├── README.md
```



## What This Project Does

- Loads and cleans the raw Flipkart review dataset
- Handles missing values, duplicates, and encoding issues
- Performs exploratory data analysis with charts and visualizations
- Labels sentiment from star ratings (Positive, Neutral, Negative)
- Applies VADER and TextBlob for rule-based sentiment scoring
- Trains Logistic Regression and Naive Bayes ML classifiers
- Evaluates and compares all model performances
- Deploys an interactive web app using Streamlit


## Model Performance

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 91% |
| Naive Bayes | 88% |
| VADER | 74% |
| TextBlob | 69% |



## Technologies Used

- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn
- VADER, TextBlob
- Scikit-learn
- WordCloud
- Streamlit

2. Install dependencies
```
pip install pandas numpy matplotlib seaborn vaderSentiment textblob scikit-learn wordcloud streamlit
```


## Key Findings

- Over 68% of Flipkart reviews are 5 star ratings
- Logistic Regression with TF-IDF gives the best accuracy of 91%
- Negative reviews are longer on average than positive reviews
- Most common positive words: good, quality, product, best, value
- Most common negative words: worst, return, damage, waste, broken

