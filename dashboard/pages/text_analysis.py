import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import string
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pyLDAvis.gensim_models as gensimvis

st.set_page_config(page_title="Text Analysis",page_icon=":pencil:",  menu_items={
        'About': "# Work developed by:\n - Rafael Gil\n- David Raposo\n\n ### Made in the context of Introduction to Data Science course"
    })
st.title("Text analysis")

st.sidebar.title("Controls & Parameters")

st.markdown("---")

############################## DATA LOADING ##############################

def load_data():
    return pd.read_csv('data/scopus.csv')

DATA = None

# Streamlit allows storing values in session state for dynamic usage of data
if "csv_data" not in st.session_state:
    st.session_state.csv_data = load_data()  # Load data into session state
else:
    DATA = st.session_state.csv_data  # Use the preloaded data


############################## AUXILIARY FUNCTIONS ##############################
def advanced_text_preprocess(
    text, 
    lowercase=True, 
    remove_punctuation=True, 
    remove_numbers=True, 
    remove_stopwords=True, 
    lemmatize=True
):
    # Ensure text is a string and not NaN
    if not isinstance(text, str):
        return ''
    
    # Lowercase conversion
    if lowercase:
        text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove punctuation
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Return processed tokens as a string
    return ' '.join(tokens)

def apply_preprocessing_abstract():
    DATA['Processed_Abstract'] = DATA['Abstract'].apply(advanced_text_preprocess)
    DATA['Processed_Abstract'] = DATA['Processed_Abstract'].fillna("")
    DATA['Processed_Abstract'] = DATA['Processed_Abstract'].astype(str)

    DATA['Custom_Processed'] = DATA['Abstract'].apply(
        lambda x: advanced_text_preprocess(
            x, 
            lowercase=True, 
            remove_numbers=False, 
            lemmatize=True
        )
    )

def enhanced_advanced_features(text):
    words = text.split()
    if not words:
        return {
            'word_count': 0,
            'unique_word_count': 0,
            'avg_word_length': 0,
            'text_complexity': 0,
            'word_length_std': 0,
            'vocabulary_richness': 0
        }
    
    return {
        'word_count': len(words),
        'unique_word_count': len(set(words)),
        'avg_word_length': np.mean([len(word) for word in words]),
        'text_complexity': len(text) / len(words),
        'word_length_std': np.std([len(word) for word in words]),  # Word length variation
        'vocabulary_richness': len(set(words)) / len(words)  # Lexical diversity ratio
    }

############################## PLOT FUNCTIONS ##############################

def display_advanced_features():
    st.title("Advanced Features")

    advanced_features = DATA['Processed_Abstract'].apply(enhanced_advanced_features).apply(pd.Series)

    plot_type = st.sidebar.selectbox(
        "Select Plot", 
        ["Word Count Distribution", "Unique vs Total Words", "Average Word Length", 
         "Text Complexity", "Vocabulary Richness", "Word Length Variation"]
    )

    if plot_type == "Word Count Distribution":
        fig = px.histogram(advanced_features, x='word_count', nbins=30, title="Word Count Distribution")
        st.plotly_chart(fig)

    elif plot_type == "Unique vs Total Words":
        fig = px.scatter(
            advanced_features, x='word_count', y='unique_word_count', 
            title="Unique vs Total Words", labels={"x": "Total Words", "y": "Unique Words"}
        )
        st.plotly_chart(fig)

    elif plot_type == "Average Word Length":
        fig = px.histogram(advanced_features, x='avg_word_length', nbins=30, title="Average Word Length Distribution")
        st.plotly_chart(fig)

    elif plot_type == "Text Complexity":
        fig = px.histogram(advanced_features, x='text_complexity', nbins=30, title="Text Complexity Distribution")
        st.plotly_chart(fig)

    elif plot_type == "Vocabulary Richness":
        fig = px.histogram(advanced_features, x='vocabulary_richness', nbins=30, title="Vocabulary Richness")
        st.plotly_chart(fig)

    elif plot_type == "Word Length Variation":
        fig = px.histogram(advanced_features, x='word_length_std', nbins=30, title="Word Length Variation")
        st.plotly_chart(fig)

def display_clustering():
    st.title("Clustering Analysis")

    num_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)

    tfidf = TfidfVectorizer(max_features=1000)
    features = tfidf.fit_transform(DATA['Processed_Abstract'])  # Replace with actual column

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(features)

    DATA['Cluster'] = clusters

    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(features.toarray())
    reduced_df = pd.DataFrame(reduced_data, columns=['PCA1', 'PCA2'])
    reduced_df['Cluster'] = clusters

    fig = px.scatter(
        reduced_df, x='PCA1', y='PCA2', color='Cluster', 
        title="Clusters (PCA Reduced)", color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig)

def display_topic_modeling():
    st.title("Topic Modeling")

    num_topics = st.sidebar.slider("Number of Topics", 2, 20, 5)
    passes = st.sidebar.slider("Number of Passes", 1, 20, 10)

    st.write("Generating LDA Model... This may take a couple minutes")
    DATA['tokenized_Abstract'] = DATA['Processed_Abstract'].apply(lambda x: x.split())
    dictionary = Dictionary(DATA['tokenized_Abstract'])
    corpus = [dictionary.doc2bow(text) for text in DATA['tokenized_Abstract']]

    lda_model = LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary, passes=passes, workers=4)

    topics = lda_model.print_topics(num_words=4)
    st.write("### Topics")
    for topic in topics:
        st.write(f"**Topic {topic[0]}**: {topic[1]}")

    coherence_model = CoherenceModel(model=lda_model, texts=DATA['tokenized_Abstract'], dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    st.write(f"### Coherence Score: {coherence_score:.4f}")

    for t in range(lda_model.num_topics):
        wordcloud = WordCloud().fit_words(dict(lda_model.show_topic(t, 200)))
        plt.figure()
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.title(f"Topic #{t}")
        st.pyplot(plt)

    st.write("### Interactive Topic Visualization")
    vis = gensimvis.prepare(lda_model, corpus, dictionary, R=10)
    st.write(vis)

def display_preprocessing_comparison():
    apply_preprocessing_abstract()
    selected_columns = DATA[['Abstract', 'Processed_Abstract']]
    num_rows = st.sidebar.slider(
        label="Select number of rows to display",
        min_value=1,
        max_value=10,
        value=5,  # Default value
        step=1
    )

    st.dataframe(selected_columns.head(num_rows))

############################## SIDEBAR ##############################
feature_selection = st.sidebar.selectbox("Select Feature to Analyze", ["Preprocessing Comparison", "Advanced Features", "Topic Modeling","Clustering"])

if feature_selection == "Preprocessing Comparison":
    display_preprocessing_comparison()
elif feature_selection == "Advanced Features":
    display_advanced_features()
elif feature_selection == "Clustering":
    display_clustering()
elif feature_selection == "Topic Modeling":
    display_topic_modeling()
