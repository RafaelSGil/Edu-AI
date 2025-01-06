import streamlit as st
import pandas as pd

st.set_page_config(page_title="Edu_AI Dashboard",page_icon="🏠",  menu_items={
        'About': "# Work developed by:\n - Rafael Gil\n- David Raposo\n\n ### Made in the context of Introduction to Data Science course"
    })
st.title("The Impact of Generative Artificial Intelligence on Higher Education")

def load_data():
    return pd.read_csv('data/scopus.csv')

# Streamlit allows to store values in Session State allowing for dynamic usage of data
# Without having to load it everytime we switch page, making the app much faster
# Check if data is already in session state
if "csv_data" not in st.session_state:
    # Load data to Session State, if not there already
    st.session_state.csv_data = load_data()

st.markdown("""
### Table of Contents
1. [Introduction](#introduction)
2. [Part 1: Bibliometric Analysis](#part-1-bibliometric-analysis)
3. [Part 2: Text Mining Analysis](#part-2-text-mining-analysis)
4. [Methodology](#methodology)
5. [Expected Outcomes](#expected-outcomes)  
6. [How To Use](#how-to)

### Introduction
<a name="introduction"></a>
Generative Artificial intelligence (GenAI) is transforming higher education, influencing teaching, learning, and administrative practices. As GenAI becomes more prevalent in educational settings, there is a growing body of research examining its implications, benefits, and challenges in higher education. This notebook aims to explore this research landscape by conducting a two-part analysis.

### Part 1: Bibliometric Analysis
<a name="part-1-bibliometric-analysis"></a>
- **Quantifying Publication Trends ->** Determining how research on GenAI in higher education has grown over recent years.
- **Identifying Key Contributors ->** Recognizing influential authors, institutions, and countries that are leading research efforts.

### Part 2: Text Mining Analysis
<a name="part-2-text-mining-analysis"></a>
Following the bibliometric analysis, we will apply text mining techniques to the articles themselves. This phase will enable us to delve deeper into the content, uncovering nuanced insights into how AI is being discussed and understood in higher education contexts. In particular, we aim to:
- **Extract Key Topics ->** Use natural language processing (NLP) methods to identify key themes and subtopics.
- **Analyze Sentiment and Context ->** Examine how GenAI’s impact is portrayed in higher education, focusing on sentiments around its benefits and challenges.

### Methodology
<a name="methodology"></a>
We will use SCOPUS API to aquire articles based on the following search query:
``` "impact" AND "high* education" AND ( "generative artificial intelligence" OR "GenAI" ) ```. Our approach will follow these steps:
1. **Data collection ->** Retrieve bibliometric data from SCOPUS, including article titles, abstracts, authors, affiliations, and publication years.
2. **Data Processing ->** Organize and clean the data for analysis, ensuring it is suitable for quantitative and qualitative assessments.
3. **Analysis:**
    * **Publication Trends ->** Analyze the number of publications over time to identify growth patterns.
    * **Key Contributors ->** Identify leading authors, institutions, and countries in AI research within higher education.
    * **Research Themes ->** Use text mining techniques to uncover major themes and topics in the literature.

### Expected outcomes
<a name="expected-outcomes"></a>
This analysis will contribute to understanding the broader impact of GenAI on higher education, offering valuable perspectives for academics, practitioners, and policymakers aiming to leverage AI technologies to enhance educational experiences and outcomes.

            
### How to use this dashboard
<a name="how-to"></a>
Navigate this dashboard using the tabs on the lateral menu. Each page has a set of controls and parameters on the sidebar that allows the user to see different visualizations.
""", unsafe_allow_html=True)