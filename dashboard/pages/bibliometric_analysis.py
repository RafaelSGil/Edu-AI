import streamlit as st
import pandas as pd
import plotly.express as px
import re

st.set_page_config(
    page_title="Bibliometric Analysis",
    page_icon=":books:",
    menu_items={
        'About': "# Work developed by:\n - Rafael Gil\n- David Raposo\n\n ### Made in the context of Introduction to Data Science course"
    },
    layout="wide"
)

st.title("Bibliometric Analysis")
st.write("Each dropbox, on the sidebar, provides visualizations that answer each of the questions: What?, Where?, Who?,")

st.markdown("---")

st.sidebar.title("Controls & Parameters")

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

def extract_countries(df):
        countries_list = []

        for _, row in df.iterrows():
            affiliation = str(row['Affiliations']).strip()
            if affiliation and affiliation.lower() != 'nan':
                parts = affiliation.split(',', 1)
                if len(parts) > 1:
                    country = parts[1].strip()
                    countries_list.append(country)

        return countries_list

def clean_author_name(author_string):
    # Regex to clean anything inside parentheses, parentheses included
    cleaned_name = re.sub(r'\(.*?\)', '', author_string).strip()
    return cleaned_name
############################## PLOT FUNCTIONS ##############################

def plot_publication_types(df):
    publication_type_counts = df['Document Type'].value_counts().reset_index()
    publication_type_counts.columns = ['Publication Type', 'Count']

    fig = px.bar(
        publication_type_counts,
        x="Count",
        y="Publication Type",
        orientation="h",
        color="Publication Type",
        title="Distribution of Publication Types",
        labels={"Publication Type": "Type of Publication", "Count": "Number of Publications"},
        color_discrete_sequence=px.colors.sequential.Viridis,
    )

    st.plotly_chart(fig, use_container_width=True)

def plot_top_keywords(df):
    num_keywords = st.slider("Number of Keywords to Display", min_value=5, max_value=15, value=10)
    all_keywords = [kw.strip() for keywords in df['Author Keywords'].dropna() for kw in keywords.split(';')]
    top_keywords = pd.Series(all_keywords).value_counts().head(num_keywords).reset_index()
    top_keywords.columns = ['Keyword', 'Frequency']
    top_keywords = top_keywords.sort_values(by="Frequency", ascending=False)

    fig = px.bar(
        top_keywords,
        x="Frequency",
        y="Keyword",
        orientation="h",
        color="Frequency",
        title=f"Top {num_keywords} Keywords in Publications",
        labels={"Keyword": "Keyword", "Frequency": "Frequency"},
        color_continuous_scale="darkmint",
    )

    fig.update_yaxes(categoryorder="total ascending")

    st.plotly_chart(fig, use_container_width=True)

def plot_open_access(df):
    open_access_counts = df['Open Access'].value_counts()
    open_access_df = open_access_counts.reset_index()
    open_access_df.columns = ['Open Access', 'Count']

    fig = px.bar(
        open_access_df,
        x="Open Access",
        y="Count",
        color="Open Access",
        title="Open Access Counts",
        labels={"Open Access": "Access Type", "Count": "Number of Publications"},
        color_discrete_sequence=px.colors.sequential.Viridis
    )

    st.plotly_chart(fig, use_container_width=True)

def plot_relevant_sources(df):
    num = st.slider("Number of Journals and Conferences to Display", min_value=5, max_value=15, value=10)
    top_sources = df['Source title'].value_counts().head(num)

    fig = px.bar(
        x=top_sources.values, 
        y=top_sources.index, 
        labels={"x": "Number of Publications", "y": "Source"},
        title=f"Top {num} Journals and Conferences for AI in Higher Education",
        color=top_sources.values, 
        color_continuous_scale="magma",
    )

    fig.update_yaxes(categoryorder="total ascending")

    st.plotly_chart(fig, use_container_width=True)

def plot_top_institutions(df):
    all_institutions = []
    
    # Extract and clean affiliations
    for _, row in df.iterrows():
        affiliation = str(row['Affiliations']).strip()
        if affiliation and affiliation.lower() != 'nan':  
            all_institutions.append(affiliation)

    if not all_institutions:
        st.write("No valid affiliations found.")
        return

    num = st.slider("Number of Institutions to Display", min_value=5, max_value=15, value=10)
    institution_counts = pd.Series(all_institutions).value_counts().head(num)

    fig = px.bar(
        x=institution_counts.values,
        y=institution_counts.index,
        orientation='h',
        color=institution_counts.values,
        color_continuous_scale='plasma',
        labels={'x': 'Number of Publications', 'y': 'Institution'},
        title=f"Top {num} Institutions by Publication Count"
    )

    fig.update_yaxes(categoryorder="total ascending")

    st.plotly_chart(fig, use_container_width=True)

def plot_top_countries(df):    
    countries = extract_countries(df)
    
    if countries:
        num = st.slider("Number of Countries to Display", min_value=5, max_value=15, value=10)
        countries_counts = pd.Series(countries).value_counts().head(num)
        
        fig = px.bar(
            x=countries_counts.values,
            y=countries_counts.index,
            orientation='h',
            color=countries_counts.values,
            color_continuous_scale='plasma',
            labels={'x': 'Number of Publications', 'y': 'Country'},
            title=f"Top {num} Countries by Publication Count"
        )
        fig.update_yaxes(categoryorder="total ascending")
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No country data found to plot")

def plot_most_prominent_authors(df):
    author_citations_dict = {}
    
    for _, row in df.iterrows():
        if isinstance(row['Author full names'], str) and row['Author full names'].strip() != '':
            authors = [clean_author_name(author.strip()) for author in row['Author full names'].split(';')]
            citation_count = row['Cited by']
            
            for author in authors:
                if author in author_citations_dict:
                    author_citations_dict[author] += citation_count
                else:
                    author_citations_dict[author] = citation_count

    num = st.slider("Number of Authors to Display", min_value=5, max_value=15, value=10)
    author_citations_series = pd.Series(author_citations_dict).sort_values(ascending=False).head(num)
    
    fig = px.bar(
            x=author_citations_series.values,
            y=author_citations_series.index,
            orientation='h',
            color=author_citations_series.values,
            color_continuous_scale='plasma',
            labels={'x': 'Total Citations', 'y': 'Author'},
            title=f"Top {num} Authors by Total Citations"
        )
    fig.update_yaxes(categoryorder="total ascending")
    
    st.plotly_chart(fig, use_container_width=True)

def plot_authors_with_most_publications(df):
    authors_list = df['Author full names'].dropna().apply(lambda x: [clean_author_name(author.strip()) for author in x.split(';')])
    all_authors = [author for sublist in authors_list for author in sublist]

    num = st.slider("Number of Authors to Display", min_value=5, max_value=15, value=10)
    author_counts = pd.Series(all_authors).value_counts().head(num)
    
    fig = px.bar(
        x=author_counts.values,
        y=author_counts.index,
        orientation='h',
        color=author_counts.values,
        color_continuous_scale='darkmint',
        labels={'x': 'Number of Publications', 'y': 'Author'},
        title=f"Top {num} Authors with the Most Publications"
    )
    fig.update_yaxes(categoryorder="total ascending")
    
    st.plotly_chart(fig, use_container_width=True)

def plot_top_sponsors(df):
    all_sponsors = [sponsor.strip() for sponsors in df['Sponsors'].dropna() for sponsor in sponsors.split(';')]
    
    num = st.slider("Number of Authors to Display", min_value=5, max_value=15, value=10)
    top_sponsors = pd.Series(all_sponsors).value_counts().head(num)
    
    fig = px.bar(
        x=top_sponsors.values,
        y=top_sponsors.index,
        orientation='h',
        color=top_sponsors.values,
        color_continuous_scale='cividis',
        labels={'x': 'Number of Publications', 'y': 'Sponsor'},
        title=f"Top {num} Sponsors by Publication Count"
    )
    fig.update_yaxes(categoryorder="total ascending")
    
    st.plotly_chart(fig, use_container_width=True)

def plot_publication_distribution(df):
    min_year = df['Year'].min()
    max_year = df['Year'].max()

    start_year, end_year = st.slider(
        "Select Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1
    )
    
    df_filtered = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]
    
    publications_per_year = df_filtered['Year'].value_counts().sort_index()
    
    fig = px.line(
            publications_per_year,
            x=publications_per_year.index,
            y=publications_per_year.values,
            markers=True,
            title=f"Publications Over Time ({start_year} - {end_year})",
            labels={'x': 'Year', 'y': 'Number of Publications'}
        )

    st.plotly_chart(fig, use_container_width=True)

def plot_open_access_distribution(data):
    unique_access_values = data['Open Access'].unique()
    st.write("Unique 'Open Access' Values:", unique_access_values)

    data['Is Open Access'] = data['Open Access'].str.contains("All Open Access", na=False)

    min_year = data['Year'].min()
    max_year = data['Year'].max()

    start_year, end_year = st.slider(
        "Select Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1
    )
    
    data_filtered = data[(data['Year'] >= start_year) & (data['Year'] <= end_year)]
    
    document_counts = data_filtered.groupby(['Year', 'Is Open Access']).size().unstack(fill_value=0)

    fig = px.line(
            document_counts,
            x=document_counts.index,
            y=document_counts.columns,
            title=f"Open Access vs Non-Open Access Documents ({start_year} - {end_year})",
            labels={'x': 'Year', 'y': 'Number of Documents'},
            markers=True,
        )
    
    colors = ['red', 'green']
    for i, trace in enumerate(fig.data):
        trace.line.color = colors[i]

    st.plotly_chart(fig, use_container_width=True)

def plot_citations_per_year(data):
    data['Is Open Access'] = data['Open Access'].str.contains("All Open Access", na=False)

    min_year = data['Year'].min()
    max_year = data['Year'].max()

    start_year, end_year = st.slider(
        "Select Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1
    )
    
    data_filtered = data[(data['Year'] >= start_year) & (data['Year'] <= end_year)]

    grouped_data = data_filtered.groupby(['Year', 'Is Open Access'])['Cited by'].sum().unstack(fill_value=0)

    fig = px.line(
            grouped_data,
            x=grouped_data.index,
            y=grouped_data.columns,
            title=f'Citations per Year: Open Access vs Non-Open Access ({start_year} - {end_year})',
            labels={'x': 'Year', 'y': 'Total Citations'},
            markers=True
        )
    colors = ['red', 'green']
    for i, trace in enumerate(fig.data):
        trace.line.color = colors[i]

    st.plotly_chart(fig, use_container_width=True)

############################## Advanced Filtering and Analysis ##############################

def advanced_filtering_and_analysis(df):
    st.header("Advanced Filtering and Analysis")

    year_range = st.slider(
        "Select Year Range",
        min_value=int(df['Year'].min()),
        max_value=int(df['Year'].max()),
        value=(int(df['Year'].min()), int(df['Year'].max()))
    )

    filtered_df = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]

    doc_types = st.multiselect(
        "Select Document Types",
        options=df['Document Type'].unique().tolist(),
        default=df['Document Type'].unique().tolist()
    )
    filtered_df = filtered_df[filtered_df['Document Type'].isin(doc_types)]

    max_docs = st.slider(
        "Select Number of Documents to Display",
        min_value=1,
        max_value=len(DATA),
        value=10
    )

    st.write(f"Filtered Publications: {len(filtered_df)}")
    st.write(f"Displaying Top {max_docs} Documents")

    if st.button("Show Filtered Data"):
        st.dataframe(filtered_df.head(max_docs))


############################## SIDEBAR ##############################

plot_choice_what = st.sidebar.selectbox(
    "**What?**",
    options=["Select a plot", "Publication Types", "Top Keywords", "Open Access vs Closed Access Counts"]
)

if plot_choice_what == "Select a plot":
    st.header("What? section")

elif plot_choice_what == "Publication Types":
    st.header("Distribution of Publication Types")
    plot_publication_types(DATA)

elif plot_choice_what == "Top Keywords":
    st.header("Top Keywords in Publications")
    plot_top_keywords(DATA)

elif plot_choice_what == "Open Access vs Closed Access Counts":
    st.header("Open Access vs Closed Access Counts")
    plot_open_access(DATA)

st.markdown("---")

plot_choice_where = st.sidebar.selectbox(
    "**Where?**",
    options=["Select a plot", "Top journals and conferences", "Most prominent institutions", 
             "Most prominent countries"]
)

if plot_choice_where == "Select a plot":
    st.header("Where? section")

elif plot_choice_where == "Top journals and conferences":
    st.header("Top journals and conferences")
    plot_relevant_sources(DATA)

elif plot_choice_where == "Most prominent institutions":
    st.header("Most prominent institutions")
    plot_top_institutions(DATA)

elif plot_choice_where == "Most prominent countries":
    st.header("Most prominent countries")
    plot_top_countries(DATA)

st.markdown("---")

plot_choice_who = st.sidebar.selectbox(
    "**Who?**",
    options=["Select a plot", "Most prominent authors", "Authors with the most publications", 
             "Sponsorships and Funding"]
)

if plot_choice_who == "Select a plot":
    st.header("Who? section")
elif plot_choice_who == "Most prominent authors":
    st.header("Most prominent authors")
    plot_most_prominent_authors(DATA)
elif plot_choice_who == "Authors with the most publications":
    st.header("Authors with the most publications")
    plot_authors_with_most_publications(DATA)
elif plot_choice_who == "Sponsorships and Funding":
    st.header("Sponsorships and Funding")
    plot_top_sponsors(DATA)


st.markdown("---")

plot_choice_when = st.sidebar.selectbox(
    "**When?**",
    options=["Select a plot", "Number of publications over time", "Number of Documents per Year", 
             "Citations per Year"]
)

if plot_choice_when == "Select a plot":
    st.header("When? section")

elif plot_choice_when == "Number of publications over time":
    st.header("Number of publications over time")
    plot_publication_distribution(DATA)

elif plot_choice_when == "Number of Documents per Year":
    st.header("Number of Documents per Year")
    plot_open_access_distribution(DATA)

elif plot_choice_when == "Citations per Year":
    st.header("Citations per Year")
    plot_citations_per_year(DATA)

st.markdown("---")

# Always show the advanced filtering section below the plots
advanced_filtering_and_analysis(DATA)