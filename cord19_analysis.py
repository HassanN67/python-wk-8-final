import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import streamlit as st
from datetime import datetime
import re

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load the data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('metadata.csv', low_memory=False)
        return df
    except FileNotFoundError:
        st.error("File 'metadata.csv' not found. Please make sure it's in the same directory.")
        return None

# Basic data exploration
def explore_data(df):
    st.subheader("Dataset Overview")
    st.write(f"Dataset shape: {df.shape}")
    
    st.write("First few rows:")
    st.dataframe(df.head())
    
    st.write("Data types:")
    st.write(df.dtypes)
    
    st.write("Missing values:")
    missing_data = df.isnull().sum().sort_values(ascending=False)
    missing_df = pd.DataFrame({'Column': missing_data.index, 'Missing Values': missing_data.values})
    st.dataframe(missing_df.head(10))  # Show top 10 columns with most missing values
    
    st.write("Basic statistics for numerical columns:")
    st.write(df.describe())

# Data cleaning and preparation
def clean_data(df):
    st.subheader("Data Cleaning")
    
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Convert publish_time to datetime
    df_clean['publish_time'] = pd.to_datetime(df_clean['publish_time'], errors='coerce')
    
    # Extract year from publication date
    df_clean['year'] = df_clean['publish_time'].dt.year
    
    # Create abstract word count
    df_clean['abstract_word_count'] = df_clean['abstract'].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 0)
    
    # Fill missing values in important columns
    df_clean['journal'] = df_clean['journal'].fillna('Unknown Journal')
    
    st.write("Data after cleaning:")
    st.write(f"New shape: {df_clean.shape}")
    st.write("Sample of cleaned data:")
    st.dataframe(df_clean[['title', 'journal', 'year', 'abstract_word_count']].head())
    
    return df_clean

# Analysis and visualization functions
def plot_publications_over_time(df):
    st.subheader("Publications Over Time")
    
    # Count publications by year
    yearly_counts = df['year'].value_counts().sort_index()
    
    fig, ax = plt.subplots()
    ax.plot(yearly_counts.index, yearly_counts.values, marker='o', linewidth=2)
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Publications')
    ax.set_title('COVID-19 Publications Over Time')
    ax.grid(True)
    
    st.pyplot(fig)

def plot_top_journals(df):
    st.subheader("Top Journals Publishing COVID-19 Research")
    
    # Get top 10 journals
    top_journals = df['journal'].value_counts().head(10)
    
    fig, ax = plt.subplots()
    ax.barh(top_journals.index, top_journals.values)
    ax.set_xlabel('Number of Publications')
    ax.set_title('Top 10 Journals by COVID-19 Publications')
    
    st.pyplot(fig)

def create_wordcloud(df):
    st.subheader("Most Frequent Words in Titles")
    
    # Combine all titles
    text = ' '.join(df['title'].dropna().values)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Word Cloud of Paper Titles')
    
    st.pyplot(fig)

def plot_source_distribution(df):
    st.subheader("Distribution of Papers by Source")
    
    # Get top sources
    source_counts = df['source_x'].value_counts().head(10)
    
    fig, ax = plt.subplots()
    ax.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%')
    ax.set_title('Top 10 Sources of Papers')
    
    st.pyplot(fig)

# Main function to run the Streamlit app
def main():
    st.title("CORD-19 Data Explorer")
    st.write("Simple exploration of COVID-19 research papers")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Go to", 
                               ["Dataset Overview", 
                                "Data Cleaning", 
                                "Publications Over Time",
                                "Top Journals",
                                "Title Word Cloud",
                                "Source Distribution"])
    
    # Display selected section
    if options == "Dataset Overview":
        explore_data(df)
    elif options == "Data Cleaning":
        df_clean = clean_data(df)
    else:
        # Clean data for analysis
        df_clean = clean_data(df)
        
        if options == "Publications Over Time":
            plot_publications_over_time(df_clean)
        elif options == "Top Journals":
            plot_top_journals(df_clean)
        elif options == "Title Word Cloud":
            create_wordcloud(df_clean)
        elif options == "Source Distribution":
            plot_source_distribution(df_clean)
    
    # Add some summary statistics
    st.sidebar.subheader("Dataset Summary")
    st.sidebar.write(f"Total papers: {df.shape[0]}")
    st.sidebar.write(f"Total columns: {df.shape[1]}")
    
    if 'df_clean' in locals():
        st.sidebar.write(f"Earliest publication year: {df_clean['year'].min()}")
        st.sidebar.write(f"Latest publication year: {df_clean['year'].max()}")

if __name__ == "_main_":
    main()