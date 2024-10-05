import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import streamlit as st
import io
import requests

# Load the dataset from the GitHub repository
file_url = 'https://raw.githubusercontent.com/PatricRc/ArroyoDNA/main/Human%20Skills%20Resultados%201.xlsx'
try:
    response = requests.get(file_url)
    response.raise_for_status()  # Raise an error for bad status codes
    file_data = io.BytesIO(response.content)
    df = pd.read_excel(file_data, engine='openpyxl', sheet_name='Sheet1')
except requests.exceptions.RequestException as e:
    st.error(f"Error loading the dataset: {e}")
    st.stop()
except ValueError as e:
    st.error(f"Error reading the Excel file: {e}")
    st.stop()

# Streamlit app setup
st.set_page_config(page_title='Employee Survey EDA', page_icon='📊', layout='wide')
st.title('📊 Employee Survey EDA')

# Display basic information about the dataset
st.subheader('Dataset Information')
buf = []
df.info(buf=buf)
st.text("\n".join(buf))

# Summary statistics
st.subheader('Summary Statistics')
st.write(df.describe())

# Correlation Heatmap for Numerical Features
st.subheader('Correlation Heatmap for Numerical Features')
plt.figure(figsize=(16, 12))
sns.heatmap(df.corr(), annot=False, cmap='viridis')
st.pyplot(plt)

# Distribution of Age
st.subheader('Distribution of Age')
plt.figure(figsize=(10, 6))
sns.histplot(df['Edad'], kde=True, color='blue')
st.pyplot(plt)

# Gender Countplot
st.subheader('Gender Distribution')
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Genero', palette='Set2')
st.pyplot(plt)

# Experience vs. Total Score
st.subheader('Years of Experience vs. Total Score')
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Años de experiencia', y='TOTAL', hue='Genero', palette='Set1')
st.pyplot(plt)

# Boxplot of Total Scores by Role
st.subheader('Boxplot of Total Scores by Role')
plt.figure(figsize=(18, 8))
sns.boxplot(data=df, x='Rol ', y='TOTAL', palette='Set3')
plt.xticks(rotation=90)
st.pyplot(plt)

# Total Score Distribution
st.subheader('Distribution of Total Scores')
plt.figure(figsize=(10, 6))
sns.histplot(df['TOTAL'], kde=True, color='green')
st.pyplot(plt)

# Relationship between Age and Total Score
st.subheader('Age vs. Total Score')
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='Edad', y='TOTAL', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
st.pyplot(plt)

# Pairplot to observe relationships between select numerical features
st.subheader('Pairplot of Selected Numerical Features')
selected_features = ['Edad', 'Meses en Arroyo', 'Años de experiencia', 'TOTAL']
sns.pairplot(df[selected_features], hue='TOTAL', palette='coolwarm')
st.pyplot(plt)

# Summary of categorical features
st.subheader('Summary of Categorical Features')
st.write(df.describe(include=['object']))

# Countplot of Country
st.subheader('Country Distribution')
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='País', palette='viridis')
plt.xticks(rotation=45)
st.pyplot(plt)
