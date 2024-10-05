import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import streamlit as st
import io
import requests
from sklearn.ensemble import RandomForestRegressor

# Load the dataset from the GitHub repository
file_url = 'https://github.com/PatricRc/ArroyoDNA/raw/main/Human%20Skills%20Resultados%20%201.xlsx'
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

# Select only the relevant columns
columns_to_keep = [
    'ID', 'Rol ', 'Genero', 'Edad', 'País', 'Meses en Arroyo', 'Años de experiencia', 'Nivel de inglés',
    'Autogestión', 'Compromiso con la excelencia', 'Trabajo en equipo', 'Comunicación efectiva',
    'Pensamiento análitico', 'Adaptabilidad', 'Responsabilidad', 'Atención al detalle',
    'Liderazgo', 'Gestión de problemas', 'Orientación a resultados', 'Pensamiento estratégico',
    'Apertura', 'Iniciativa', 'Orientación al cliente', 'Autoaprendizaje',
    'Tolerancia a la presión', 'Negociación', 'Discreción', 'Integridad'
]
df = df[columns_to_keep]

# Streamlit app setup
st.set_page_config(page_title='Employee Survey EDA', page_icon='📊', layout='wide')
st.title('📊 Employee Survey EDA')

# Filters for DataFrame
st.sidebar.header('Filter the Data')

# Get top 20 roles by number of unique IDs in ascending order
top_20_roles = df.groupby('Rol ')['ID'].nunique().sort_values(ascending=True).head(20).index.tolist()

# Role filter for top 20 roles
role_filter = st.sidebar.multiselect('Select Role', options=top_20_roles, default=top_20_roles)

# Country filter
country_filter = st.sidebar.multiselect('Select Country', options=df['País'].unique(), default=df['País'].unique())

# Age filter
age_filter = st.sidebar.slider('Select Age Range', int(df['Edad'].min()), int(df['Edad'].max()), (int(df['Edad'].min()), int(df['Edad'].max())))

# Numeric slicers
months_in_company_filter = st.sidebar.slider('Select Meses en Arroyo Range', int(df['Meses en Arroyo'].min()), int(df['Meses en Arroyo'].max()), (int(df['Meses en Arroyo'].min()), int(df['Meses en Arroyo'].max())))
experience_filter = st.sidebar.slider('Select Años de experiencia Range', int(df['Años de experiencia'].min()), int(df['Años de experiencia'].max()), (int(df['Años de experiencia'].min()), int(df['Años de experiencia'].max())))

# Filter DataFrame
filtered_df = df[
    (df['Rol '].isin(role_filter)) &
    (df['País'].isin(country_filter)) &
    (df['Edad'] >= age_filter[0]) & (df['Edad'] <= age_filter[1]) &
    (df['Meses en Arroyo'] >= months_in_company_filter[0]) & (df['Meses en Arroyo'] <= months_in_company_filter[1]) &
    (df['Años de experiencia'] >= experience_filter[0]) & (df['Años de experiencia'] <= experience_filter[1])
]

# Display filtered DataFrame
st.subheader('Filtered Dataset')
st.write(filtered_df)

# Summary statistics
st.subheader('Summary Statistics')
st.write(filtered_df.describe())

# Correlation Heatmap for Numerical Features
st.subheader('Correlation Heatmap for Numerical Features')
plt.figure(figsize=(int(16 * 0.6), int(12 * 0.6)))
sns.heatmap(filtered_df.select_dtypes(include=[np.number]).corr(), annot=False, cmap='viridis')
st.pyplot(plt)

# Distribution of Age
st.subheader('Distribution of Age')
plt.figure(figsize=(int(10 * 0.6), int(6 * 0.6)))
sns.histplot(filtered_df['Edad'], kde=True, color='blue')
st.pyplot(plt)

# Gender Countplot
st.subheader('Gender Distribution')
plt.figure(figsize=(int(8 * 0.6), int(5 * 0.6)))
sns.countplot(data=filtered_df, x='Genero', palette='Set2')
st.pyplot(plt)

# Experience vs. Nivel de inglés
st.subheader('Years of Experience vs. Nivel de inglés')
plt.figure(figsize=(int(10 * 0.6), int(6 * 0.6)))
sns.scatterplot(data=filtered_df, x='Años de experiencia', y='Nivel de inglés', hue='Genero', palette='Set1')
st.pyplot(plt)

# Distribution of Nivel de inglés
st.subheader('Distribution of Nivel de inglés')
plt.figure(figsize=(int(10 * 0.6), int(6 * 0.6)))
sns.histplot(filtered_df['Nivel de inglés'], kde=True, color='green')
st.pyplot(plt)

# Relationship between Age and Nivel de inglés
st.subheader('Age vs. Nivel de inglés')
filtered_df['Edad'] = pd.to_numeric(filtered_df['Edad'], errors='coerce')
filtered_df['Nivel de inglés'] = pd.to_numeric(filtered_df['Nivel de inglés'], errors='coerce')
plt.figure(figsize=(int(10 * 0.6), int(6 * 0.6)))
sns.regplot(data=filtered_df, x='Edad', y='Nivel de inglés', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
st.pyplot(plt)

# Pairplot to observe relationships between select numerical features
st.subheader('Pairplot of Selected Numerical Features')
selected_features = ['Edad', 'Meses en Arroyo', 'Años de experiencia', 'Nivel de inglés']
sns.pairplot(filtered_df[selected_features], height=2.4)
st.pyplot(plt)

# Summary of categorical features
st.subheader('Summary of Categorical Features')
st.write(filtered_df.describe(include=['object']))

# Countplot of Country
st.subheader('Country Distribution')
plt.figure(figsize=(int(12 * 0.6), int(6 * 0.6)))
sns.countplot(data=filtered_df, x='País', palette='viridis')
plt.xticks(rotation=45)
st.pyplot(plt)

# Feature Importance Section
st.subheader('Feature Importance for Predicting Employee Adaptability')

# Prepare data for feature importance calculation
excluded_columns = ['ID', 'Rol ', 'Genero', 'Edad', 'País', 'Meses en Arroyo', 'Años de experiencia', 'Nivel de inglés']
X = filtered_df.drop(columns=excluded_columns + ['Adaptabilidad'])
y = filtered_df['Adaptabilidad']

# One-hot encoding for categorical variables
X = pd.get_dummies(X, drop_first=True)

# Train a Random Forest model to determine feature importance
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Get feature importances from the model
feature_importances = model.feature_importances_

# Create a DataFrame for feature importance
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Plot the top 15 most important features
plt.figure(figsize=(int(10 * 0.6), int(6 * 0.6)))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), palette='magma')
plt.title('Top 15 Important Features for Predicting Employee Adaptability')
plt.xlabel('Importance')
plt.ylabel('Feature')
st.pyplot(plt)
