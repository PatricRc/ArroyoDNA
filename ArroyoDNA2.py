import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import requests
from sklearn.ensemble import RandomForestRegressor
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain

# Load the dataset from the GitHub repository
file_url = 'https://raw.githubusercontent.com/PatricRc/ArroyoDNA/main/Human%20Skills%20Resultados%20%201.xlsx'
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

# Ensure columns exist in the dataset before selecting
existing_columns = df.columns.tolist()
columns_to_keep = [
    'ID', 'Rol', 'Genero', 'Edad', 'País', 'Meses en Arroyo', 'Años de experiencia', 'Nivel de inglés',
    'Autogestión', 'Compromiso con la excelencia', 'Trabajo en equipo', 'Comunicación efectiva',
    'Pensamiento ánalitico', 'Adaptabilidad', 'Responsabilidad', 'Atención al detalle',
    'Liderazgo', 'Gestión de problemas', 'Orientación a resultados', 'Pensamiento estratégico',
    'Apertura', 'Iniciativa', 'Orientación al cliente', 'Autoaprendizaje',
    'Tolerancia a la presión', 'Negociación', 'Discreción', 'Integridad'
]
columns_to_keep = [col for col in columns_to_keep if col in existing_columns]
df = df[columns_to_keep]

# Convert 'Rol' column to string
if 'Rol' in df.columns:
    df['Rol'] = df['Rol'].astype(str)

# Streamlit app setup
st.set_page_config(page_title='Employee Survey EDA', page_icon='📈', layout='wide')

# Sidebar for page navigation
st.sidebar.title('Navigation')
page = st.sidebar.radio("Go to", ["Survey EDA", "Machine Learning Prediction", "Chat with Survey Data"])

if page == "Survey EDA":
    st.title('📈 Employee Survey EDA')

    # Filters for DataFrame
    # Filters on the page
    top_20_roles = df.groupby('Rol')['ID'].nunique().sort_values(ascending=True).head(20).index.tolist()
    all_roles = df['Rol'].unique().tolist()

    # Role filter for top 20 roles
    role_filter = st.multiselect('Select Role', options=all_roles, default=top_20_roles)

    # Country filter
    country_filter = st.multiselect('Select Country', options=df['País'].unique(), default=df['País'].unique())

    # Age filter
    age_filter = st.slider('Select Age Range', int(df['Edad'].min()), int(df['Edad'].max()), (int(df['Edad'].min()), int(df['Edad'].max())))

    # Numeric slicers
    months_in_company_filter = st.slider('Select Meses en Arroyo Range', int(df['Meses en Arroyo'].min()), int(df['Meses en Arroyo'].max()), (int(df['Meses en Arroyo'].min()), int(df['Meses en Arroyo'].max())))
    experience_filter = st.slider('Select Años de experiencia Range', int(df['Años de experiencia'].min()), int(df['Años de experiencia'].max()), (int(df['Años de experiencia'].min()), int(df['Años de experiencia'].max())))

    # Filter DataFrame
    filtered_df = df[
        (df['Rol'].isin(role_filter)) &
        (df['País'].isin(country_filter)) &
        (df['Edad'] >= age_filter[0]) & (df['Edad'] <= age_filter[1]) &
        (df['Meses en Arroyo'] >= months_in_company_filter[0]) & (df['Meses en Arroyo'] <= months_in_company_filter[1]) &
        (df['Años de experiencia'] >= experience_filter[0]) & (df['Años de experiencia'] <= experience_filter[1])
    ]

    # Display filtered DataFrame
    st.subheader('Filtered Dataset')
    st.write(filtered_df)

    # Summary statistics
    with st.expander('Analytics Section'):
        st.subheader('Summary Statistics')
        st.write(df.describe())

        # Additional Analytics
        st.subheader('Additional Analytics')

        # Top 10 By Roles
        role_counts = df['Rol'].value_counts().head(10)
        st.write('Top 10 By Roles')
        st.write(role_counts)

        # Gender Breakdown
        gender_counts = df['Genero'].value_counts()
        st.write('Gender Breakdown')
        st.write(gender_counts)

        # Top 10 by Age
        age_counts = df['Edad'].value_counts().head(10)
        st.write('Top 10 by Age')
        st.write(age_counts)

        # Breakdown by Country
        country_counts = df['País'].value_counts()
        st.write('Breakdown by Country')
        st.write(country_counts)

        # Top 10 by Years of Experience
        experience_counts = df['Años de experiencia'].value_counts().head(10)
        st.write('Top 10 by Years of Experience')
        st.write(experience_counts)

        # Breakdown by Nivel de Inglés
        english_level_counts = df['Nivel de inglés'].value_counts()
        st.write('Breakdown by Nivel de Inglés')
        st.write(english_level_counts)

    with st.expander('Visualizations Section'):
        # Correlation Heatmap for Numerical Features
        st.subheader('Correlation Heatmap for Numerical Features')
        plt.figure(figsize=(5, 3))
        sns.heatmap(filtered_df.select_dtypes(include=[np.number]).corr(), annot=False, cmap='viridis')
        st.pyplot(plt)
            
        # Distribution of Age
        st.subheader('Distribution of Age')
        plt.figure(figsize=(5, 3))
        sns.histplot(filtered_df['Edad'], kde=True, color='blue')
        st.pyplot(plt)
            
        # Gender Countplot
        st.subheader('Gender Distribution')
        plt.figure(figsize=(5, 3))
        sns.countplot(data=filtered_df, x='Genero', palette='Set2')
        st.pyplot(plt)
            
        # Experience vs. Nivel de inglés
        st.subheader('Years of Experience vs. Nivel de inglés')
        plt.figure(figsize=(5, 3))
        sns.scatterplot(data=filtered_df, x='Años de experiencia', y='Nivel de inglés', hue='Genero', palette='Set1')
        st.pyplot(plt)
            
        # Distribution of Nivel de inglés
        st.subheader('Distribution of Nivel de inglés')
        plt.figure(figsize=(5, 3))
        sns.histplot(filtered_df['Nivel de inglés'], kde=True, color='green')
        st.pyplot(plt)    

        # Relationship between Age and Nivel de inglés
        st.subheader('Age vs. Nivel de inglés')
        filtered_df['Edad'] = pd.to_numeric(filtered_df['Edad'], errors='coerce')
        filtered_df['Nivel de inglés'] = pd.to_numeric(filtered_df['Nivel de inglés'], errors='coerce')
        plt.figure(figsize=(5, 3))
        sns.regplot(data=filtered_df, x='Edad', y='Nivel de inglés', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
        st.pyplot(plt)
            
        # Pairplot to observe relationships between select numerical features
        st.subheader('Pairplot of Selected Numerical Features')
        selected_features = ['Edad', 'Meses en Arroyo', 'Años de experiencia', 'Nivel de inglés']
        sns.pairplot(filtered_df[selected_features], height=4)
        st.pyplot(plt)
            
        # Summary of categorical features
        st.subheader('Summary of Categorical Features')
        st.write(filtered_df.describe(include=['object']))
        
        # Countplot of Country
        st.subheader('Country Distribution')
        plt.figure(figsize=(5, 3))
        sns.countplot(data=filtered_df, x='País', palette='viridis')
        plt.xticks(rotation=45)
        st.pyplot(plt)
            
        # Feature Importance Section
        st.subheader('Feature Importance for Predicting Employee Adaptability')
        
        # Prepare data for feature importance calculation
        excluded_columns = ['ID', 'Rol', 'Genero', 'Edad', 'País', 'Meses en Arroyo', 'Años de experiencia', 'Nivel de inglés']
        X = filtered_df.drop(columns=[col for col in excluded_columns if col in filtered_df.columns] + ['Adaptabilidad'])
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
        plt.figure(figsize=(5, 3))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), palette='magma')
        plt.title('Top 15 Important Features for Predicting Employee Adaptability')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        st.pyplot(plt)

elif page == "Machine Learning Prediction":
    st.title('🤖 Machine Learning Prediction')
    st.write("This section will contain machine learning models to predict employee outcomes based on survey data.")

elif page == "Chat with Survey Data":
    st.title('💬 Chat with Survey Data')

    # File upload for survey data
    uploaded_file = st.file_uploader("Upload the survey Excel or CSV file", type=["xlsx", "csv"])
    if uploaded_file is not None:
        df_chat = load_data(uploaded_file)
        if df_chat is not None:
            st.success("Survey data loaded successfully.")
            st.write(df_chat.head())

            # Text input for OpenAI API Key
            api_key = st.text_input("Enter your OpenAI API Key", type="password")

            # Enter the query for analysis
            input_text = st.text_area("Enter your query")

            # Perform analysis
            if input_text and api_key and st.button("Chat with data"):
                chat_with_data(df_chat, input_text, api_key)

@st.cache_data
def load_data(uploaded_file):
    """Load data from the uploaded file."""
    try:
        # Load based on file type
        if uploaded_file.name.endswith("xlsx"):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        elif uploaded_file.name.endswith("csv"):
            df = pd.read_csv(uploaded_file)
        else:
            st.error("Unsupported file type.")
            return None

        # Columns to keep
        columns_to_keep = [
            'ID', 'Roles', 'Genero', 'Edad', 'País', 'Meses en Arroyo', 'Años de experiencia', 'Nivel de inglés',
            'Autogestión', 'Compromiso con la excelencia', 'Trabajo en equipo', 'Comunicación efectiva',
            'Pensamiento ánalitico', 'Adaptabilidad', 'Responsabilidad', 'Atención al detalle',
            'Liderazgo', 'Gestión de problemas', 'Orientación a resultados', 'Pensamiento estratégico',
            'Apertura', 'Iniciativa', 'Orientación al cliente', 'Autoaprendizaje',
            'Tolerancia a la presión', 'Negociación', 'Discreción', 'Integridad'
        ]
        columns_to_keep = [col for col in columns_to_keep if col in df.columns]
        df = df[columns_to_keep]

        # Convert 'Roles' column to string if it exists
        if 'Roles' in df.columns:
            df['Roles'] = df['Roles'].astype(str)

        return df

    except Exception as e:
        st.error(f"Error processing the file: {e}")
        return None

def chat_with_data(df_chat, input_text, api_key):
    """Chat with the survey data using OpenAI."""
    try:
        # Convert DataFrame to a format suitable for context
        context = df_chat.to_string(index=False)

        # Create a prompt template
        message = f"""
        Answer the following question using the context provided:

        Context:
        {context}

        Question:
        {input_text}

        Answer:
        """

        # Initialize OpenAI LLM with model 'gpt-3.5-turbo'
        llm = ChatOpenAI(model_name="gpt-4o-2024-08-06", openai_api_key=api_key)

        # Generate response
        response = llm.predict(message)

        st.write(response)

    except Exception as e:
        st.error(f"Error during chat: {e}")

if __name__ == "__main__":
    main()
