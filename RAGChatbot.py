import pandas as pd
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain

# Define the Streamlit app
def main():
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
