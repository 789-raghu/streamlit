import sys
import subprocess

requirements = ["pandas",
              "streamlit",
              "langchain",
              "langchain_experimental"]
for lib in requirements:
    subprocess.check_call(["pip","install",lib])

import os
import pandas as pd
import streamlit as st
import os
import pandas as pd
import streamlit as st
from langchain.llms import OpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent

# Set the OpenAI API key (Consider using Streamlit secrets for better security)
st.sidebar.title("OpenAI API Key")
api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

# Load the data
df = pd.read_csv("customers.csv")

# Set the title of the Streamlit app
st.title("Data Analysis Agent")

# Sidebar for model parameters
st.sidebar.title("Model Parameters")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.number_input("Max Tokens", min_value=1, max_value=1024, value=150)

if api_key:
    # Initialize GPT model for data analysis
    data_analysis_agent = create_pandas_dataframe_agent(
        OpenAI(temperature=temperature, max_tokens=max_tokens),
        df,
        verbose=True
    )

    # Initialize session state for messages if not already done
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input field for new messages
    if prompt := st.chat_input("Ask your data analysis question here"):
        # Store user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from the GPT model
        response = data_analysis_agent.run(prompt)
        
        # Store and display assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
else:
    st.warning("Please enter your OpenAI API key to use the app.")

from langchain_openai import OpenAI
#from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = "sk-proj-WHjQr9dsmdCMr00fu49fT3BlbkFJpRk9RRwfpliFyO2ozpiU"

# Load the data
df = pd.read_csv("customers.csv")

# Set the title of the Streamlit app
st.title("Data Analysis Agent")

# Sidebar for model parameters
st.sidebar.title("Model Parameters")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.number_input("Max Tokens", min_value=1, max_value=1024, value=150)

# Initialize GPT model for data analysis
data_analysis_agent = create_pandas_dataframe_agent(
    OpenAI(temperature=temperature), 
    df, 
    verbose=True
)

# Initialize session state for messages if not already done
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input field for new messages
if prompt := st.chat_input("What is up?"):
    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from the GPT model
    response = data_analysis_agent.run(prompt)
    
    # Store and display assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
