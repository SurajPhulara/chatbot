from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from typing import List, Dict, Optional
import streamlit as st
import json
from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPEN_API_KEY2")

# Define the JSON structure for parsing the OpenAI response
class FieldUpdate(BaseModel):
    field_name: str
    answer: str

class ResponseStructure(BaseModel):
    inferences: Optional[List[FieldUpdate]] = None
    next_question: str


def read_abc_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(data)
    return data

# Read the abc.json file
abc_json = read_abc_json('abc.json')



# Sample JSON template (this will be dynamically updated)
tripplan_json = {
    # "optimizeType": {},
    "firstDestination": {},
    "trip_theme": {},
    "destination": {},
    "traveller_type": {},
    "Origin_city": {},
    "budget": {},
    "food": {},
    "trip_direction": {}
}

def call_openai_api(chat_history, current_json):
    if 'function_count' not in st.session_state:
        st.session_state['function_count'] = 0
    else:
        st.session_state['function_count'] += 1
        print("Total number of times the GPT call is made is:", st.session_state['function_count'])

    model = ChatOpenAI(api_key=openai_api_key, temperature=0)

    # Define the parser
    parser = JsonOutputParser(pydantic_object=ResponseStructure)

    example = {
        "optimizeType": "manual",
        "firstDestination": "Goa",
        "trip_theme": "beach",
        "destination": ["Goa", "Kerala"],
        "traveller_type": "family",
        "Origin_city": "Mumbai",
        "budget": "comfortable spending",
        "food": "vegetarian",
        "trip_direction": "return"
    }

    # Define the prompt template
    prompt = PromptTemplate(
        template="""
            You are a helpful assistant. You will receive the current conversation history and a JSON template that needs to be filled based on the user inputs.

            details of how you have to fill json and what each filed in json is for so that you can ask proper questions : {details}

            chat history : 
            {chat_history}

            The current JSON template is:
            {current_json}

            Please provide any inferences you can make from the conversation in the following format:
            {{
                "inferences": [
                    {{"field_name": "firstDestination", "answer": "Goa"}},
                    {{"field_name": "trip_theme", "answer": "beach"}},
                    {{"field_name": "destination", "answer": ["Goa", "Kerala"]}},
                    {{"field_name": "traveller_type", "answer": "family"}},
                    {{"field_name": "Origin_city", "answer": "Mumbai"}},
                    {{"field_name": "budget", "answer": "comfortable spending"}},
                    {{"field_name": "food", "answer": "vegetarian"}},
                    {{"field_name": "trip_direction", "answer": "return"}}
                ],
                "next_question": "What is your first destination?"
            }}

            this is jsut an example for you to learn.

            only give inferences about the fields that are actually answered by the user and dont make up any data by yourself
            and do not ask the questions that have already been answered. 

            only give inferences to the questions that the user have answered and check the history before answering

            remember to carefully analyze the user query and chat history and then give your next question as a human would do.
            remember first think then answer

            INSTRUCTIONS:
                - communicate as a human be kind and polite and speak directly (be interactive dont be exact straight forward try to get that data out of him by politely asking and the same thing in another way to increase user retention)  (most important)
                - remember never to ask direct question. be polite you know how to handle customers right. take the context of the chat history before answering
                - along with each question tell the user what type of response you are excepting
        """,
        input_variables=["chat_history", "current_json", "latest_user_input"],
    )

    # Create the LLMChain
    chain = prompt | model | parser

    # Prepare the input for the chain
    input_data = {
        "chat_history": chat_history,
        "current_json": json.dumps(current_json, indent=4),
        "latest_user_input": chat_history[-1],
        "details": json.dumps(abc_json, indent=4),
    }

    # Invoke the chain
    response = chain.invoke(input_data)

    # Convert response to dictionary
    updated_json = response

    # Update the current JSON with the new answers
    if "inferences" in updated_json and updated_json["inferences"]:
        for update in updated_json["inferences"]:
            current_json[update["field_name"]] = update["answer"]

    print("response:", json.dumps(response, indent=4))

    return current_json, updated_json.get("next_question", "")

def initialize_session_state():
    """Initialize session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
        chatbot_intro = "Hello! I am here to help you plan your vacation. Let's get started! What is your destination?"
        st.session_state['chat_history'].append(f"Bot: {chatbot_intro}")
    if 'json_data' not in st.session_state:
        st.session_state['json_data'] = tripplan_json
    if 'next_question' not in st.session_state:
        st.session_state['next_question'] = ""
    if 'function_count' not in st.session_state:
        st.session_state['function_count'] = 0

def handle_user_input(user_input):
    """
    Handle user input: update chat history, call OpenAI API, and update JSON data.
    
    Args:
        user_input (str): The user input text.
    """
    st.session_state['chat_history'].append(f"You: {user_input}")

    # Call OpenAI API to update JSON data
    updated_json, next_question = call_openai_api(
        st.session_state['chat_history'],
        st.session_state['json_data']
    )
    st.session_state['json_data'] = updated_json
    st.session_state['next_question'] = next_question

    # Generate a chatbot response
    chatbot_response = f"{next_question}"
    st.session_state['chat_history'].append(f"Bot: {chatbot_response}")

def render_chatbot_ui():
    """Render the chatbot user interface."""
    st.header("Chatbot")
    st.markdown(
        """
        <style>
        .stTextInput > div > div > input {
            width: 100%;
        }
        </style>
        """, unsafe_allow_html=True)

    user_input = st.text_input("You:", "")
    
    if user_input:
        print("user input is:", user_input)
        handle_user_input(user_input)
    
    chat_history_container = st.container()
    with chat_history_container:
        for chat in st.session_state['chat_history']:
            st.write(chat)

def render_json_ui():
    """Render the JSON data user interface."""
    st.header("Current JSON Data")
    st.json(st.session_state['json_data'])

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(layout="wide")
    st.title("Travel Planner Chatbot")

    initialize_session_state()

    # Create two columns with wider width
    col1, col2 = st.columns([2, 2])

    # Left column: Chatbot interaction
    with col1:
        render_chatbot_ui()

    # Right column: Display JSON data
    with col2:
        render_json_ui()

    # Update the displayed JSON data
    # st.write(json.dumps(st.session_state['json_data'], indent=4))

def clear_session_state():
    """Clear all session state variables."""
    for key in st.session_state.keys():
        del st.session_state[key]

if __name__ == "__main__":
    main()
