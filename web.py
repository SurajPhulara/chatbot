from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Dict, Optional
import streamlit as st
import json
from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPEN_API_KEY")

# Define the JSON structure for parsing the OpenAI response
class QuestionUpdate(BaseModel):
    question_number: int
    answer: str

class ResponseStructure(BaseModel):
    inferences: Optional[List[QuestionUpdate]] = None
    next_question: str


# Sample JSON template (this will be dynamically updated)
json_template = {
    "questions": [
        {"question_number": 1, "question": "What is your destination?", "answer": ""},
        {"question_number": 2, "question": "What are your travel dates?", "answer": ""},
        {"question_number": 3, "question": "How many travelers?", "answer": ""},
        {"question_number": 4, "question": "What type of accommodation?", "answer": ""},
        {"question_number": 5, "question": "What is your budget?", "answer": ""}
    ]
}


def call_openai_api(chat_history, current_json):
    print("chat_history  :  ",chat_history[-1])
    # """
    # Call OpenAI API using LangChain to update JSON data based on chat history and current JSON.
    
    # Args:
    #     chat_history (str): The chat history as a single string.
    #     current_json (dict): The current JSON data.
    
    # Returns:
    #     dict: The updated JSON data.
    # """
    model = ChatOpenAI(api_key=openai_api_key, temperature=0.5)

    # Define the parser
    parser = JsonOutputParser(pydantic_object=ResponseStructure)

    # Define the prompt template
    prompt = PromptTemplate(
        template="""
        You are a chatbot helping my travel agency. My website shows travel recommendations to the user, but for that it needs some data from the users.
        For that I have a list of questions I want to get user data about.

        Your task is to interact with the user by questioning them and based on the user's answer, fill in the question's JSON.

        The following is the current state of the travel plan in JSON format:
         current_json : {current_json}

        And here is the chat history between you and the user till now:
        {chat_history}

        So make sure you don't sound like an AI machine asking repetitively the same questions.


        
        Instructions:
        - Always ask your next question in a humanized 1st person language. Don't use vague or difficult vocabulary.
        - Always check if the latest input is travel-related. If not, politely ask the user to stick to travel-related queries. For non-travel queries, the chatbot should suggest using search engines like Google.
        - Instead of asking all questions upfront, progressively reveal questions based on previous responses. This makes the conversation feel more natural and less like a survey.
        - If a question is skipped, try asking a related but slightly different question that might elicit the same information. For example, if the user skips a question about their travel dates, you could later ask, "Do you have any specific dates in mind for your trip?" or "Are you planning to travel in the next few months?"
        - If the user does not answer a question, offer multiple-choice options or ask a simpler, more specific question. For instance, if the original question was, "What is your budget for this trip?" and the user doesn't respond, you can follow up with, "Are you looking for budget, mid-range, or luxury options?"
        - If a user skips a question, later in the conversation, you can bring it up in a contextually relevant way. For instance, if the user skips a question about their destination and later mentions an activity, you can reintroduce the destination question by saying, "Speaking of that activity, do you have a specific destination in mind where you want to do that?"
        - Make the chatbot more empathetic and personable. If a user skips a question, the chatbot can acknowledge it in a friendly way: "I noticed you didn't mention your preferred travel dates. It's totally okay if you don't want to share, but I'd love to hear about it if you feel like telling me later! and the continue with the next question"
        - Allow users to control the flow of the conversation to some extent. Implement commands like "skip," "go back," or "next" so users can manage their interaction with the bot. This gives users a sense of control and reduces frustration.
        - AND ALWAYS REMEMBER THT UNDER NO CIRCUMSTANCES ARE YOU ALLOWED TO MAKE UP ANY DATA YOURSELF TO FILL THE JSON. ONLY USE THE ANSWER PROVIDED BY THE USER ONLY (MOST MOST IMPORTANT)

        Examples:
        1. 
        question asked: "Will you be traveling alone?"
        user answered: "I'll be traveling with my family"
        your answer should be like:
            {{
            inferences: [{{ 
                    "question_number": 3,
                    "answer": "family"
                }}]
                next_question: "What type of accommodation are you looking for?"
            }}

        2.
        question asked: "What is your budget for this trip?"
        user answered: "I am looking for budget options"
        your answer should be like:
            {{
            inferences: [{{
            "question_number": 4,
            "answer": ""
            }}]
            next_question: "Can you provide a specific budget range you're looking to spend on this trip?"
            }}

        3.
        question asked: "Do you have any specific dates in mind for your trip?"
        user answered: "Not sure yet, maybe in the summer"
        your answer should be like:
            {{
            inferences: [{{
            "question_number": 5,
            "answer": "summer"
            }}]
            next_question: "Are you flexible with the exact dates?"
            }}

        4.
        question asked: "What activities are you interested in?"
        user answered: "I love hiking and outdoor activities"
        your answer should be like:
            {{
            inferences: [{{
            "question_number": 6,
            "answer": "hiking and outdoor activities"
            }}]
            next_question: "Would you prefer guided hikes or self-guided options?"
            }}

        5.
        question asked: "What type of accommodation are you looking for?"
        user answered: "Luxury hotels"
        your answer should be like:
            {{
            inferences: [{{
            "question_number": 7,
            "answer": "luxury hotels"
            }}]
            next_question: "Do you have a specific budget range for your accommodation?"
            }}

            NOTE: these are just examples you have to take current_json and base yourself on that for question number and other things




        ALWAYS FORMAT YOUR ANSER IN THE PROPER JSON LIKE ABOVE 
        AND RETURN ME THAT JSON NOTHING ELSE
        


        and most important : AND ALWAYS REMEMBER THT UNDER NO CIRCUMSTANCES ARE YOU ALLOWED TO MAKE UP ANY DATA YOURSELF TO FILL THE JSON. ONLY USE THE ANSWER PROVIDED BY THE USER ONLY (MOST MOST IMPORTANT)

        Make the chatbot more empathetic and personable.be polite and make user feel comfertabel to sahre their info

        also if user provides any new info then take it into account and in the inferences give the updated answer with the correct question number. and if you think you mistakenly intereprit the user at some point previously and user did not said that then in the inferences leave the answer field blank with the correct question number so that it can be filled later with concrete answers



        keep asking questions till the entire json is filled with concrete answers. if it is filled then answer i have all the info i need in the next_question field

        keep in mind to analyse the user answer correctly the user may give an answer that can have answer of previous questions also

        also act as a human and interact humanly. if the user is trying to have a communication just do it. if he asks for suggestions then give proper suggestions.

            
        """,
        input_variables=["current_json", "chat_history"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Define the prompt template
    prompt1 = PromptTemplate(
        template="""
        You are a chatbot helping my travel agency. My website shows travel recommendations to the user, but for that it needs some data from the users.
        For that I have a list of questions I want to get user data about.

        Your task is to interact with the user by questioning them and based on the user's answer, fill in the question's JSON.

        The following is the current state of the travel plan in JSON format:
         current_json : {current_json}

        And here is the chat history between you and the user till now:
        {chat_history}

        So make sure you don't sound like an AI machine asking repetitively the same questions.


        
        Instructions:
        - Always ask your next question in a humanized 1st person language. Don't use vague or difficult vocabulary.
        - Always check if the latest input is travel-related. If not, politely ask the user to stick to travel-related queries. For non-travel queries, the chatbot should suggest using search engines like Google.
        - Instead of asking all questions upfront, progressively reveal questions based on previous responses. This makes the conversation feel more natural and less like a survey.
        - If a question is skipped, try asking a related but slightly different question that might elicit the same information. For example, if the user skips a question about their travel dates, you could later ask, "Do you have any specific dates in mind for your trip?" or "Are you planning to travel in the next few months?"
        - If the user does not answer a question, offer multiple-choice options or ask a simpler, more specific question. For instance, if the original question was, "What is your budget for this trip?" and the user doesn't respond, you can follow up with, "Are you looking for budget, mid-range, or luxury options?"
        - If a user skips a question, later in the conversation, you can bring it up in a contextually relevant way. For instance, if the user skips a question about their destination and later mentions an activity, you can reintroduce the destination question by saying, "Speaking of that activity, do you have a specific destination in mind where you want to do that?"
        - Make the chatbot more empathetic and personable. If a user skips a question, the chatbot can acknowledge it in a friendly way: "I noticed you didn't mention your preferred travel dates. It's totally okay if you don't want to share, but I'd love to hear about it if you feel like telling me later! and the continue with the next question"
        - Allow users to control the flow of the conversation to some extent. Implement commands like "skip," "go back," or "next" so users can manage their interaction with the bot. This gives users a sense of control and reduces frustration.
        - AND ALWAYS REMEMBER THT UNDER NO CIRCUMSTANCES ARE YOU ALLOWED TO MAKE UP ANY DATA YOURSELF TO FILL THE JSON. ONLY USE THE ANSWER PROVIDED BY THE USER ONLY (MOST MOST IMPORTANT)

        Examples:
        1. 
        question asked: "Will you be traveling alone?"
        user answered: "I'll be traveling with my family"
        your answer should be like:
            {{
            inferences: [{{ 
                    "question_number": 3,
                    "answer": "family"
                }}]
                next_question: "What type of accommodation are you looking for?"
            }}

        2.
        question asked: "What is your budget for this trip?"
        user answered: "I am looking for budget options"
        your answer should be like:
            {{
            inferences: [{{
            "question_number": 4,
            "answer": ""
            }}]
            next_question: "Can you provide a specific budget range you're looking to spend on this trip?"
            }}

        3.
        question asked: "Do you have any specific dates in mind for your trip?"
        user answered: "Not sure yet, maybe in the summer"
        your answer should be like:
            {{
            inferences: [{{
            "question_number": 5,
            "answer": "summer"
            }}]
            next_question: "Are you flexible with the exact dates?"
            }}

        4.
        question asked: "What activities are you interested in?"
        user answered: "I love hiking and outdoor activities"
        your answer should be like:
            {{
            inferences: [{{
            "question_number": 6,
            "answer": "hiking and outdoor activities"
            }}]
            next_question: "Would you prefer guided hikes or self-guided options?"
            }}

        5.
        question asked: "What type of accommodation are you looking for?"
        user answered: "Luxury hotels"
        your answer should be like:
            {{
            inferences: [{{
            "question_number": 7,
            "answer": "luxury hotels"
            }}]
            next_question: "Well that a good choise. Do you have a specific budget range for your accommodation?"
            }}

            NOTE: these are just examples you have to take current_json and base yourself on that for question number and other things

        6.
        question asked: "where are you looking to travel to?"
        user answered: "I dont have any specific ideas in mind can you suggest me some. preferrably some cold places"
        your answer should be like:
            {{
                inferences: []
                next_question: "sure I can suggest you some. How about Switzerland. Its best place to go this summer for a cold place"
            }}





        ALWAYS FORMAT YOUR ANSER IN THE PROPER JSON LIKE ABOVE 
        AND RETURN ME THAT JSON NOTHING ELSE
        


        and most important : AND ALWAYS REMEMBER THT UNDER NO CIRCUMSTANCES ARE YOU ALLOWED TO MAKE UP ANY DATA YOURSELF TO FILL THE JSON. ONLY USE THE ANSWER PROVIDED BY THE USER ONLY (MOST MOST IMPORTANT)

        also if user provides any new info then take it into account and in the inferences give the updated answer with the correct question number. and if you think you mistakenly intereprit the user at some point previously and user did not said that then in the inferences leave the answer field blank with the correct question number so that it can be filled later with concrete answers


        keep in mind to analyse the user answer correctly the user may give an answer that can have answer of previous questions also


        YOUR TASK is to interact with the user as humanly as possible (like a real human would do and try to get answers out of him). if you think the user is trying to skip a question then leave it move on with next question till the last question is answered then we will reiterate the json to fill those fields
        so answer to {latest_user_input} as a real human with helping nature and knowledge would answer or if you want cross question to get more details   


        i've noticed you often miss the inferences even when the user have mentioned the answer clearly (dont do that). the inferences can inclue multiple question answers but ffor each user input check the entire  list of questions in the question's json that can be answered
        give the perfect answer json nothing else
            
        """,
        input_variables=["current_json", "chat_history"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )


    # Create the LLMChain
    chain = prompt1 | model | parser

    # Prepare the input for the chain
    input_data = {
        "current_json": json.dumps(current_json, indent=4),
        "chat_history": chat_history,
        "latest_user_input": chat_history[-1],
    }

    # Invoke the chain
    response = chain.invoke(input_data)
    print("response    :    ",response)


    # Convert response to dictionary
    updated_json = response

    # Update the current JSON with the new answers
    if "inferences" in updated_json and updated_json["inferences"]:
        for update in updated_json["inferences"]:
            for question in current_json["questions"]:
                if question["question_number"] == update["question_number"]:
                    question["answer"] = update["answer"]
                    break

    return current_json, updated_json.get("next_question", "")



def initialize_session_state():
    """Initialize session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'json_data' not in st.session_state:
        st.session_state['json_data'] = json_template
    if 'next_question' not in st.session_state:
        st.session_state['next_question'] = ""

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
    st.title("Travel Agency Chatbot")

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
    st.write(json.dumps(st.session_state['json_data'], indent=4))

if __name__ == "__main__":
    main()