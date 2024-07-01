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
        {"question_number": 1, "question": "What is your destination?", "instructions":"this field will contain specific places with country, state, city", "answer": ""},
        {"question_number": 2, "question": "What are your travel dates?", "instructions":"this field will contain specific dates range in dd/mm/yyyy format", "answer": ""},
        {"question_number": 3, "question": "How many travelers?", "instructions":"this field will contain specific integer numbers", "answer": ""},
        {"question_number": 4, "question": "What type of accommodation?", "instructions":"this field will contain specific user preferences", "answer": ""},
        {"question_number": 5, "question": "What is your budget?", "instructions":"this field will contain specific range to money that the user is willing to spend", "answer": ""}
    ]
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

    example =[
        {"question_number": 1, "question": "What is your destination?", "instructions":"this field will contain specific places with country, state, city", "answer": "India, Goa, Baga Beach"},
        {"question_number": 2, "question": "What are your travel dates?", "instructions":"this field will contain specific dates range in dd/mm/yyyy format", "answer": "01/06/2024 - 10/06/2024"},
        {"question_number": 3, "question": "How many travelers?", "instructions":"this field will contain specific integer numbers", "answer": "3"},
        {"question_number": 4, "question": "What type of accommodation?", "instructions":"this field will contain specific user preferences", "answer": "5 star hotels"},
        {"question_number": 5, "question": "What is your budget?", "instructions":"this field will contain specific range to money that the user is willing to spend", "answer": "10000 - 15000"}
    ]

    # Define the prompt template
    prompt4 = PromptTemplate(
        template="""

            hey i am stuck in an emergency.

            The problem is that there is a visitor in my travel website but my chatbot system malfunctioned.
            
            basically my chatbot was designed to interact with the user and ask him questions to gather info about him which then will be used to give recommendations to that user for his/her vacations.

            this is the questions that needs to be answered before my recommendation system can give recommendations {current_json}


            but my chatbot broke and i am having a crises 

            You are my last hope in this situation.

            so the solution i have decided is that I will provide you with whatever the user say in the chatbot ie i will give you chat history

            and based on the latest user input you have to produce inference (ie if any of the questions have been answered by the user. if not answered leave inferences blank)

            along with inference you also have to give what reply should be given by the chatbot to take the conversation further 

            So basically You need to gather specific details from the user. 


            So far, you have asked the following questions and received these responses:
            {chat_history}




            If the user tries to skip or does not answer, politely insist on the importance of providing this information and try rephrasing the question.

            for example:
                
                for user input : "i am planning to go to goa baga beach "
                your answer should be like:
                {{
                    "inferences": [
                        {{
                            "question_number": 1,
                            "answer": "India, Goa, Baga beach"
                        }}
                    ],
                    "next_question": "this will contain your next reply as the chatbot "
                }}

            Answer in the above JSON format only and return me the JSON only, nothing else.
            remember i will replace the answer of that question number with your answer and update the json

            
            before answering carefully read the latest conversation between  the user and the bot (here the "bot" is you and "you" will be the user )
            latest user input : {latest_user_input}



            fill the json in such a way that the final questions json looks something like this : 
            [
                {{"question_number": 1, "question": "What is your destination?", "instructions":"this field will contain specific places with country, state, city", "answer": "India, Goa, Baga Beach"}},
                {{"question_number": 2, "question": "What are your travel dates?", "instructions":"this field will contain specific dates range in dd/mm/yyyy format", "answer": "01/06/2024 - 10/06/2024"}},
                {{"question_number": 3, "question": "How many travelers?", "instructions":"this field will contain specific integer numbers", "answer": "3"}},
                {{"question_number": 4, "question": "What type of accommodation?", "instructions":"this field will contain specific user preferences", "answer": "5 star hotels"}},
                {{"question_number": 5, "question": "What is your budget?", "instructions":"this field will contain specific range to money that the user is willing to spend", "answer": "10000 - 15000"}}
            ]

            note you bastard this is just an example only. you have to  fill based on user input. tis example is just for you to learn. do not take these values as the answers. dont avoid your work

            """,
        input_variables=["chat_history", "current_json", "latest_user_input"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Create the LLMChain
    chain = prompt4 | model | parser

    # Prepare the input for the chain
    input_data = {
        "chat_history": chat_history,
        "current_json": json.dumps(current_json, indent=4),
        "latest_user_input": chat_history[-1],
    }

    # Invoke the chain
    response = chain.invoke(input_data)
    
    # Convert response to dictionary
    updated_json = response

    # Update the current JSON with the new answers
    if "inferences" in updated_json and updated_json["inferences"]:
        for update in updated_json["inferences"]:
            for question in current_json["questions"]:
                if question["question_number"] == update["question_number"]:
                    question["answer"] = update["answer"]
                    break

    print ("response : ",json.dumps(response, indent=4) )

    return current_json, updated_json.get("next_question", "")



def initialize_session_state():
    """Initialize session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
        chatbot_intro = f"""Hello! I am here to help you plan your vacations. Where are you looking to travel to? Any specific destination in mind or any preferences you have in terms of the type of place you want to visit?"""
        st.session_state['chat_history'].append(f"Bot: {chatbot_intro}")
    if 'json_data' not in st.session_state:
        st.session_state['json_data'] = json_template
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
        print("user input is : ", user_input)
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

def clear_session_state():
    """Clear all session state variables."""
    for key in st.session_state.keys():
        del st.session_state[key]

        
if __name__ == "__main__":
    main()