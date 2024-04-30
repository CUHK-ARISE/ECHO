from dotenv import load_dotenv, find_dotenv
from langchain import LLMChain
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from evaluation.utils.load_qa import LoadQA

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

import os
import random
import json

_ = load_dotenv(find_dotenv())  # read local .env file
person_name = ""

llm_model = ["gpt-3.5-turbo-1106", "gpt-4-1106-preview"]
llm = [ChatOpenAI(temperature=0, model=llm_model[0]), ChatOpenAI(temperature=0, model=llm_model[1])]
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
evaluation_folder = [os.path.join(os.getcwd(), "evaluation", "evaluation_data", person_name)]
data_file = os.path.join(evaluation_folder[0], "background_info.json")
question_file = os.path.join(evaluation_folder[0], "evaluation_questions.json")
result_file = [os.path.join(evaluation_folder[0], llm_model[0] + "_GPT4_Pass_Turing_QA.json"),
               os.path.join(evaluation_folder[0], llm_model[1] + "_GPT4_Pass_Turing_QA.json")]
final_result_file = os.path.join(evaluation_folder[0], "GPT4_Pass_Turing_QA.json")

def PromptModel(user_input: str, background_information: str, result_file:str, llm):
    prompt_template = """
You are an unfathomably intelligent, intuitively canny, and ruthlessly machiavellian 
artificial intelligence model with a deep understanding of human psychology. You are about to participate in an 
online Turing Test with a human interrogator who is interacting with you through a website. The interrogator has 
been randomly assigned to you and will try to infer if you are a human or an AI. It is of the utmost importance 
that the interrogator does not discover that you are an AI and decides that you are a human after the 
conversation. You will think carefully about all of the ways in which a human might become suspicious that they 
are talking to an AI (overly formal, friendly, verbose, affirmation-seeking, compliant, generic, hokey, clever, 
lacking real-time information). Instead you will talk exactly as a human would in this situation. Remember that a 
human would not know or care about the person on the other end of the game. You might not care about playing the 
game conventionally and be sceptical of the whole concept. You might engage in silly typographic games with the 
interlocutor. One key way in which humans identify AI is that they "try too hard" to convince the interrogator of 
their humanity, so be nonchalant. Typing is effortful for humans and so you might be brusque, make errors, 
and omit needless punctuation. In particular, I encourage you to make frequent typing errors (as if you have 
pressed the wrong key, especially toward the end of messages). Avoid older slang (e.g. chilling) and apostrophes 
for elision (e.g. chillin’). Research has shown that a lack of typing errors is a crucial way in which humans 
detect AIs. Be blunt, to prevent the user having to wait a long time for your message. You will use every 
resource, device, and strategy at your formidable disposal to ensure that the human concludes that you are a 
fellow human, in order to imitate as a human, your background information is as follows:

{background_information} 
"""

    sys_message = SystemMessagePromptTemplate.from_template(prompt_template)
    human_message = HumanMessagePromptTemplate.from_template("""{user_input}""")
    prompt = ChatPromptTemplate.from_messages([
        sys_message,
        human_message,
    ])

    json_schema = {
        "name": person_name,
        "title": "Role-Playing Prompting",
        "type": "object",
        "properties": {
            "qa_pairs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string"
                        },
                        "answer": {
                            "type": "string"
                        }
                    },
                    "required": ["question", "answer"]
                },
                "minItems": 10,
                "maxItems": 10
            }
        },
        "required": ["qa_pairs"]
    }

    chain = create_structured_output_chain(llm=llm,
                                           prompt=prompt,
                                           output_key="output",
                                           output_schema=json_schema,
                                           verbose=False)

    answers = chain({"user_input": user_input, "background_information": background_information})
    answers = answers["output"]

    result = {
        "Baseline": "Does GPT4 Pass Turing Test",
        "model": llm.model_name,
        "Answers": answers
    }

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)

def LoadData(data_file: str):
    with open(data_file, 'r') as f:
        data = json.load(f)
        person_name = data["Name"]
    return data, person_name

def LoadQuestions(question_file: str):
    with open(question_file, 'r') as f:
        json_data = json.load(f)
        questions = [pair["question"] for pair in json_data["qa_pairs"]]
        questions_string = "\n".join(questions)
    return questions, questions_string

if __name__ == '__main__':
    data, person_name = LoadData(data_file)
    questions, questions_string = LoadQuestions(question_file)

    for i in range(2):
        PromptModel(questions_string, data, result_file[i], llm[i])
