from dotenv import load_dotenv, find_dotenv
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
import os, json, random

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

_ = load_dotenv(find_dotenv())  # read local .env file

llm_model = ["gpt-3.5-turbo-1106", "gpt-4-1106-preview"]
llm = [ChatOpenAI(temperature=0, model=llm_model[0]), ChatOpenAI(temperature=0, model=llm_model[1])]

person_name = ""
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
# core_dir = os.path.join(parent_dir, "core")
# core_dir = os.path.join(os.getcwd(), "core")
# Here are two types of evaluation folder, one is in the parent directory, the other is in the current directory
evaluation_folder = [os.path.join(os.getcwd(), "evaluation", "evaluation_data", person_name)]
data_file = os.path.join(evaluation_folder[0], "background_info.json")
question_file = os.path.join(evaluation_folder[0], "evaluation_questions.json")
result_file = [os.path.join(evaluation_folder[0], llm_model[0] + "_Better_Zero_Shot_QA.json"),
               os.path.join(evaluation_folder[0], llm_model[1] + "_Better_Zero_Shot_QA.json")]
final_result_file = os.path.join(evaluation_folder[0], "Better_Zero_Shot_QA.json")

# Method of loading the background information of the person, assume it has "Name" field
def LoadData(data_file: str):
    with open(data_file, 'r') as f:
        data = json.load(f)
        person_name = data["Name"]
    return data, person_name

# Method of loading the questions from the question file
def LoadQuestions(question_file: str):
    with open(question_file, 'r') as f:
        json_data = json.load(f)
        questions = [pair["question"] for pair in json_data["qa_pairs"]]
        questions_string = "\n".join(questions)
    return questions, questions_string

def PromptModel(question_file: str, data_file: str, result_file: str, llm) -> None:
    data, person_name = LoadData(data_file)
    questions, questions_string = LoadQuestions(question_file)

    # The first stage of the prompt is to introduce the user and the background information and get the response
    user_prompt = """From now on, you called {person_name}.
     And I am one of your friend and you will answer different questions related to you.
     Here is the background information about you:
    {background_info}
     """

    stage_1_prompt = PromptTemplate(
        template=user_prompt,
        input_variables=["person_name", "background_info"],
        output_variables=["Response"],
    )

    stage_1_chain = LLMChain(llm=llm, prompt=stage_1_prompt, output_key="Response")

    # The second stage of the prompt is to ask the questions and get the answers
    stage2_human_prompt = HumanMessagePromptTemplate.from_template(
        template=user_prompt
    )

    stage2_system_prompt = SystemMessagePromptTemplate.from_template(
        template="""
    {Response}
        """
    )
    stage2_human_prompt_2 = HumanMessagePromptTemplate.from_template(
        template=questions_string
    )
    stage_2_prompt = ChatPromptTemplate.from_messages(
        [
            stage2_human_prompt,
            stage2_system_prompt,
            stage2_human_prompt_2,
        ],
    )

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

    stage_2_chain = create_structured_output_chain(output_schema=json_schema, llm=llm, prompt=stage_2_prompt,
                                                   output_key="Answers")

    from langchain.chains import SequentialChain

    overall_chain = SequentialChain(chains=[stage_1_chain, stage_2_chain], verbose=False,
                                    input_variables=["person_name", "background_info"],
                                    output_variables=["Response", "Answers"],
                                    return_all=True)

    answers = overall_chain({"person_name": person_name, "background_info": data})
    answers = answers["Answers"]

    print(answers)
    result = {
        "Baseline": "Better_Zero_Shot",
        "model": llm.model_name,
        "Answers": answers
    }

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)

def main() -> None:
    
    if not os.path.exists(data_file):
        raise Exception(f"Data File not found in {data_file}")
    
    if not os.path.exists(question_file):
        raise Exception(f"Question File not found in {question_file}")

    for i in range(2):  
        PromptModel(question_file, data_file, result_file[i], llm[i])
    
if __name__ == "__main__":
    main()