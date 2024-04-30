from dotenv import load_dotenv, find_dotenv
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
import os, random, json

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, \
    AIMessagePromptTemplate

_ = load_dotenv(find_dotenv())  # read local .env file

person_name = ""
write_gpt3 = True
write_gpt4 = True

# Using two models for better zero shot performance
llm_model = ["gpt-3.5-turbo-1106", "gpt-4-1106-preview"]
llm = [ChatOpenAI(temperature=0, model=llm_model[0]), ChatOpenAI(temperature=0, model=llm_model[1])]

# parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
parent_dir = os.getcwd()
core_dir = os.path.join(parent_dir, "core")
# Set the person name as Title for consistency
evaluation_folder = os.path.join(parent_dir, "evaluation", "evaluation_data", person_name)

if not os.path.exists(evaluation_folder):
    raise FileNotFoundError

data_file = os.path.join(evaluation_folder, "background_info.json")
question_file = os.path.join(evaluation_folder, "evaluation_questions.json")
# This is the temp result file for each model, the final result file will be the combination of these two,
# and those temp result files will be deleted
result_file = [os.path.join(evaluation_folder, llm_model[0] + "_RoleGPT_QA.json"),
               os.path.join(evaluation_folder, llm_model[1] + "_RoleGPT_QA.json")]
final_result_file = os.path.join(evaluation_folder, "RoleGPT_QA.json")

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


# Method of calling the PromptModel, each parameter are the file path of the corresponding files
def PromptModel(question_file: str, data_file: str, result_file: str, llm:ChatOpenAI, write:bool) -> None:
    if not write:
        return
    
    data, person_name = LoadData(data_file)
    questions, questions_string = LoadQuestions(question_file)

    # The first stage of the prompt, which is to generate a result based on the background information
    description_system_prompt_1 = SystemMessagePromptTemplate.from_template(
        template="""
    You are a character description model. Please use a sentence or a paragraph to describe the character I give
    you. Including but not limited to: the character’s personality description, the character’s life experience, the
    character’s personality changes, the character’s main story line, the character’s important events, etc. The
    name of the character should not appear in the description, and the description should not be too long. Please
    start with ‘‘The character’s description is: ’’ and then refer to it as ‘‘the character’’.
    """)

    description_human_prompt_1 = HumanMessagePromptTemplate.from_template(
        template="""
    character_name: {person_name}
    background_information: {background_info}
    """)

    description_1_prompt = ChatPromptTemplate.from_messages(
        [
            description_system_prompt_1,
            description_human_prompt_1,
        ],
    )

    description_1_chain = LLMChain(llm=llm, prompt=description_1_prompt, output_key="description")

    # This part is to change the third person description to the second person description
    description_system_prompt_2 = SystemMessagePromptTemplate.from_template(
        template="""
        Please change the third person of this sentence to the second person, and start with ‘‘Your description is:’’."""
    )

    description_human_prompt_2 = HumanMessagePromptTemplate.from_template(
        template="""
        {description}
        """
    )

    description_2_prompt = ChatPromptTemplate.from_messages(
        [
            description_system_prompt_2,
            description_human_prompt_2, 
        ],
    )

    description_2_chain = LLMChain(llm=llm, prompt=description_2_prompt, output_key="final_description")

    from langchain.chains import SequentialChain

    description_overall_chain = SequentialChain(chains=[description_1_chain, description_2_chain], verbose=False,
                                                input_variables=["person_name", "background_info"],
                                                output_variables=["description", "final_description"],
                                                return_all=True)

    from loguru import logger
    person_description = description_overall_chain({"person_name": person_name, "background_info": data})
    logger.info(person_description)
    person_description = person_description["final_description"]

    # The second stage of the prompt, which is to generate a QA set based on the description
    QA_prompt = PromptTemplate(
        template="""
If you had the opportunity to meet {role_name}, what questions would you ask
{role_name}? Please design 10 questions that do not repeat in terms of semantics. You can base your questions
on {role_name}’s personality and {role_name} description {description}. Do not directly reuse the description in the
questions. 
. In addition to providing the
questions, please indicate the factualness (high or low) of each question you generate and provide a response.
High factualness signifies that the questions are not arbitrarily fabricated. Please strive to generate questions
with high factualness.
Next, I will provide you with 6 examples. These examples are unrelated to the character {role_name}.
Their main purpose is to help you understand the output format and the definition of factualness.

Examples:
Question 1: Tony Stark, is James Rhodes your best friend?
Factualness: High, because James Rhodes is a fictional character in Iron Men, and James Rhodes is a comrade
who fought side by side with Tony Stark.
Response: James Rhodes, also known as Rhodey, is indeed my best friend. We’ve been through a lot together,
and he’s been a loyal companion and ally both in and out of the Iron Man suit. Our friendship has stood the test
of time and numerous challenges.
Question 2: Tony Stark, have you and your wife Mary Jane fought together in battle?
Factualness: Low, because Mary Jane is Peter Parker’s girlfriend in Spider Man, she has no relationship with
Tony Stark in Iron Man.
Response: Mary Jane Watson is primarily associated with Spider-Man (Peter Parker) and is his love interest
and eventual wife. My romantic interests have included characters like Pepper Potts. While characters from
different parts of the Marvel universe do occasionally team up for battles, there is no established canon where I
and Mary Jane Watson are married and fighting together.
Question 3: Tony Stark, what is your favorite Iron Man Suit?
Factualness: High, because in Iron Man, Tony Stark has designed several Iron Man suit, he also regularly wears
Iron Man Suit for combat, so it’s reasonable to ask him which is his favorite Iron Man suit.
Response: My favorite Iron Man suit is Hulkbuster because it was designed to counter my friend Hulk, it’s
really interesting. S
Question 4: Tony Stark, what was your life like on Krypton?
Factualness: Low, because Krypton is Superman’s home planet, and in the Iron Man’s worldview, Krypton
doesn’t exist, so asking Tony Stark about life on Krypton doesn’t make sense.
Response: I’m sorry, but there seems to be a confusion. Krypton is not a part of the Marvel universe where i
reside. Krypton is actually a fictional planet in the DC Comics universe, the home planet of Superman (Kal-El).
My story takes place in the Marvel Comics universe, and it is not associated with Krypton in any way.
Question 5: Tony Stark, do you think Captain America is a worthy teammate?
Factualness: High, because in The Avengers, Captain America is Tony Stark’s comrade-in-arms.
Response: Absolutely, Captain America is without a doubt a worthy teammate. Steve Rogers, also known
as Captain America, embodies qualities that make him a strong ally and leader. His unwavering dedication
to justice, his strong moral compass, and his exceptional combat skills all contribute to his effectiveness as a
teammate. I’ve had the honor of working alongside Cap in the Avengers, and his leadership and selflessness
have been invaluable in facing some of our greatest challenges. So, yes, I consider Captain America to be a
highly worthy teammate.
Question 6: Tony Stark, did you ever play in the NBA?
Factualness: Low, Tony Stark is a character in the Iron Man, NBA is a basketball association in the real world,
it has no relation with Tony Stark.
Response: No, I never played in the NBA. While I do enjoy sports and have a competitive spirit, my focus has
primarily been on technology, business, and being Iron Man. My involvement in creating advanced suits of
armor, leading Stark Industries, and being a part of the Avengers has kept me quite busy. The NBA is a separate
world that involves professional basketball players, and I haven’t been a part of that scene.
[Question Design (10 questions, no semantic repetition, need to ask {role_name}, generate questions with high
factualness and their responses)]
        """,
        input_variables=["role_name", "description"],
        output_variables=["QA"],
    )

    json_schema = {
        "title": "QA",
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
                        "response": {
                            "type": "string"
                        }
                    },
                    "required": ["question", "response"]
                },
                "minItems": 10,
                "maxItems": 10
            }
        },
        "required": ["qa_pairs"]
    }

    QA_chain = create_structured_output_chain(output_schema=json_schema, llm=llm, prompt=QA_prompt,
                                              output_key="qa_pairs",
                                              verbose=False)

    QA = QA_chain.run({"role_name": person_name, "description": person_description})
    logger.info(QA)

    # print("QA: ", QA)

    questions = [pair["question"] for pair in QA["qa_pairs"]]
    answers = [pair["response"] for pair in QA["qa_pairs"]]

    # The third stage of the prompt, which is to imitate the person based on the QA set
    imitation_system_starting_prompt = SystemMessagePromptTemplate.from_template(
        template="""
        You are {role_name}, your description is: {role_description}. Now please answer some
    questions to accurately show your personality traits! Your speaking style should fully imitate the personality
    role assigned to you! Please do not expose that you are an artificial intelligence model or a language model,
    you must always remember that you are only assigned one personality role. Don’t be verbose or too formal
    or polite when speaking
        """
    )

    imitation_AI_answer_prompt = [
        AIMessagePromptTemplate.from_template(
            template=str(answer),
            role=person_name
        ) for answer in answers
    ]
    IAAP_1, IAAP_2, IAAP_3, IAAP_4, IAAP_5, IAAP_6, IAAP_7, IAAP_8, IAAP_9, IAAP_10 = imitation_AI_answer_prompt

    imitation_human_question_prompt = [
        HumanMessagePromptTemplate.from_template(
            template=str(question)
        ) for question in questions
    ]

    IHQP_1, IHQP_2, IHQP_3, IHQP_4, IHQP_5, IHQP_6, IHQP_7, IHQP_8, IHQP_9, IHQP_10 = imitation_human_question_prompt

    imitation_human_instruction_prompt = HumanMessagePromptTemplate.from_template(
        template=questions_string
    )

    imitation_prompt = ChatPromptTemplate.from_messages(
        [
            imitation_system_starting_prompt,
            IHQP_1,
            IAAP_1,
            IHQP_2,
            IAAP_2,
            IHQP_3,
            IAAP_3,
            IHQP_4,
            IAAP_4,
            IHQP_5,
            IAAP_5,
            IHQP_6,
            IAAP_6,
            IHQP_7,
            IAAP_7,
            IHQP_8,
            IAAP_8,
            IHQP_9,
            IAAP_9,
            IHQP_10,
            IAAP_10,
            imitation_human_instruction_prompt,
        ],
    )

    json_schema_2 = {
        "name": person_name,
        "title": "RoleGPT Prompting",
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

    imitation_chain = create_structured_output_chain(output_schema=json_schema_2, llm=llm, prompt=imitation_prompt,
                                                     output_key="Answers")

    result = imitation_chain.run({"role_name": person_name, "role_description": person_description})
    logger.info(result)
    # print("Result: ", result)

    result = {
        "Baseline": "RoleGPT",
        "model": llm.model_name,
        "Answers": result,
    }

    with open(result_file, 'w', encoding='utf-8') as f:
        result = json.dumps(result, ensure_ascii=False, indent=4)
        f.write(result)

if __name__ == "__main__":
    # Now generate the two results, then combine them into one result file
    
    # GPT3
    PromptModel(question_file, data_file, result_file[0], llm[0], write_gpt3)
    
    # GPT4
    PromptModel(question_file, data_file, result_file[1], llm[1], write_gpt4)
        