from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# Define the model
model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me 5 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# prompt = template.format(
#     topic="Artificial Intelligence"
# )

chain = template | model | parser
result = chain.invoke({'topic': 'Artificial Intelligence'})

print(result)
# result = model.invoke(prompt)
# final_result =parser.parse(result.content)

# print(final_result)
# print(type(final_result))