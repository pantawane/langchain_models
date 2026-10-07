from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()


# Define the model
model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# Define the output structure
class Person(BaseModel):
     name: str = Field(description='Name of the person')
     age: int = Field(gt=18, description='Age of the person')
     city: str = Field(description='Name of the city the person belongs to')

# Make the model return structured output
parser = PydanticOutputParser(pydantic_object=Person)


# Prompt
template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

final_result = chain.invoke({'place':'USA'})


# prompt = template.invoke({'place':'Indian'})

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

print(final_result)