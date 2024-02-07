from customllm import CustomLLM

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

question = "Who won the FIFA World Cup in the year 1994? "

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm = CustomLLM()
llm_chain = LLMChain(prompt=prompt, llm=llm)

print(llm_chain.invoke(question))