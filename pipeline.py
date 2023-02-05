from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, initialize_agent
from langchain import SerpAPIWrapper

# define large laguage model
llm = OpenAI(temperature=0)

# get user input claim
claim = input("Enter a claim: ")

# insert into template
prompt = PromptTemplate(
    input_variables=["claim"],
    template="Is it true that {claim}?",
)

# load serpapi access
search = SerpAPIWrapper()
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run
    )
]

# get self-ask with search
agent = initialize_agent(
    tools, 
    llm, 
    agent="self-ask-with-search", 
    verbose=True
)
agent.run(prompt.format(claim=claim))