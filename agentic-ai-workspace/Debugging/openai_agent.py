from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
import os

from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

#print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
#print("LANGCHAIN_API_KEY:", os.getenv("LANGCHAIN_API_KEY"))



class State(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages] 
    
model = ChatOpenAI(temperature=0)

def make_default_graph():
    workflow = StateGraph(State)

    def call_model(state):       
        return {"messages": [model.invoke(state["messages"])]}
    
    workflow.add_node("agent", call_model)
    workflow.add_edge("agent", END)
    workflow.add_edge(START, "agent")
    
    agent = workflow.compile()
    return agent

def make_alternative_graph():
    
    @tool
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b
    
    tool_node = ToolNode([add_numbers])
    model_with_tools = model.bind_tools([add_numbers])
    
    def call_model(state):
        return {"messages": [model_with_tools.invoke(state["messages"])]}
    
    def should_continue(state):
        if state["messages"][-1].tool_calls:
            return "tools"
        else:
            return END
    
    workflow = StateGraph(State)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_edge("tools","agent")
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent",should_continue)
    
    agent = workflow.compile()
    return agent
   

#agent = make_default_graph()
agent = make_alternative_graph()


#### to RUN
### To Debugging in langsmith studio


#### langgraph dev


