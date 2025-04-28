import os
import json

from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables from .env file
load_dotenv(override=True)

model = ChatOllama(
        model='llama3.2',
        temperature=0.3,
    )

tavily = os.getenv("TAVILY_API_KEY")

tool = TavilySearchResults(max_results=2)
tools = [tool]

memory = MemorySaver()

config = {
    "configurable": {"thread_id": 1}
}

# STEP 1: Build a Basic Chatbot
from langgraph.graph.message import add_messages


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


def bot(state: State):
    # print(state.items())
    print(f"Bot call:{state["messages"]}")
    return {"messages": [model.invoke(state["messages"])]}

graph_builder = StateGraph(State)

# instantiate the ToolNode with the tools
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)  # Add the node to the graph

# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "__end__" if it is fine directly responding. This conditional routing defines the main agent loop.

graph_builder.add_conditional_edges(
    "bot",
    tools_condition,
)

graph_builder.add_node("bot", bot)

graph_builder.set_entry_point("bot")

graph = graph_builder.compile(
    checkpointer=memory
)

user_input = "Hi there! My name is Tommy. and I am an Engineer"

events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)

for event in events:
    print(f"Event: {event}")
    print("## Pretty printed event:\n")
    event["messages"][-1].pretty_print()
    
user_input = "do you remember my name, and what is my job?"

# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)

for event in events:
    print(f"Event: {event}")
    print("## Pretty printed event:\n")
    event["messages"][-1].pretty_print()


snapshot = graph.get_state(config)
print(snapshot)

# from langchain_core.messages import BaseMessage

# while True:
#     user_input = input("User: ")
#     if user_input.lower() in ["quit", "exit", "q"]:
#         print("Goodbye!")
#         break
#     for event in graph.stream({"messages": [("user", user_input)]}):
#         for value in event.values():
#             if isinstance(value["messages"][-1], BaseMessage):
#                 print("Assistant:", value["messages"][-1].content)

###################################################

# Event: {'messages': [HumanMessage(content='Hi there! My name is Tommy. and I am an Engineer', additional_kwargs={}, response_metadata={}, id='444dbfa5-6409-44ff-b5eb-3123f4326418')]}
# ## Pretty printed event:

# ================================ Human Message =================================

# Hi there! My name is Tommy. and I am an Engineer
# Bot call:[HumanMessage(content='Hi there! My name is Tommy. and I am an Engineer', additional_kwargs={}, response_metadata={}, id='444dbfa5-6409-44ff-b5eb-3123f4326418')]

###################################################

# Event: {'messages': [HumanMessage(content='Hi there! My name is Tommy. and I am an Engineer', additional_kwargs={}, response_metadata={}, id='444dbfa5-6409-44ff-b5eb-3123f4326418'), AIMessage(content="Hello Tommy! Nice to meet you! As an engineer, I'm sure you're always looking for new ways to solve problems and innovate. What area of engineering do you specialize in? Are you working on a specific project or have any exciting plans coming up?", additional_kwargs={}, response_metadata={'model': 'llama3.2', 'created_at': '2025-04-22T03:37:29.4394223Z', 'done': True, 'done_reason': 'stop', 'total_duration': 5082173900, 'load_duration': 77309600, 'prompt_eval_count': 38, 'prompt_eval_duration': 159982300, 'eval_count': 53, 'eval_duration': 4842002300, 'message': Message(role='assistant', content='', images=None, tool_calls=None), 'model_name': 'llama3.2'}, id='run-a3b5559c-ec88-4f91-a479-8fe00fd3f68e-0', usage_metadata={'input_tokens': 38, 'output_tokens': 53, 'total_tokens': 91})]}
# ## Pretty printed event:

# ================================== Ai Message ==================================

# Hello Tommy! Nice to meet you! As an engineer, I'm sure you're always looking for new ways to solve problems and innovate. What area of engineering do you specialize in? Are you working on a specific project or have any exciting plans coming up?

###################################################

# Event: {'messages': [HumanMessage(content='Hi there! My name is Tommy. and I am an Engineer', additional_kwargs={}, response_metadata={}, id='444dbfa5-6409-44ff-b5eb-3123f4326418'), AIMessage(content="Hello Tommy! Nice to meet you! As an engineer, I'm sure you're always looking for new ways to solve problems and innovate. What area of engineering do you specialize in? Are you working on a specific project or have any exciting plans coming up?", additional_kwargs={}, response_metadata={'model': 'llama3.2', 'created_at': '2025-04-22T03:37:29.4394223Z', 'done': True, 'done_reason': 'stop', 'total_duration': 5082173900, 'load_duration': 77309600, 'prompt_eval_count': 38, 'prompt_eval_duration': 159982300, 'eval_count': 53, 'eval_duration': 4842002300, 'message': {'role': 'assistant', 'content': '', 'images': None, 'tool_calls': None}, 'model_name': 'llama3.2'}, id='run-a3b5559c-ec88-4f91-a479-8fe00fd3f68e-0', usage_metadata={'input_tokens': 38, 'output_tokens': 53, 'total_tokens': 91}), HumanMessage(content='do you remember my name, and what is my job?', additional_kwargs={}, response_metadata={}, id='cc2f1a0c-6d50-409f-aed6-d06e2fb5e750')]}
# ## Pretty printed event:

# ================================ Human Message =================================

# do you remember my name, and what is my job?
# Bot call:[HumanMessage(content='Hi there! My name is Tommy. and I am an Engineer', additional_kwargs={}, response_metadata={}, id='444dbfa5-6409-44ff-b5eb-3123f4326418'), AIMessage(content="Hello Tommy! Nice to meet you! As an engineer, I'm sure you're always looking for new ways to solve problems and innovate. What area of engineering do you specialize in? Are you working on a specific project or have any exciting plans coming up?", additional_kwargs={}, response_metadata={'model': 'llama3.2', 'created_at': '2025-04-22T03:37:29.4394223Z', 'done': True, 'done_reason': 'stop', 'total_duration': 5082173900, 'load_duration': 77309600, 'prompt_eval_count': 38, 'prompt_eval_duration': 159982300, 'eval_count': 53, 'eval_duration': 4842002300, 'message': {'role': 'assistant', 'content': '', 'images': None, 'tool_calls': None}, 'model_name': 'llama3.2'}, id='run-a3b5559c-ec88-4f91-a479-8fe00fd3f68e-0', usage_metadata={'input_tokens': 38, 'output_tokens': 53, 'total_tokens': 91}), HumanMessage(content='do you remember my name, and what is my job?', additional_kwargs={}, response_metadata={}, id='cc2f1a0c-6d50-409f-aed6-d06e2fb5e750')]

###################################################

# Event: {'messages': [HumanMessage(content='Hi there! My name is Tommy. and I am an Engineer', additional_kwargs={}, response_metadata={}, id='444dbfa5-6409-44ff-b5eb-3123f4326418'), AIMessage(content="Hello Tommy! Nice to meet you! As an engineer, I'm sure you're always looking for new ways to solve problems and innovate. What area of engineering do you specialize in? Are you working on a specific project or have any exciting plans coming up?", additional_kwargs={}, response_metadata={'model': 'llama3.2', 'created_at': '2025-04-22T03:37:29.4394223Z', 'done': True, 'done_reason': 'stop', 'total_duration': 5082173900, 'load_duration': 77309600, 'prompt_eval_count': 38, 'prompt_eval_duration': 159982300, 'eval_count': 53, 'eval_duration': 4842002300, 'message': {'role': 'assistant', 'content': '', 'images': None, 'tool_calls': None}, 'model_name': 'llama3.2'}, id='run-a3b5559c-ec88-4f91-a479-8fe00fd3f68e-0', usage_metadata={'input_tokens': 38, 'output_tokens': 53, 'total_tokens': 91}), HumanMessage(content='do you remember my name, and what is my job?', additional_kwargs={}, response_metadata={}, id='cc2f1a0c-6d50-409f-aed6-d06e2fb5e750'), AIMessage(content="I remember that your name is Tommy, and you're an Engineer! I'm glad I could recall that correctly earlier. Is there anything else you'd like to chat about related to engineering or something else entirely?", additional_kwargs={}, response_metadata={'model': 'llama3.2', 'created_at': '2025-04-22T03:37:34.5785417Z', 'done': True, 'done_reason': 'stop', 'total_duration': 5133613500, 'load_duration': 67312800, 'prompt_eval_count': 112, 'prompt_eval_duration': 901352600, 'eval_count': 43, 'eval_duration': 4163061200, 'message': Message(role='assistant', content='', images=None, tool_calls=None), 'model_name': 'llama3.2'}, id='run-51b95d4f-8c16-419c-8e3c-265284e0abac-0', usage_metadata={'input_tokens': 112, 'output_tokens': 43, 'total_tokens': 155})]}
# ## Pretty printed event:

# ================================== Ai Message ==================================

# I remember that your name is Tommy, and you're an Engineer! I'm glad I could recall that correctly earlier. Is there anything else you'd like to chat about related to engineering or something else entirely?

###################################################
    
# StateSnapshot(values={'messages': [HumanMessage(content='Hi there! My name is Tommy. and I am an Engineer', additional_kwargs={}, response_metadata={}, id='444dbfa5-6409-44ff-b5eb-3123f4326418'), AIMessage(content="Hello Tommy! Nice to meet you! As an engineer, I'm sure you're always looking for new ways to solve problems and innovate. What area of engineering do you specialize in? Are you working on a specific project or have any exciting plans coming up?", additional_kwargs={}, response_metadata={'model': 'llama3.2', 'created_at': '2025-04-22T03:37:29.4394223Z', 'done': True, 'done_reason': 'stop', 'total_duration': 5082173900, 'load_duration': 77309600, 'prompt_eval_count': 38, 'prompt_eval_duration': 159982300, 'eval_count': 53, 'eval_duration': 4842002300, 'message': {'role': 'assistant', 'content': '', 'images': None, 'tool_calls': None}, 'model_name': 'llama3.2'}, id='run-a3b5559c-ec88-4f91-a479-8fe00fd3f68e-0', usage_metadata={'input_tokens': 38, 'output_tokens': 53, 'total_tokens': 91}), HumanMessage(content='do you remember my name, and what is my job?', additional_kwargs={}, response_metadata={}, id='cc2f1a0c-6d50-409f-aed6-d06e2fb5e750'), AIMessage(content="I remember that your name is Tommy, and you're an Engineer! I'm glad I could recall that correctly earlier. Is there anything else you'd like to chat about related to engineering or something else entirely?", additional_kwargs={}, response_metadata={'model': 'llama3.2', 'created_at': '2025-04-22T03:37:34.5785417Z', 'done': True, 'done_reason': 'stop', 'total_duration': 5133613500, 'load_duration': 67312800, 'prompt_eval_count': 112, 'prompt_eval_duration': 901352600, 'eval_count': 43, 'eval_duration': 4163061200, 'message': {'role': 'assistant', 'content': '', 'images': None, 'tool_calls': None}, 'model_name': 'llama3.2'}, id='run-51b95d4f-8c16-419c-8e3c-265284e0abac-0', usage_metadata={'input_tokens': 112, 'output_tokens': 43, 'total_tokens': 155})]}, next=(), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f01f2b2-0fa1-64fb-8004-da2ce1bfc3b4'}}, metadata={'source': 'loop', 'writes': {'bot': {'messages': [AIMessage(content="I remember that your name is Tommy, and you're an Engineer! I'm glad I could recall that correctly earlier. Is there anything else you'd like to chat about related to engineering or something else entirely?", additional_kwargs={}, response_metadata={'model': 'llama3.2', 'created_at': '2025-04-22T03:37:34.5785417Z', 'done': True, 'done_reason': 'stop', 'total_duration': 5133613500, 'load_duration': 67312800, 'prompt_eval_count': 112, 'prompt_eval_duration': 901352600, 'eval_count': 43, 'eval_duration': 4163061200, 'message': {'role': 'assistant', 'content': '', 'images': None, 'tool_calls': None}, 'model_name': 'llama3.2'}, id='run-51b95d4f-8c16-419c-8e3c-265284e0abac-0', usage_metadata={'input_tokens': 112, 'output_tokens': 43, 'total_tokens': 155})]}}, 'step': 4, 'parents': {}, 'thread_id': 1}, created_at='2025-04-22T03:37:34.580030+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f01f2b1-dea2-6ec6-8003-1a6ef33467cf'}}, tasks=())