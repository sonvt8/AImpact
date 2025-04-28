import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from openai import OpenAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END


# Load environment variables from .env file
load_dotenv(override=True)

openai_key = os.getenv("OPENAI_API_KEY")


# llm_name = "gpt-4o-mini"

# model = ChatOpenAI(api_key=openai_key, model=llm_name)
model = ChatOllama(
        model='llama3.2',
        temperature=0.3,
    )


# STEP 1: Build a Basic Chatbot
from langgraph.graph.message import add_messages


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


def bot(state: State):
    # print(state.items())
    print(state["messages"])
    return {"messages": [model.invoke(state["messages"])]}


graph_builder = StateGraph(State)

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("bot", bot)


# STEP 3: Add an entry point to the graph
graph_builder.set_entry_point("bot")

# STEP 4: and end point to the graph
graph_builder.set_finish_point("bot")


# STEP 5: Compile the graph
graph = graph_builder.compile()

state = {"messages": []}

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    # Append the new user message to the existing state
    state["messages"].append(("user", user_input))
    print(f"State: {state["messages"]}")
    # Stream the graph with the updated state
    for event in graph.stream(state):
        print(f"Event: {event}")
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)
            
# User: Hey, How've you been lately?
# State: [('user', "Hey, How've you been lately?")]
# [HumanMessage(content="Hey, How've you been lately?", additional_kwargs={}, response_metadata={}, id='68e432ea-9608-43cd-8bd4-6051c3d34737')]
# Event: {'bot': {'messages': [AIMessage(content="I'm just a language model, I don't have emotions or personal experiences like humans do. However, I'm functioning properly and ready to assist you with any questions or tasks you may have! How can Ike humans do. However, I'm functioning properly and ready to assist you with any questions or tasks you may have! How can I help you today?", additional_kwargs={}, response_metadata={'model': 'llama3.2', 'created_at': '2025-04-21T08:11:42.6247768Z', 'done': True, 'done_reason': 'stop', 'total_duration': 8068313600, 'load_duration': 2701258100, 'prompt_eval_count': 33, 'prompt_eval_duration': 1289677800, 'eval_count': 47, 'eval_duration': 4075664700, 'message': Message(role='assistant', content='', images=None, tool_calls=None), 'model_name': 'llama3.2'}, id='run-b766d414-3bf5-46a6-9a7d-b735af89cadb-0', usage_metadata={'input_tokens': 33, 'output_tokens': 47, 'total_tokens': 80})]}}
# Assistant: I'm just a language model, I don't have emotions or personal experiences like humans do. However, I'm functioning properly and ready to assist you with any questions or tasks you may have! How can I help you today?

#############################################

# User: Fine, Do you know me?
# State: [('user', "Hey, How've you been lately?"), ('user', 'Fine, Do you know me?')]
# [HumanMessage(content="Hey, How've you been lately?", additional_kwargs={}, response_metadata={}, id='9166504b-6681-44f6-a91e-b762b819bfd2'), HumanMessage(content='Fine, Do you know me?', additional_kwargs={}, response_metadata={}, id='425cd97b-c76d-4cce-9aea-5666fe4b5268')]
# Event: {'bot': {'messages': [AIMessage(content="I'm just a language model, I don't have personal experiences or emotions like humans do. However, I'm functioning properly and ready to assist with any questions or topics you'd like to discuss.\n\nAs for knowing you, we haven't had a previous conversation, so I don't have any prior knowledge about you. But feel free to share something about yourself if you'd like!", additional_kwargs={}, response_metadata={'model': 'llama3.2', 'created_at': '2025-04-21T08:13:05.5059614Z', 'done': True, 'done_reason': 'stop', 'total_duration': 7346527700, 'load_duration': 72459800, 'prompt_eval_count': 40, 'prompt_eval_duration': 548571700, 'eval_count': 78, 'eval_duration': 6722490100, 'message': Message(role='assistant', content='', images=None, tool_calls=None), 'model_name': 'llama3.2'}, id='run-ad7d798d-9526-473e-8df9-2d419b76b17e-0', usage_metadata={'input_tokens': 40, 'output_tokens': 78, 'total_tokens': 118})]}}
# Assistant: I'm just a language model, I don't have personal experiences or emotions like humans do. However, I'm functioning properly and ready to assist with any questions or topics you'd like to discuss.

# As for knowing you, we haven't had a previous conversation, so I don't have any prior knowledge about you. But feel free to share something about yourself if you'd like!

#############################################

# User: My name is Tommy and I am an Engineer
# State: [('user', "Hey, How've you been lately?"), ('user', 'Fine, Do you know me?'), ('user', 'My name is Tommy and I am an Engineer')]
# [HumanMessage(content="Hey, How've you been lately?", additional_kwargs={}, response_metadata={}, id='428d1d4a-d355-4d3e-a4bb-b7f5144ce2d2'), HumanMessage(content='Fine, Do you know me?', additional_kwargs={}, response_metadata={}, id='52c75e58-74e3-411b-838b-78704e080992'), HumanMessage(content='My name is Tommy and I am an Engineer', additional_kwargs={}, response_metadata={}, id='d81570b2-9901-4c95-be10-d7e792e691c1')]
# Event: {'bot': {'messages': [AIMessage(content="Nice to meet you, Tommy! I'm just a language model, I don't have personal experiences or emotions like humans do, but I'm always happy to chat with someone new!\n\nAs for knowing you, I don't have any prior knowledge about your identity or background. You're the first time we've interacted, so let's start fresh! Being an engineer is a great profession - what kind of engineering do you specialize in?", additional_kwargs={}, response_metadata={'model': 'llama3.2', 'created_at': '2025-04-21T08:13:51.9920873Z', 'done': True, 'done_reason': 'stop', 'total_duration': 8133599200, 'load_duration': 71074500, 'prompt_eval_count': 49, 'prompt_eval_duration': 593007400, 'eval_count': 89, 'eval_duration': 7466833300, 'message': Message(role='assistant', content='', images=None, tool_calls=None), 'model_name': 'llama3.2'}, id='run-4c60d488-60ee-4dfc-9a6c-f7a628485b76-0', usage_metadata={'input_tokens': 49, 'output_tokens': 89, 'total_tokens': 138})]}}
# Assistant: Nice to meet you, Tommy! I'm just a language model, I don't have personal experiences or emotions like humans do, but I'm always happy to chat with someone new!

# As for knowing you, I don't have any prior knowledge about your identity or background. You're the first time we've interacted, so let's start fresh! Being an engineer is a great profession - what kind of engineering do you specialize in? 

#############################################

# User: Sound nice, so what is my name?
# State: [('user', "Hey, How've you been lately?"), ('user', 'Fine, Do you know me?'), ('user', 'My name is Tommy and I am an Engineer'), ('user', 'Okie, anyway, What is my job?'), ('user', 'Sound nice, so what is my name?')]
# [HumanMessage(content="Hey, How've you been lately?", additional_kwargs={}, response_metadata={}, id='71423102-89b9-41e7-8111-afe90e7526fe'), HumanMessage(content='Fine, Do you know me?', additional_kwargs={}, response_metadata={}, id='a4f90fb6-18d5-43fc-9773-19186bb73a52'), HumanMessage(content='My name is Tommy and I am an Engineer', additional_kwargs={}, response_metadata={}, id='2973e601-195a-436e-869e-a0329d5011bb'), HumanMessage(content='Okie, anyway, What is my job?', additional_kwargs={}, response_metadata={}, id='fd86a860-6e2a-400c-98c2-350b5e558542'), HumanMessage(content='Sound nice, so what is my name?', additional_kwargs={}, response_metadata={}, id='2e07ea4a-c43a-4c0a-8d74-d1bc82dbf449')]
# Event: {'bot': {'messages': [AIMessage(content="Nice to meet you, Tommy! It sounds like we're having a fun conversation.\n\nAs for your job, it seems like you're an engineer. That's great! Engineers are always solving problems and coming up with creative solutions. What kind of engineering do you specialize in (e.g., mechanical, electrical, civil)?", additional_kwargs={}, response_metadata={'model': 'llama3.2', 'created_at': '2025-04-21T08:17:17.4283588Z', 'done': True, 'done_reason': 'stop', 'total_duration': 6187032800, 'load_duration': 75580400, 'prompt_eval_count': 69, 'prompt_eval_duration': 605670500, 'eval_count': 65, 'eval_duration': 5503199500, 'message': Message(role='assistant', content='', images=None, tool_calls=None), 'model_name': 'llama3.2'}, id='run-9330d1a0-0b42-4054-9eaf-d2e061f75508-0', usage_metadata={'input_tokens': 69, 'output_tokens': 65, 'total_tokens': 134})]}}
# Assistant: Nice to meet you, Tommy! It sounds like we're having a fun conversation.

# As for your job, it seems like you're an engineer. That's great! Engineers are always solving problems and coming up with creative solutions. What kind of engineering do you specialize in (e.g., mechanical, electrical, civil)?

#############################################

# User: haha, I just want to make sure you know who I am and you have a context of our conversation
# State: [('user', "Hey, How've you been lately?"), ('user', 'Fine, Do you know me?'), ('user', 'My name is Tommy and I am an Engineer'), ('user', 'Okie, anyway, What is my job?'), ('user', 'Sound nice, so what is my name?'), ('user', 'haha, I just want to make sure you know who I am and you have a context of our conversation')]
# [HumanMessage(content="Hey, How've you been lately?", additional_kwargs={}, response_metadata={}, id='af3c4fdc-094e-4be1-b8d9-b096e9d24659'), HumanMessage(content='Fine, Do you know me?', additional_kwargs={}, response_metadata={}, id='c94946de-aa12-44b6-8dd1-94634efe986c'), HumanMessage(content='My name is Tommy and I am an Engineer', additional_kwargs={}, response_metadata={}, id='988728cb-ab32-476a-8610-66f5d21063e0'), HumanMessage(content='Okie, anyway, What is my job?', additional_kwargs={}, response_metadata={}, id='3ba7bc04-484f-40a8-8e6c-f18e9d2af933'), HumanMessage(content='Sound nice, so what is my name?', additional_kwargs={}, response_metadata={}, id='fbc10688-a01a-401f-bf32-75029d4c4e09'), HumanMessage(content='haha, I just want to make sure you know who I am and you have a context of our conversation', additional_kwargs={}, response_metadata={}, id='964d2e67-70f6-4014-80c3-afde5a374694')]
# Event: {'bot': {'messages': [AIMessage(content="Nice to meet you, Tommy! I'm glad you're excited about our conversation.\n\nJust to confirm, based on your introduction, here's the context:\n\n* You're an Engineer.\n* Your name is Tommy.\n\nI'll keep this in mind as we chat. What would you like to talk about, Tommy?", additional_kwargs={}, response_metadata={'model': 'llama3.2', 'created_at': '2025-04-21T08:21:52.6477628Z', 'done': True, 'done_reason': 'stop', 'total_duration': 6625868500, 'load_duration': 73821800, 'prompt_eval_count': 91, 'prompt_eval_duration': 1155229400, 'eval_count': 64, 'eval_duration': 5392608200, 'message': Message(role='assistant', content='', images=None, tool_calls=None), 'model_name': 'llama3.2'}, id='run-f4762e0b-2f07-436a-9855-318b222ec7c1-0', usage_metadata={'input_tokens': 91, 'output_tokens': 64, 'total_tokens': 155})]}}
# Assistant: Nice to meet you, Tommy! I'm glad you're excited about our conversation.

# Just to confirm, based on your introduction, here's the context:

# * You're an Engineer.
# * Your name is Tommy.