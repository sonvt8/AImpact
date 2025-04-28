import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, END
from langchain_community.tools.tavily_search import TavilySearchResults


# Load environment variables from .env file
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
tavily = os.getenv("TAVILY_API_KEY")


llm_name = "gpt-4o-mini"

# client = OpenAI(api_key=openai_key)
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


graph_builder = StateGraph(State)

# create tools
tool = TavilySearchResults(max_results=2)
tools = [tool]
# rest = tool.invoke("What is the capital of France?")
# print(rest)

model_with_tools = model.bind_tools(tools)

# Below, implement a BasicToolNode that checks the most recent
# message in the state and calls tools if the message contains tool_calls
import json
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition


def bot(state: State):
    # print(state.items())
    print(state["messages"])
    return {"messages": [model_with_tools.invoke(state["messages"])]}


# instantiate the ToolNode with the tools
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)  # Add the node to the graph


# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "__end__" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "bot",
    tools_condition,
)

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("bot", bot)


# STEP 3: Add an entry point to the graph
graph_builder.set_entry_point("bot")

# ADD MEMORY NODE
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string(":memory:") as memory:
    
    # STEP 5: Compile the graph
    graph = graph_builder.compile(
        checkpointer=memory,
        interrupt_before=["tools"],
    )
    # MEMORY CODE CONTINUES ===
    # Now we can run the chatbot and see how it behaves
    # PICK A TRHEAD FIRST
    config = {
        "configurable": {"thread_id": 1}
    }  # a thread where the agent will dump its memory to
    user_input = "I'm learning about AI. Could you do some research on it for me?"

    # The config is the **second positional argument** to stream() or invoke()!
    events = graph.stream(
        {"messages": [("user", user_input)]}, config, stream_mode="values"
    )
    for event in events:
        event["messages"][-1].pretty_print()

    # inspect the state
    snapshot = graph.get_state(config)
    next_step = snapshot.next
    # this will show "action", because we've interrupted the flow before the tools node

    print(
        "===>>>", next_step
    )  # this will show "action", because we've interrupted the flow before the tools node


    existing_message = snapshot.values["messages"][-1]
    all_tools = existing_message.tool_calls

    print("tools to be called::", all_tools)

    # Continue the conversation passing None to say continue - all is good
    # `None` will append nothing new to the current state, letting it resume as if it had never been interrupted
    events = graph.stream(None, config, stream_mode="values")
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()
            
# ================================ Human Message =================================

# I'm learning about AI. Could you do some research on it for me?
# [HumanMessage(content="I'm learning about AI. Could you do some research on it for me?", additional_kwargs={}, response_metadata={}, id='9e16b184-43ff-481a-89e9-d3ead4128942')]
# ================================== Ai Message ==================================
# Tool Calls:
#   tavily_search_results_json (ab2aa064-0084-448c-9f52-213b0168faa1)
#  Call ID: ab2aa064-0084-448c-9f52-213b0168faa1
#   Args:
#     query: AI research
# ===>>> ('tools',)
# tools to be called:: [{'name': 'tavily_search_results_json', 'args': {'query': 'AI research'}, 'id': 'ab2aa064-0084-448c-9f52-213b0168faa1', 'type': 'tool_call'}]
# ================================== Ai Message ==================================
# Tool Calls:
#   tavily_search_results_json (ab2aa064-0084-448c-9f52-213b0168faa1)
#  Call ID: ab2aa064-0084-448c-9f52-213b0168faa1
#   Args:
#     query: AI research
# ================================= Tool Message =================================
# Name: tavily_search_results_json

# [{"title": "Artificial Intelligence - AI Research - Caltech Science Exchange", "url": "https://scienceexchange.caltech.edu/topics/artificial-intelligence-research", "content": "What Is AI? Artificial intelligence is transforming scientific research as well as everyday life, from communications to transportation to health care and more.", "score": 0.76596165}, {"title": "AI for research: the ultimate guide to choosing the right tool - Nature", "url": "https://www.nature.com/articles/d41586-025-01069-0", "content": "Daniel Weld, chief scientist at the academic search engine Semantic Scholar, who is based in Seattle, Washington, says that many popular AI platforms have “advanced enormously” in an area called active learning — a method that mimics how a person would approach a research question. Programs such as Google’s Gemini Deep Research and OpenAI’s Deep Research offer the most powerful tools in this regard, and many companies are launching similar products. [...] But even for students who do not have access to something as advanced as CRESt, AI can still function as a helpful colleague. Gemini Deep Research, for example, can generate a “personalized multi-point research plan” among other features, and resources such as Scite and Elicit are billed as research assistants. Users can give these programs a handful of papers or a working hypothesis, for example, and ask for a set of experiments to test the theory. [...] Shafi now says that the arrival of AI has been “a revolution for research”, a sentiment seemingly shared by others. Surveys show that many university students and scientists are using AI in their work, often on a weekly or even daily basis. And whereas many educators and academic institutions initially responded with wariness, academia seems increasingly willing to allow students to use AI, albeit in controlled ways. Although it wouldn’t be impossible to go back to the way he did things before,", "score": 0.72042775}]