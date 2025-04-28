import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import Optional


from openai import OpenAI
import json

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# from langgraph.store.memory import InMemoryStore

# Add this checkpoint saver class:

import json

import pandas as pd
from io import StringIO

# memory = InMemoryStore()


# Load environment variables from .env file
load_dotenv(override=True)

openai_key = os.getenv("OPENAI_API_KEY")

tavily = os.getenv("TAVILY_API_KEY")

llm_name = "gpt-4o-mini"
model = ChatOpenAI(api_key=openai_key, model=llm_name)

from tavily import TavilyClient

tavily = TavilyClient(api_key=tavily)


from typing import TypedDict, List
from pydantic import BaseModel


class AgentState(TypedDict):
    task: str
    competitors: List[str]
    csv_file: str
    financial_data: str
    analysis: str
    competitor_data: str
    comparison: str
    feedback: str
    report: str
    content: List[str]
    revision_number: int
    max_revisions: int


class Queries(BaseModel):
    queries: List[str]


# Define the prompts for each node - IMPROVE AS NEEDED
GATHER_FINANCIALS_PROMPT = """You are an expert financial analyst. Gather the financial data for the given company. Provide detailed financial data."""
ANALYZE_DATA_PROMPT = """You are an expert financial analyst. Analyze the provided financial data and provide detailed insights and analysis."""
RESEARCH_COMPETITORS_PROMPT = """You are a researcher tasked with providing information about similar companies for performance comparison. Generate a list of search queries to gather relevant information. Only generate 3 queries max."""
COMPETE_PERFORMANCE_PROMPT = """You are an expert financial analyst. Compare the financial performance of the given company with its competitors based on the provided data:
**MAKE SURE TO INCLUDE THE NAMES OF THE COMPETITORS IN THE COMPARISON.**
{content}"""
FEEDBACK_PROMPT = """You are a reviewer. Provide detailed feedback and critique for the provided financial comparison report. Include any additional information or revisions needed."""
WRITE_REPORT_PROMPT = """You are a financial report writer. Write a comprehensive financial report based on the analysis, competitor research, comparison, and feedback provided."""
RESEARCH_CRITIQUE_PROMPT = """You are a researcher tasked with providing information to address the provided critique. Generate a list of search queries to gather relevant information. Only generate 3 queries max."""

# Bước 1: Dựa trên các số liệu trong file csv cung cấp, Node sẽ viết ra một báo cáo tài chính chi tiết
def gather_financials_node(state: AgentState):
    # Read the CSV file into a pandas DataFrame
    csv_file = state["csv_file"]
    df = pd.read_csv(StringIO(csv_file))

    # Convert the DataFrame to a string
    financial_data_str = df.to_string(index=False)

    # Combine the financial data string with the task
    combined_content = (
        f"{state['task']}\n\nHere is the financial data:\n\n{financial_data_str}"
    )

    messages = [
        SystemMessage(content=GATHER_FINANCIALS_PROMPT),
        HumanMessage(content=combined_content),
    ]

    response = model.invoke(messages)
    return {"financial_data": response.content}

# Bước 2: Dựa vào báo cáo tài chính chi tiết đã viết ở trên, thực hiện phân tích dữ liệu tài chính.
def analyze_data_node(state: AgentState):
    messages = [
        SystemMessage(content=ANALYZE_DATA_PROMPT),
        HumanMessage(content=state["financial_data"]),
    ]
    response = model.invoke(messages)
    return {"analysis": response.content}

# Bước 3: Dựa vào danh sách các đối thủ, tạo ra các truy vấn tìm kiếm thông tin về đối thủ. Mỗi đối thủ gồm 3 truy vấn tìm kiếm => dùng công cụ Tavily để tìm kiếm thông tin => Lấy ra 2 mẫu tìm kiếm cho 1 truy vấn.
def research_competitors_node(state: AgentState):
    content = state["content"] or []
    for competitor in state["competitors"]:
        queries = model.with_structured_output(Queries).invoke(
            [
                SystemMessage(content=RESEARCH_COMPETITORS_PROMPT),
                HumanMessage(content=competitor),
            ]
        )
        for q in queries.queries:
            response = tavily.search(query=q, max_results=2)
            for r in response["results"]:
                content.append(r["content"])
    return {"content": content}

# Bước 4: Dựa vào thông tin phân tích báo cáo tài chính của công ty ở bước 2 và dữ liệu thông tin của đối thủ ở bước 3. Thực hiện so sánh
def compare_performance_node(state: AgentState):
    content = "\n\n".join(state["content"] or [])
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is the financial analysis:\n\n{state['analysis']}"
    )
    messages = [
        SystemMessage(content=COMPETE_PERFORMANCE_PROMPT.format(content=content)),
        user_message,
    ]
    response = model.invoke(messages)
    return {
        "comparison": response.content,
        "revision_number": state.get("revision_number", 1) + 1,
    }

# Bước 5: Dựa trên nội dung so sánh đã tạo ra ở bước trên, đưa ra lời khuyến nghị các mục cần được cải thiện hoặc yêu cầu các sửa đổi cần thiết
def collect_feedback_node(state: AgentState):
    messages = [
        SystemMessage(content=FEEDBACK_PROMPT),
        HumanMessage(content=state["comparison"]),
    ]
    response = model.invoke(messages)
    return {"feedback": response.content}

# Bước 6: Dựa trên các feedbacks ở trên, thực hiện tạo các truy vấn để thu thập thông tin cho các feedback, để cải thiện báo cáo => Mỗi truy vấn dùng Tavily để lấy ra 2 mẫu thông tin.
def research_critique_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke(
        [
            SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
            HumanMessage(content=state["feedback"]),
        ]
    )
    content = state["content"] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response["results"]:
            content.append(r["content"])
    return {"content": content}

# Bước 7: Dựa trên các thông tin tìm kiếm bổ sung và bài báo cáo so sánh trước đó. Thực hiện viết lại báo cáo tài chính.
def write_report_node(state: AgentState):
    messages = [
        SystemMessage(content=WRITE_REPORT_PROMPT),
        HumanMessage(content=state["comparison"]),
    ]
    response = model.invoke(messages)
    return {"report": response.content}

# Bước 8: So sánh số lần review, không vượt quá số lần cho phép.
def should_continue(state):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "collect_feedback"


builder = StateGraph(AgentState)

builder.add_node("gather_financials", gather_financials_node)
builder.add_node("analyze_data", analyze_data_node)
builder.add_node("research_competitors", research_competitors_node)
builder.add_node("compare_performance", compare_performance_node)
builder.add_node("collect_feedback", collect_feedback_node)
builder.add_node("research_critique", research_critique_node)

builder.add_node("write_report", write_report_node)


builder.set_entry_point("gather_financials")


builder.add_conditional_edges(
    "compare_performance",
    should_continue,
    {END: END, "collect_feedback": "collect_feedback"},
)

builder.add_edge("gather_financials", "analyze_data")
builder.add_edge("analyze_data", "research_competitors")
# Bắt đầu vòng loop
builder.add_edge("research_competitors", "compare_performance")
builder.add_edge("collect_feedback", "research_critique")
builder.add_edge("research_critique", "compare_performance")
# Kết thúc vòng loop nếu thỏa điều kiện should_continue
builder.add_edge("compare_performance", "write_report")

# graph = builder.compile(checkpointer=memory)
graph = builder.compile()


# ==== For Console Testing ====
# def read_csv_file(file_path):
#     with open(file_path, "r") as file:
#         print("Reading CSV file...")
#         return file.read()


# if __name__ == "__main__":
#     task = "Analyze the financial performance of our (MegaAICo) company compared to competitors"
#     competitors = ["Microsoft", "Nvidia", "Google"]
#     csv_file_path = "./data/financials.csv"

#     if not os.path.exists(csv_file_path):
#         print(f"CSV file not found at {csv_file_path}")
#     else:
#         print("Starting the conversation...")
#         csv_data = read_csv_file(csv_file_path)

#         initial_state = {
#             "task": task,
#             "competitors": competitors,
#             "csv_file": csv_data,
#             "max_revisions": 2,
#             "revision_number": 1,
#             # Add these initialized keys
#             "content": [],
#             "financial_data": "",
#             "analysis": "",
#             "competitor_data": "",
#             "comparison": "",
#             "feedback": "",
#             "report": ""
#         }

#         for s in graph.stream(initial_state):
#             print(s)
# === End Console Testing ===

# ==== Streamlit UI ====
import streamlit as st


def main():
    st.title("Financial Performance Reporting Agent")

    task = st.text_input(
        "Enter the task:",
        "Analyze the financial performance of our company (MyAICo.AI) compared to competitors",
    )
    competitors = st.text_area("Enter competitor names (one per line):").split("\n")
    max_revisions = st.number_input("Max Revisions", min_value=1, value=2)
    uploaded_file = st.file_uploader(
        "Upload a CSV file with the company's financial data", type=["csv"]
    )

    if st.button("Start Analysis") and uploaded_file is not None:
        # Read the uploaded CSV file
        csv_data = uploaded_file.getvalue().decode("utf-8")

        # Initialize state with all required keys
        initial_state = {
            "task": task,
            "competitors": [comp.strip() for comp in competitors if comp.strip()],
            "csv_file": csv_data,
            "max_revisions": max_revisions,
            "revision_number": 1,
            # Add these initialized keys
            "content": [],
            "financial_data": "",
            "analysis": "",
            "competitor_data": "",
            "comparison": "",
            "feedback": "",
            "report": "",
        }

        state_placeholder = st.empty()

        thread = {"configurable": {"thread_id": "1"}}

        try:
            state = initial_state.copy()  # Start with the initial state
            for s in graph.stream(initial_state):
                # Update the state with the output of the current node
                node_name = list(s.keys())[0]  # Get the node name (e.g., "write_report")
                state.update(s[node_name])     # Merge the node's output into the state
                with state_placeholder.container():
                    st.write("Current State:", s)
            
            # After the loop, state contains the final merged state
            if "report" in state:
                st.subheader("Final Report")
                st.markdown(state["report"])
            else:
                st.error("No report generated. Please check the agent execution.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()

# ==== End Streamlit UI ====