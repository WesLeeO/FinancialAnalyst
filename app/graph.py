from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


from app.config import get_llm          # you'll write these files next
from app.tools import TOOLS
from app.schemas import AdviceReport
from app.prompts import ANALYST_PROMPT, REPORT_PROMPT

 
# Every node receives the full state as its input: messages, ticker, question and report.
# Every node returns only what it changes, as a partial update. LangGraph merges that update in:
# analyst returns {"messages": [...]}, which gets appended because of add_messages.
# report returns {"report": ...}, which gets overwritten, the default behavior.
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # add_messages = append, don't overwrite
    ticker: str
    question: str
    report: AdviceReport | None


llm = get_llm()
analyst_llm = llm.bind_tools(TOOLS)                  # can request tool calls
report_llm = llm.with_structured_output(AdviceReport)  # must return valid AdviceReport JSON

def analyst(state: AgentState) -> dict:
    system = SystemMessage(ANALYST_PROMPT.format(ticker=state["ticker"]))
    response = analyst_llm.invoke([system] + state["messages"])
    return {"messages": [response]}        # appended thanks to add_messages


def report(state: AgentState) -> dict:
    system = SystemMessage(REPORT_PROMPT.format(ticker=state["ticker"]))
    result = report_llm.invoke([system] + state["messages"])
    return {"report": result}              # an AdviceReport object

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("analyst", analyst)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_node("report", report)

    g.add_edge(START, "analyst")
    g.add_conditional_edges(
        "analyst",
        tools_condition,                    # looks at the analyst's last message: tool_calls or not?
        {"tools": "tools", END: "report"},  # remap "done" to the report node instead of ending
    )
    g.add_edge("tools", "analyst")          # results go back to the analyst to reason over
    g.add_edge("report", END)
    return g.compile()



if __name__ == "__main__":
    graph = build_graph()
    question = "Is NVDA a reasonable buy for a 6-month hold?"
    # give input state to the graph, and a recursion limit to prevent infinite loops
    result = graph.invoke(
        {"messages": [HumanMessage(question)], "ticker": "NVDA",
         "question": question, "report": None},
        {"recursion_limit": 20},   # safety cap so the tool loop can't run forever
    )
    print(result["report"].model_dump_json(indent=2))