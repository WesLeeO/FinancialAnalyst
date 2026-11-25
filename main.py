"""Run the stock analyst from the terminal.

    python main.py --ticker NVDA
    python main.py --ticker tsla --question "Is it overvalued after the last rally?"
"""

import argparse
import time

from langchain_core.messages import HumanMessage

from app.graph import build_graph

DISCLAIMER = "Educational analysis based on public data, not personal financial advice."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI stock analyst (LangGraph + local LLM)")
    parser.add_argument("--ticker", required=True, help="stock ticker, e.g. NVDA")
    parser.add_argument("--question", default="Is this stock a reasonable buy right now?",
                        help="your question in quotes (optional)")
    parser.add_argument("--max-steps", type=int, default=20,
                        help="safety cap on graph steps (default 20)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ticker = args.ticker.strip().upper()
    print(f"\nAnalysing {ticker}: {args.question}\n")

    graph = build_graph()
    start = time.time()
    report = None

    # stream_mode="updates" yields {node_name: what_that_node_returned} after every step
    for step in graph.stream(
        {"messages": [HumanMessage(args.question)], "ticker": ticker,
         "question": args.question, "report": None},
        {"recursion_limit": args.max_steps},
        stream_mode="updates",
    ):  
        # update only continas what a node returned, not the full state. The graph merges it in.
        for node, update in step.items():
            elapsed = f"[{time.time() - start:5.1f}s]"
            if node == "analyst":
                msg = update["messages"][0]
                if msg.tool_calls:
                    for call in msg.tool_calls:
                        print(f"{elapsed} analyst -> calls {call['name']}({call['args']})")
                else:
                    print(f"{elapsed} analyst -> done researching, notes:\n")
                    print("   " + msg.content.strip().replace("\n", "\n   ") + "\n")
            elif node == "tools":
                for msg in update["messages"]:
                    print(f"{elapsed} tools   -> {msg.name} returned {len(msg.content)} chars")
            elif node == "report":
                report = update["report"]
                print(f"{elapsed} report  -> structured report ready")

    if report is None:
        print("\nNo report produced.")
        return

    report.disclaimer = DISCLAIMER  # don't let the model rewrite the disclaimer
    print("\n" + report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
