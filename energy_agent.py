"""
Lesson 1: An energy-market analyst agent for German electricity data (SMARD),
built as an explicit LangGraph state machine.

What this teaches (the parts companies actually use):
  - Tool calling with typed, well-described tools
  - LangGraph StateGraph: state, nodes, conditional edges
  - Self-correction: SQL errors are returned to the model, not raised
  - Guardrails: read-only connection, SELECT-only, row limit, query timeout
  - Checkpointing: multi-turn conversations via thread_id

Run:
  pip install -r requirements.txt
  python setup_db.py            # downloads SMARD data into energy.db
  export GROQ_API_KEY=...       # free key from console.groq.com
  python energy_agent.py
"""

import os
import re
import sqlite3
import time
from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

DB_PATH = os.getenv("DB_PATH", "energy.db")
MODEL = os.getenv("MODEL", "groq:openai/gpt-oss-120b")
MAX_ROWS = 50
QUERY_TIMEOUT_S = 5.0


# ---------------------------------------------------------------- guardrails
def get_conn() -> sqlite3.Connection:
    """Read-only connection: the database itself refuses writes."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    deadline = time.monotonic() + QUERY_TIMEOUT_S
    # Abort long-running queries (returning non-zero cancels the query).
    conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
    return conn


FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|pragma|vacuum)\b",
    re.IGNORECASE,
)


def _table_names() -> list[str]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    return [r[0] for r in rows]


# --------------------------------------------------------------------- tools
# Tool docstrings are what the model reads to decide when and how to call
# them. Writing them well is a core skill, not an afterthought.

@tool
def list_tables() -> str:
    """List all tables in the database. Call this first to orient yourself."""
    return ", ".join(_table_names())


@tool
def describe_table(table: str) -> str:
    """Show the columns (name, type) and 3 sample rows of one table.
    Use this before writing a query against a table you haven't inspected."""
    if table not in _table_names():  # never interpolate unvalidated input
        return f"Unknown table '{table}'. Call list_tables to see valid names."
    with get_conn() as conn:
        cols = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
        sample = conn.execute(f'SELECT * FROM "{table}" LIMIT 3').fetchall()
    col_desc = "\n".join(f"  {c[1]} ({c[2]})" for c in cols)
    return f"Table {table}:\n{col_desc}\nSample rows:\n" + "\n".join(map(str, sample))


@tool
def run_query(sql: str) -> str:
    """Run a single read-only SQL SELECT (or WITH ... SELECT) query on SQLite.
    Returns at most 50 rows. If the query fails, the error message is
    returned so you can fix the query and try again."""
    stripped = sql.strip().rstrip(";")
    if ";" in stripped:
        return "Error: only one statement is allowed."
    if not re.match(r"^\s*(select|with)\b", stripped, re.IGNORECASE) or FORBIDDEN.search(stripped):
        return "Error: only read-only SELECT queries are allowed."
    try:
        with get_conn() as conn:
            cur = conn.execute(stripped)
            headers = [d[0] for d in cur.description]
            rows = cur.fetchmany(MAX_ROWS + 1)
    except sqlite3.Error as e:
        # Returning the error (instead of raising) lets the agent self-correct.
        return f"SQL error: {e}"
    truncated = len(rows) > MAX_ROWS
    lines = [" | ".join(headers)] + [" | ".join(map(str, r)) for r in rows[:MAX_ROWS]]
    if truncated:
        lines.append(f"... truncated to {MAX_ROWS} rows. Aggregate or add LIMIT.")
    return "\n".join(lines)


TOOLS = [list_tables, describe_table, run_query]

SYSTEM_PROMPT = """You are an energy market analyst for the German electricity system.
You answer questions using a SQLite database built from SMARD (Bundesnetzagentur) data.

Domain knowledge:
- All values are hourly. generation.mwh and consumption.*_mwh are MWh per hour
  (numerically equal to average MW in that hour). Divide by 1,000,000 for TWh.
- Renewable share = renewable generation / total generation. Use the
  sources.category column, never a hard-coded list of sources.
- prices.eur_per_mwh is the day-ahead price for the DE-LU bidding zone and CAN be negative.
- Residual load = grid load minus wind and solar. High residual load means
  conventional plants must run, which usually means higher prices.
- Use ts_local (Europe/Berlin) for questions about days, hours, weekdays or months.
  SQLite helpers: date(ts_local), strftime('%H', ts_local), strftime('%w', ts_local) (0 = Sunday).
  ts_local is not unique (the hour repeats when daylight saving ends), so join tables on ts_utc.

Process:
1. Inspect the schema (list_tables, describe_table) before querying unfamiliar tables.
2. Check the available date range before answering time-based questions.
3. Write precise SELECT queries. Aggregate in SQL instead of pulling raw rows.
4. If a query errors or a result looks implausible, investigate and fix it.
5. Answer concisely with units (MWh, TWh, EUR/MWh, %), and show the final SQL.
Never guess numbers: every figure in your answer must come from a query result.
If the data cannot answer the question, say so."""


# --------------------------------------------------------------------- graph
class State(TypedDict):
    # add_messages is a reducer: new messages are appended, not overwritten.
    messages: Annotated[list[AnyMessage], add_messages]


def build_graph():
    llm = init_chat_model(MODEL, temperature=0).bind_tools(TOOLS)

    def agent(state: State) -> dict:
        msgs = [SystemMessage(SYSTEM_PROMPT)] + state["messages"]
        return {"messages": [llm.invoke(msgs)]}

    g = StateGraph(State)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_edge(START, "agent")
    # If the model requested tool calls -> "tools", otherwise -> END.
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    # The checkpointer stores state per thread_id -> multi-turn memory.
    return g.compile(checkpointer=InMemorySaver())


def main():
    graph = build_graph()
    config = {"configurable": {"thread_id": "cli"}, "recursion_limit": 25}
    print("Energy agent ready. Ask about the German electricity market (Ctrl+C to quit).")
    while True:
        try:
            q = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not q:
            continue
        # stream_mode="updates" shows each step: useful for seeing the loop.
        for update in graph.stream({"messages": [("user", q)]}, config, stream_mode="updates"):
            for node, data in update.items():
                msg = data["messages"][-1]
                if node == "agent" and getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls:
                        print(f"  -> {tc['name']}({tc['args']})")
                elif node == "agent":
                    print(f"\nAgent: {msg.content}")


if __name__ == "__main__":
    main()
