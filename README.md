# German Energy Market Agent

An AI agent that answers questions about the German electricity market in plain language.
It explores a database of official grid data, writes SQL, corrects its own mistakes,
and reports answers with units and the query it used.

> *"On how many days last summer were prices negative, and what was the solar share on those days?"*
> *"How does the day-ahead price relate to residual load by hour of day?"*
> *"Which month had the highest share of lignite generation?"*

## Data

Hourly data for Germany from **[SMARD](https://www.smard.de)**, the electricity market
platform of the Bundesnetzagentur (CC BY 4.0):

| Table | Content |
|---|---|
| `generation` | MWh per hour by source (solar, wind on/offshore, lignite, gas, ...) |
| `sources` | Source → category (`renewable` / `conventional` / `storage`) |
| `consumption` | Grid load and residual load (MWh) |
| `prices` | Day-ahead wholesale price, DE-LU bidding zone (EUR/MWh, can be negative) |

Timestamps are stored in UTC (join key) and Europe/Berlin local time (for daily and hourly analysis),
which handles daylight-saving changes correctly.

## Architecture

```
START -> agent (LLM) --tool calls?--> tools (list_tables / describe_table / run_query)
             ^                              |
             +------------------------------+
         no tool calls -> END
```

- **LangGraph `StateGraph`** with a message reducer and checkpointer (multi-turn memory via `thread_id`)
- **Domain knowledge in the system prompt**: units, renewable share definition, negative prices, DST pitfalls
- **Provider-agnostic** model via `init_chat_model`: runs free on Groq (open-weight `gpt-oss-120b`) by default; switch to a local Ollama model, Anthropic or OpenAI with one environment variable

## Guardrails

| Risk | Mitigation |
|---|---|
| Writes / destructive SQL | Read-only connection + SELECT-only check + single statement |
| Injection via tool arguments | Table names validated against the schema |
| Huge results / context overflow | 50-row cap with truncation notice |
| Runaway queries | 5 s query timeout |
| Infinite agent loops | `recursion_limit=25` |
| Made-up numbers | Prompt requires every figure to come from a query result |

## Run

```bash
pip install -r requirements.txt
python setup_db.py --weeks 52      # downloads ~1 year of SMARD data (a few minutes)
export GROQ_API_KEY=...            # free key, no credit card: console.groq.com
python energy_agent.py
```

## Roadmap

- [ ] Evaluation suite: ground-truth questions, accuracy score, tracing
- [ ] MCP server exposing the energy tools
- [ ] Human-in-the-loop approval with LangGraph interrupts
- [ ] Forecasting tool (e.g. next-day price) the agent can call
