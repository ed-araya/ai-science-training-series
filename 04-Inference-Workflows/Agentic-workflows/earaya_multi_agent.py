from typing import TypedDict, Annotated

from langgraph.graph import add_messages
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from inference_auth_token import get_access_token

#from earaya_tools import molecule_name_to_smiles, smiles_to_coordinate_file, run_mace_calculation, loadascii, getsubdata, RMS
from earaya_tools import loadascii, getsubdata, RMS


# ============================================================
# 1. State definition
# ============================================================
class State(TypedDict):
    messages: Annotated[list, add_messages]


# ============================================================
# 2. Routing logic
# ============================================================
def route_tools(state: State):
    """Route to the 'tools' node if the last message has tool calls; otherwise, route to 'done'.

    Parameters
    ----------
    state : State
        The current state containing messages and remaining steps

    Returns
    -------
    str
        Either 'tools' or 'done' based on the state conditions
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "done"


# ============================================================
# 3. LLM node: the "agent"
# ============================================================
def ea_agent(
    state: State,
    llm: ChatOpenAI,
    tools: list,
    system_prompt: str = "You are an assistant that use tools to solve problems ",
):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]
    llm_with_tools = llm.bind_tools(tools=tools)
    return {"messages": [llm_with_tools.invoke(messages)]}


# ============================================================
# 3*. A second agent: Handle creating structured output
# ============================================================
def structured_output_agent(
    state: State,
    llm: ChatOpenAI,
    system_prompt: str = ("You are an assistant that returns ONLY JSON. "),
):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]

    result = llm.invoke(messages)
    return {"messages": [result]}


# ============================================================
# 4. LLM / tools setup
# ============================================================
# Get token for your ALCF inference endpoint
access_token = get_access_token()

# Initialize the model hosted on the ALCF endpoint
llm = ChatOpenAI(
    model_name="openai/gpt-oss-20b",
    # model_name="Qwen/Qwen3-32B",
    api_key=access_token,
    base_url="https://data-portal-dev.cels.anl.gov/resource_server/sophia/vllm/v1",
    temperature=0,
)

# Tool list that the LLM can call
tools = [molecule_name_to_smiles, smiles_to_coordinate_file, run_mace_calculation, loadascii, RMS, getsubdata]

# ============================================================
# 5. Build the graph
# ============================================================
graph_builder = StateGraph(State)

# Agent node: calls LLM, which may decide to call tools
graph_builder.add_node(
    "ea_agent",
    lambda state: ea_agent(state, llm=llm, tools=tools),
)
graph_builder.add_node(
    "structured_output_agent",
    lambda state: structured_output_agent(state, llm=llm),
)

# Tool node: executes tool calls emitted by the LLM
tool_node = ToolNode(tools)
graph_builder.add_node("tools", tool_node)

# Graph logic
# START -> ea_agent
graph_builder.add_edge(START, "ea_agent")

# After ea_agent runs, check if we need to run tools
graph_builder.add_conditional_edges(
    "ea_agent", route_tools, {"tools": "tools", "done": "structured_output_agent"}
)

# After tools run, go back to the agent so it can use tool results
graph_builder.add_edge("tools", "ea_agent")

# After structured_output_agent, terminate the graph
graph_builder.add_edge("structured_output_agent", END)
# Compile the graph
graph = graph_builder.compile()

# ============================================================
# 6. Run / stream the graph
# ============================================================
#prompt = "Provide the list of the hyperfine structure energy transitions of the ammonia (7,7) inversion transition, with an accuracy of 10 kHz. Return the results in a JSON."
prompt = "The file OH_6035MHz_Spec_line.txt in the current directory contains two columns, x = velocity (km/s), y = flux density (Jy). Use the tools to load the data from OH_6035MHz_Spec_line.txt, the output should be a list, name it 'spec'. Report the number of velocity channels in the spectrum."


for chunk in graph.stream(
    {"messages": prompt},
    stream_mode="values",
):
    new_message = chunk["messages"][-1]
    new_message.pretty_print()

print('''Comments: I was trying to make the agent send data from one tool 
function to another to simulate basic analysis of the noise in a spectrum, 
but then I found that the agent has issues running the first tool (loadascii), 
which simply loads some data from a file. Without changing anything, a test 
resulted in the agent reporting that the dataset had 2000 channels, another 
test resulted in the agent reporting ~2048 channels. I see lots of promise 
for agentic workflows for data reduction/analysis in astrophysics, but I will 
need to play more with the tools to find out the origin of the discrepancies.''')
