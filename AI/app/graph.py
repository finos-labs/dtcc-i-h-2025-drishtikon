from pydantic_ai import Agent
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, List
from langgraph.types import interrupt
from dotenv import load_dotenv
import logfire

from app.utils import model
from app.agents import get_zerodha_agent

# Load environment variables
load_dotenv()

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire='never')

router_agent = Agent(  
    model = model,
    system_prompt='Your job is to route the user to the relevant agent.',  
)

end_conversation_agent = Agent(  
    model = model,
    system_prompt='Your job is to end a conversation and summarize the whole conversation.',  
)

# Define state schema
class AgentState(TypedDict):
    messages: Annotated[List[dict], lambda x, y: x + y]

async def router_agen(state: AgentState):
    prompt = f"""
        Based on the conversation till now, your task is to route to which agent we should go.

        {state['messages']}

        List of available agents:
        "zerodha_agent": It is capable of doing trade execution, market anaylsis and post-trade tasks.
        "end_conversation_agent:" It's task is to end the conversation and provide the final result.

        Output should the name of the agent which is chosen.

        Example Outputs:
        "end_conversation_agent"
        "zerodha_agent"
    """

    result = await router_agent.run(prompt)
    next_action = result.output

    if next_action == "zerodha_agent":
        return "zerodha_agent"
    else:
        return "end_conversation"

async def zerodga_agent(state: AgentState):
    mcp_client, mcp_agent = await get_zerodha_agent()
    
    try:
            result = await mcp_agent.run(state['messages'][-1], message_history=state['messages'][:-1])
            
            # Add the new messages to the chat history
            return {
                "messages": [
                    {
                        "role": "assistant", 
                        "content": result.output
                    }
                ]
            }
    except Exception as e:
        print(f"\n[Error] An error occurred: {str(e)}")
    finally:
        await mcp_client.cleanup()

# Interrupt the graph to get the user's next message
def get_next_user_message(state: AgentState):
    value = interrupt({})

    # Set the user's latest message for the LLM to continue the conversation
    return {
        "latest_user_message": value
    }

# End of conversation agent to give instructions for executing the agent
async def end_conversation(state: AgentState, writer):
    prompt = f"""Summarize the conversation and give the final output, ther user will see only your output, so make you sure you present a good, concise yet clear output. You may make tables, charts or any other form of visual representaiton of the data to make the output more appealing.
    
    This is the conversation:
    {state['messages']}
    """

    result = await end_conversation_agent.run(prompt)
    return {
        "messages": [
            {
                "role": "assistant", 
                "content": result.output
            }
        ]
    }

# Build workflow
builder = StateGraph(AgentState)

# Add nodes
builder.add_node("zerodha_agent", zerodga_agent)
builder.add_node("end_conversation", end_conversation)

# Set edges
builder.add_conditional_edges(
    START,
    router_agent,
    {
        "zerodha_agent": "zerodha_agent", 
        "end_conversation": "end_conversation"
    }
)
builder.add_conditional_edges(
    "zerodha_agent",
    router_agent,
    {
        "zerodha_agent": "zerodha_agent", 
        "end_conversation": "end_conversation"
    }
)
builder.add_edge("end_conversation", END)

# Configure persistence
memory = MemorySaver()
agentic_flow = builder.compile(checkpointer=memory)