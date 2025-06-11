from pydantic_ai import Agent
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, List
from langgraph.types import interrupt
from dotenv import load_dotenv
import logfire
import asyncio
import uuid

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
    # Format messages for better readability
    formatted_messages = []
    for msg in state['messages']:
        if isinstance(msg, dict):
            role = msg.get('role', 'unknown')
            content = msg.get('content', str(msg))
            formatted_messages.append(f"{role}: {content}")
        else:
            formatted_messages.append(str(msg))
    
    conversation_text = "\n".join(formatted_messages)
    
    # Check if the last message indicates technical difficulties
    last_message = state['messages'][-1] if state['messages'] else {}
    if isinstance(last_message, dict):
        last_content = last_message.get('content', '').lower()
        if any(phrase in last_content for phrase in ['technical difficulties', 'error occurred', 'having trouble', 'experiencing difficulties']):
            print("🔀 Router decision: end_conversation (due to technical issues)")
            return "end_conversation"
    
    # Count consecutive zerodha_agent responses to prevent infinite loops
    consecutive_assistant_messages = 0
    for msg in reversed(state['messages']):
        if isinstance(msg, dict) and msg.get('role') == 'assistant':
            consecutive_assistant_messages += 1
        else:
            break
    
    # If we've had too many consecutive assistant messages, end the conversation
    if consecutive_assistant_messages >= 2:
        print("🔀 Router decision: end_conversation (too many consecutive responses)")
        return "end_conversation"
    
    prompt = f"""
        Based on the conversation till now, your task is to route to which agent we should go.

        Conversation:
        {conversation_text}

        List of available agents:
        "zerodha_agent": It is capable of doing trade execution, market analysis and post-trade tasks.
        "end_conversation_agent": Its task is to end the conversation and provide the final result.

        Rules:
        - If the user is asking about trading, stocks, market analysis, portfolio, or wants to execute trades, route to "zerodha_agent"
        - If the user wants to end the conversation, get a summary, or says goodbye, route to "end_conversation_agent"
        - If there have been technical errors or difficulties, route to "end_conversation_agent"
        - If this is just a greeting or simple hello, route to "zerodha_agent" to handle it

        Output should be ONLY the name of the agent which is chosen.

        Example Outputs:
        zerodha_agent
        end_conversation_agent
    """

    result = await router_agent.run(prompt)
    next_action = result.output.strip().lower()

    print(f"🔀 Router decision: {next_action}")

    if "zerodha" in next_action:
        return "zerodha_agent"
    elif "end_conversation" in next_action:
        return "end_conversation"
    else:
        # Default to zerodha_agent if unclear
        return "zerodha_agent"

async def zerodga_agent(state: AgentState):
    print(f"🔧 Zerodha agent called with state: {state}")
    
    try:
        mcp_client, mcp_agent = await get_zerodha_agent()
    except Exception as e:
        print(f"\n[Error] Failed to get zerodha agent: {str(e)}")
        return {
            "messages": [
                {
                    "role": "assistant", 
                    "content": f"I'm having trouble connecting to the trading system. Error: {str(e)}. Would you like me to end our conversation and provide a summary instead?"
                }
            ]
        }
    
    try:
        # Extract the content from the latest message
        latest_message = state['messages'][-1]
        print(f"🔧 Latest message: {latest_message}")
        
        if isinstance(latest_message, dict) and 'content' in latest_message:
            user_message = latest_message['content']
        else:
            user_message = str(latest_message)
        
        print(f"🔧 Extracted user message: {user_message}")
        
        # Get message history (exclude the latest message)
        message_history = state['messages'][:-1]
        print(f"🔧 Message history length: {len(message_history)}")
        
        # Try different approaches to call the MCP agent
        try:
            # First try: pass just the user message content
            result = await mcp_agent.run(user_message)
            print(f"🔧 MCP agent result: {result}")
        except Exception as mcp_error:
            print(f"🔧 First attempt failed: {mcp_error}")
            try:
                # Second try: pass the message with history
                result = await mcp_agent.run(user_message, message_history=message_history)
                print(f"🔧 MCP agent result (with history): {result}")
            except Exception as mcp_error2:
                print(f"🔧 Second attempt failed: {mcp_error2}")
                # Third try: different format
                result = await mcp_agent.run(latest_message)
                print(f"🔧 MCP agent result (dict format): {result}")
        
        # Add the new messages to the chat history
        return {
            "messages": [
                {
                    "role": "assistant", 
                    "content": result.output if hasattr(result, 'output') else str(result)
                }
            ]
        }
    except Exception as e:
        print(f"\n[Error] An error occurred in zerodha agent: {str(e)}")
        print(f"🔧 Error type: {type(e)}")
        import traceback
        print(f"🔧 Full traceback: {traceback.format_exc()}")
        
        # Return a fallback response that suggests ending conversation
        return {
            "messages": [
                {
                    "role": "assistant", 
                    "content": f"I'm experiencing technical difficulties with the trading system. The error was: {str(e)}. Let me end our conversation and provide you with a summary of what we've discussed so far."
                }
            ]
        }
    finally:
        try:
            if 'mcp_client' in locals():
                await mcp_client.cleanup()
        except:
            pass

# Interrupt the graph to get the user's next message
def get_next_user_message(state: AgentState):
    value = interrupt({})

    # Set the user's latest message for the LLM to continue the conversation
    return {
        "latest_user_message": value
    }

# End of conversation agent to give instructions for executing the agent
async def end_conversation(state: AgentState):
    # Format messages for better readability
    formatted_messages = []
    for msg in state['messages']:
        if isinstance(msg, dict):
            role = msg.get('role', 'unknown')
            content = msg.get('content', str(msg))
            formatted_messages.append(f"{role}: {content}")
        else:
            formatted_messages.append(str(msg))
    
    conversation_text = "\n".join(formatted_messages)
    
    prompt = f"""Summarize the conversation and give the final output. The user will see only your output, so make sure you present a good, concise yet clear output. You may make tables, charts or any other form of visual representation of the data to make the output more appealing.
    
    This is the conversation:
    {conversation_text}
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
    router_agen,
    {
        "zerodha_agent": "zerodha_agent", 
        "end_conversation": "end_conversation"
    }
)
builder.add_conditional_edges(
    "zerodha_agent",
    router_agen,
    {
        "zerodha_agent": "zerodha_agent", 
        "end_conversation": "end_conversation"
    }
)
builder.add_edge("end_conversation", END)

# Configure persistence
memory = MemorySaver()
agentic_flow = builder.compile(checkpointer=memory)

async def run_cli():
    """Interactive CLI for testing the agentic flow"""
    print("🚀 Agentic Flow CLI Test")
    print("=" * 50)
    print("Commands:")
    print("  - Type your message to interact with agents")
    print("  - Type 'quit' or 'exit' to end the session")
    print("  - Type 'clear' to start a new conversation")
    print("  - Type 'help' to see this menu again")
    print("=" * 50)
    
    # Generate a unique thread ID for this session
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    while True:
        try:
            # Get user input
            user_input = input("\n💬 You: ").strip()
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit']:
                print("\n👋 Thanks for testing! Goodbye!")
                break
            elif user_input.lower() == 'clear':
                thread_id = str(uuid.uuid4())
                config = {"configurable": {"thread_id": thread_id}}
                print("\n🔄 New conversation started!")
                continue
            elif user_input.lower() == 'help':
                print("\nCommands:")
                print("  - Type your message to interact with agents")
                print("  - Type 'quit' or 'exit' to end the session")
                print("  - Type 'clear' to start a new conversation")
                print("  - Type 'help' to see this menu again")
                continue
            elif not user_input:
                print("Please enter a message or command.")
                continue
            
            # Prepare initial state with user message
            initial_state = {
                "messages": [
                    {
                        "role": "user",
                        "content": user_input
                    }
                ]
            }
            
            print("\n🤖 Processing...")
            
            # Run the agentic flow
            result = await agentic_flow.ainvoke(initial_state, config=config)
            
            # Display the final response
            if result and "messages" in result:
                final_message = result["messages"][-1]["content"]
                print(f"\n🎯 Assistant: {final_message}")
            else:
                print("\n⚠️  No response received from the agents.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again or type 'quit' to exit.")

def run_single_test(message: str):
    """Run a single test with a predefined message"""
    async def _run():
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        initial_state = {
            "messages": [
                {
                    "role": "user",
                    "content": message
                }
            ]
        }
        
        print(f"🧪 Testing with message: '{message}'")
        print("🤖 Processing...")
        
        try:
            result = await agentic_flow.ainvoke(initial_state, config=config)
            
            if result and "messages" in result:
                final_message = result["messages"][-1]["content"]
                print(f"\n🎯 Response: {final_message}")
            else:
                print("\n⚠️  No response received.")
                
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
    
    return asyncio.run(_run())

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # If arguments provided, run single test
        test_message = " ".join(sys.argv[1:])
        run_single_test(test_message)
    else:
        # Otherwise run interactive CLI
        asyncio.run(run_cli())