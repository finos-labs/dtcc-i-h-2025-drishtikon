import streamlit as st
import asyncio
import sys
import pathlib
from typing import List, Dict, Any
import traceback

sys.path.append(str(pathlib.Path(__file__).parent / "app"))

from app.agents.zerodha_agent.agent import mcp_client, mcp_agent

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    
    .error-message {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        color: #d32f2f;
    }
    
    .tool-info {
        background-color: #f5f5f5;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_initialized" not in st.session_state:
    st.session_state.agent_initialized = False

if "mcp_client" not in st.session_state:
    st.session_state.mcp_client = mcp_client

if "mcp_agent" not in st.session_state:
    st.session_state.mcp_agent = mcp_agent

if "available_tools" not in st.session_state:
    st.session_state.available_tools = []

async def initialize_agent():
    """Initialize the Pydantic AI agent asynchronously"""
    try:
        with st.spinner("Initializing AI agent and tools..."):
            # mcp_client, mcp_agent = await get_zerodha_agent()
            
            # Store in session state
            st.session_state.mcp_client = mcp_client
            st.session_state.mcp_agent = mcp_agent
            st.session_state.agent_initialized = True
            
            return True, "Agent initialized successfully!"
            
    except Exception as e:
        error_msg = f"Failed to initialize agent: {str(e)}"
        st.error(error_msg)
        return False, error_msg

async def get_agent_response(user_input: str, message_history: List[Any]):
    """Get response from the agent"""
    try:
        # Get the agent response
        result = await st.session_state.mcp_agent.run(
            user_input, 
            message_history=message_history
        )
        
        # Return the response data and updated message history
        return result.data, result.all_messages(), None
        
    except Exception as e:
        error_msg = f"Error getting agent response: {str(e)}\n{traceback.format_exc()}"
        return None, message_history, error_msg

def run_async(coro):
    """Helper function to run async code in Streamlit"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)

# Sidebar
with st.sidebar:
    st.title("🤖 AI Chatbot")
    st.markdown("---")
    
    # Agent initialization status
    if st.session_state.agent_initialized:
        st.success("✅ Agent Ready")
        
        # Display available tools if any
        if st.session_state.available_tools:
            st.subheader("Available Tools")
            for i, tool in enumerate(st.session_state.available_tools, 1):
                with st.expander(f"{i}. {tool.get('name', 'Unknown')}"):
                    st.write(tool.get('description', 'No description available'))
    else:
        st.warning("⚠️ Agent Not Initialized")
        if st.button("Initialize Agent"):
            success, message = run_async(initialize_agent())
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
    
    st.markdown("---")
    
    # Chat controls
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    # Settings
    st.subheader("Settings")
    show_debug = st.checkbox("Show Debug Info", value=False)

st.title("AI Assistant Chat")

# Auto-initialize agent on first load
if not st.session_state.agent_initialized:
    with st.spinner("Initializing agent..."):
        success, message = run_async(initialize_agent())
        if success:
            st.success(message)
            st.rerun()

chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show debug info if enabled
            if show_debug and "debug_info" in message:
                with st.expander("Debug Info"):
                    st.json(message["debug_info"])

if st.session_state.agent_initialized:
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user", 
            "content": user_input
        })
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Get agent response
        with st.chat_message("ass-istant"):
            with st.spinner("Thinking..."):
                try:
                    # Extract message history for the agent
                    message_history = []
                    for msg in st.session_state.messages[:-1]:  # Exclude the current user message
                        if msg["role"] == "user":
                            message_history.append({"role": "user", "content": msg["content"]})
                        elif msg["role"] == "assistant":
                            message_history.append({"role": "assistant", "content": msg["content"]})
                    
                    # Get response from agent
                    response_data, updated_messages, error = run_async(
                        get_agent_response(user_input, message_history)
                    )
                    
                    if error:
                        st.error(f"Error: {error}")
                        assistant_message = f"I encountered an error: {error}"
                    else:
                        assistant_message = str(response_data) if response_data else "I'm sorry, I couldn't generate a response."
                    
                    
                    st.markdown(assistant_message)
                    
                    assistant_msg = {
                        "role": "assistant", 
                        "content": assistant_message
                    }
                    
                    if show_debug:
                        assistant_msg["debug_info"] = {
                            "response_data": response_data,
                            "message_count": len(updated_messages) if updated_messages else 0
                        }
                    
                    st.session_state.messages.append(assistant_msg)
                    
                except Exception as e:
                    error_msg = f"Unexpected error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
        
        # Rerun to update the chat display
        st.rerun()

else:
    st.info("Please initialize the agent first using the sidebar.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
        AI Chatbot powered by Pydantic AI with MCP Integration
    </div>
    """, 
    unsafe_allow_html=True
)

# Cleanup function (called when app is stopped)
def cleanup():
    """Cleanup MCP client resources"""
    if st.session_state.mcp_client:
        try:
            run_async(st.session_state.mcp_client.cleanup())
        except Exception as e:
            st.error(f"Cleanup error: {e}")

# Register cleanup function
import atexit
atexit.register(cleanup)