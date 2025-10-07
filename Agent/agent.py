from typing import (
    Annotated,
    Sequence,
    TypedDict,
)
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
import json
from langchain_core.messages import ToolMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

lama = "llama3-70b-8192"
deep_seek = "deepseek-r1-distill-llama-70b"
open_ai = "openai/gpt-oss-120b"


class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def get_weather(location: str):
    """Call to get the weather from a specific location."""
    return "It's sunny in San Francisco, but you better look out if you're a Gemini 😈."



class CollabFlowAgent:
    def __init__(self):
        self.memory = InMemorySaver()
        self.graph = None

    def initialize(self):
        llm = ChatGroq(temperature=0.1, model_name=deep_seek)
        tools = [get_weather]
        model = llm.bind_tools(tools)

        tools_by_name = {tool.name: tool for tool in tools}
        def tool_node(state: AgentState):
            outputs = []
            for tool_call in state["messages"][-1].tool_calls:
                tool_result = tools_by_name[tool_call["name"]].invoke(
                    tool_call["args"]
                )
                outputs.append(
                    ToolMessage(
                        content=json.dumps(tool_result),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
            return {"messages": outputs}

        def call_model(state: AgentState, config: RunnableConfig):
            system_prompt = SystemMessage(
                "You are a helpful AI assistant. Respond concisely and clearly."
            )
            response = model.invoke([system_prompt] + state["messages"], config)
            return {"messages": [response]}

        def should_continue(state: AgentState):
            messages = state["messages"]
            last_message = messages[-1]
            if not getattr(last_message, "tool_calls", None):
                return "end"
            else:
                return "continue"

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {"continue": "tools", "end": END},
        )
        workflow.add_edge("tools", "agent")

        self.graph = workflow.compile(checkpointer=self.memory)

    def send_message(self, user_input: str, session_id: str) -> str:
        """Send a message to the LLM agent and return the response."""
        if not self.graph:
            raise ValueError("Graph not initialized. Call initialize() first.")

        config = {"configurable": {"thread_id": session_id}}

        # Prepare user input
        inputs = {"messages": [HumanMessage(content=user_input)]}

        # Run the graph
        final_state = self.graph.invoke(inputs, config)

        # Get the last message (AI response)
        last_message = final_state["messages"][-1]

        # Return text content
        if isinstance(last_message, AIMessage):
            return last_message.content
        return str(last_message)


# Example usage:
if __name__ == "__main__":
    agent = CollabFlowAgent()
    agent.initialize()
    response = agent.send_message("What is the weather in SF?", session_id="demo-session")
    print("LLM Response:", response)
