import os
import io
from langchain.chat_models import init_chat_model
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from PIL import Image as PILImage
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env into the environment

"""
    Create a StateGraph
"""
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


"""
    Add a node
"""
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("Missing GOOGLE_API_KEY in environment")

os.environ["GOOGLE_API_KEY"] = google_api_key

llm = init_chat_model("google_genai:gemini-2.0-flash")

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)

"""
    Add an entry point
"""

graph_builder.add_edge(START, "chatbot")

"""
    Compile the graph
"""
graph = graph_builder.compile()

"""
    Visualize the graph
"""
try:
    img_bytes = graph.get_graph().draw_mermaid_png()
    # Open image from bytes and show it
    img = PILImage.open(io.BytesIO(img_bytes))
    img.show()  
except Exception:
    # This requires some extra dependencies and is optional
    pass

"""
    Run the ChatBoot
"""
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break