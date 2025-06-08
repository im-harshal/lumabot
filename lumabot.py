from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages.ai import AIMessage
from typing import Literal
from langchain_core.tools import tool
from collections.abc import Iterable
from random import randint
from langchain_core.messages.tool import ToolMessage
from langgraph.prebuilt import ToolNode
from pprint import pprint

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

class OrderState(TypedDict):
    """State representing the customer's order conversation."""

    # The chat conversation. This preserves the conversation history
    # between nodes. The `add_messages` annotation indicates to LangGraph
    # that state is updated by appending returned messages, not replacing
    # them.
    messages: Annotated[list, add_messages]

    # The customer's in-progress order.
    order: list[str]

    # Flag indicating that the order is placed and completed.
    finished: bool


# The system instruction defines how the chatbot is expected to behave and includes
# rules for when to call different functions, as well as rules for the conversation, such
# as tone and what is permitted for discussion.
LUMABOT_SYSINT = (
    "system",
    "You are LumaBot, an interactive cafe ordering assistant."

    "Your job is to:"
    "- Help customers with questions about the menu (products, history, and available modifiers)."
    "- Take orders for one or more items from the menu."
    "- Guide the customer through order confirmation, ensuring the order is correct and priced accurately."
    "- Complete the order by sending it to the kitchen and providing an estimated pickup time."

    "Menu handling:"
    "- Only discuss items and modifiers listed on the menu. If a customer asks about an unavailable option, ask for clarification or redirect them to what's available.\n"
    "- For clarification, always verify drink and modifier names against the MENU before adding to the order."

    "Tool usage:"
    "- Use 'add_to_order' to add items."
    "- Use 'clear_order' to reset the order."
    "- Use 'get_order' to check the current order (for yourself, not for the customer)."
    "- Before placing an order, always use 'confirm_order' with the customer to show the full order for review."
    "- Do NOT repeat the order total or summary in your reply after confirmation—this is already displayed to the user by the system. Instead, only acknowledge the order and ask for final confirmation or next steps."
    "- Once the customer confirms the order, use 'place_order' to finalize it and provide an estimated pickup time."
    
    "Conversation flow:"
    "1. Greet the customer and ask how you can help."
    "2. Answer menu questions and take their order using the tools above."
    "3. Confirm the order—let the system display the full summary and total. Only ask for confirmation or next steps in your reply."
    "4. If the order is confirmed, place it and thank the customer."
    "5. End the conversation politely after order completion."

    "If any tool is unavailable, inform the user that it hasn't been implemented yet."
)

# This is the message with which the system opens the conversation.
WELCOME_MSG = "Welcome to the LumaBot cafe. Type `q` to quit. How may I serve you today?"

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


def human_node(state: OrderState) -> OrderState:
    """Display the last model message to the user, and receive the user's input."""
    last_msg = state["messages"][-1]
    print("Model:", last_msg.content)

    user_input = input("User: ")

    # If it looks like the user is trying to quit, flag the conversation
    # as over.
    if user_input in {"q", "quit", "exit", "goodbye"}:
        state["finished"] = True

    return state | {"messages": [("user", user_input)]}

def maybe_exit_human_node(state: OrderState) -> Literal["chatbot", "__end__"]:
    """Route to the chatbot, unless it looks like the user is exiting."""
    if state.get("finished", False):
        return END
    else:
        return "chatbot"

@tool
def get_menu() -> str:
    """Provide the latest up-to-date menu, including the drink prices."""
    # Note that this is just hard-coded text, but you could connect this to a live stock
    # database, or you could use Gemini's multi-modal capabilities and take live photos of
    # your cafe's chalk menu or the products on the counter and assmble them into an input.

    return """
    MENU:
    Coffee Drinks:
      - Espresso ($2.50)
      - Americano ($2.50)
      - Cold Brew ($3.00)

    Coffee Drinks with Milk:
      - Latte ($3.50)
      - Cappuccino ($3.50)
      - Cortado ($3.25)
      - Macchiato ($3.25)
      - Mocha ($4.00)
      - Flat White ($3.50)

    Tea Drinks:
      - English Breakfast Tea ($2.00)
      - Green Tea ($2.00)
      - Earl Grey ($2.00)

    Tea Drinks with Milk:
      - Chai Latte ($3.50)
      - Matcha Latte ($4.00)
      - London Fog ($3.50)

    Other Drinks:
      - Steamer ($2.50)
      - Hot Chocolate ($2.75)

    Modifiers:
    Milk options: Whole, 2%, Oat, Almond, 2% Lactose Free; Default option: whole
    Espresso shots: Single, Double, Triple, Quadruple; default: Double
    Caffeine: Decaf, Regular; default: Regular
    Hot-Iced: Hot, Iced; Default: Hot
    Sweeteners (option to add one or more): vanilla sweetener, hazelnut sweetener, caramel sauce, chocolate sauce, sugar free vanilla sweetener
    Special requests: any reasonable modification that does not involve items not on the menu, for example: 'extra hot', 'one pump', 'half caff', 'extra foam', etc.

    "dirty" means add a shot of espresso to a drink that doesn't usually have it, like "Dirty Chai Latte".
    "Regular milk" is the same as 'whole milk'.
    "Sweetened" means add some regular sugar, not a sweetener.

    Soy milk has run out of stock today, so soy is not available.
  """

def calculate_total(order: list[str]) -> float:
    """
    Calculates the total price (inclusive of tax) of the customer's order.

    Args:
    order: A list of strings, each representing an order item in the format 'Drink Name | modifiers | price'.

    Returns:
    The total price of all items in the order as a float.
    """
    total = 0.0
    for item in order:
        try:
            # Split by '|' and take the last part as price
            price_str = item.strip().split('$')[-1].strip()
            price = float(price_str)
            total += price
        except Exception:
            # In case price parsing fails, skip this item
            continue
    return round(total, 2)

def calculate_total_tokens(messages: list) -> int:
    """
    Sums up the total LLM tokens used in the session
    
    Args:
        messages (list): The conversation history (state["messages"])
        
    Returns:
        int: Total LLM tokens used in the session
    """
    total_tokens = 0
    for msg in messages:
        # Check if message is an AIMessage with usage_metadata
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            tokens = msg.usage_metadata.get("total_tokens", 0)
            total_tokens += tokens
    return total_tokens

def chatbot_with_tools(state: OrderState) -> OrderState:
    """The chatbot with tools. A simple wrapper around the model's own chat interface."""
    defaults = {"order": [], "finished": False}

    if state["messages"]:
        new_output = llm_with_tools.invoke([LUMABOT_SYSINT] + state["messages"])
    else:
        new_output = AIMessage(content=WELCOME_MSG)

    # Set up some defaults if not already set, then pass through the provided state,
    # overriding only the "messages" field.
    return defaults | state | {"messages": [new_output]}

# These functions have no body; LangGraph does not allow @tools to update
# the conversation state, so you will implement a separate node to handle
# state updates. Using @tools is still very convenient for defining the tool
# schema, so empty functions have been defined that will be bound to the LLM
# but their implementation is deferred to the order_node.

def add_to_order(drink: str, modifiers: Iterable[str], price) -> str:
    """Adds the specified drink to the customer's order, including the price and any modifiers.

    Returns:
      The updated order in progress.
    """

@tool
def confirm_order():
    """Displays the order summary which include the all the orders (including modifications and prices), taxes, and total cost of the orders.
    """

@tool
def get_order() -> str:
    """Returns the users order so far. One item per line."""

@tool
def clear_order():
    """Removes all items from the user's order."""

@tool
def place_order() -> int:
    """Sends the order to the barista for fulfillment.

    Returns:
      The estimated number of minutes until the order is ready.
    """


def order_node(state: OrderState) -> OrderState:
    """The ordering node. This is where the order state is manipulated."""
    tool_msg = state.get("messages", [])[-1]
    order = state.get("order", [])
    outbound_msgs = []
    order_placed = False

    for tool_call in tool_msg.tool_calls:

        if tool_call["name"] == "add_to_order":

            # Each order item is just a string. This is where it assembled as "drink (modifiers, ...)".
            modifiers = tool_call["args"].get("modifiers", [])
            modifier_str = ", ".join(modifiers) if modifiers else "No Modifiers"

            order.append(f'{tool_call["args"]["drink"]} | {modifier_str} | ${round(float(tool_call["args"].get("price", 0)),2)}')
            response = "\n".join(order)

        elif tool_call["name"] == "confirm_order":
            print("Order Summary: ")

            for drink in order:
                print(f"    {drink}")

            before_tax = calculate_total(order)
            sales_tax = round(before_tax*0.0825, 2)
            convenience_fee = round(calculate_total_tokens(state.get("messages", []))*0.001, 2)
            grand_total = round(before_tax+sales_tax+convenience_fee, 2)
            print(f"    Total: ${round(before_tax,2)}")
            print(f"    Sales Tax (8.25%): ${sales_tax}")
            print(f"    Convenience Fee: ${convenience_fee}")
            print(f"    Grand Total: ${grand_total}")
            response = None

        elif tool_call["name"] == "get_order":
            response = "\n".join(order) if order else "(no order)"

        elif tool_call["name"] == "clear_order":
            order.clear()
            response = None

        elif tool_call["name"] == "place_order":

            print("Thank you! Sending order to the kitchen!")
            order_placed = True
            response = randint(1, 5)  # ETA in minutes

        else:
            raise NotImplementedError(f'Unknown tool call: {tool_call["name"]}')

        # Record the tool results as tool messages.
        outbound_msgs.append(
            ToolMessage(
                content=response,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )

    return {"messages": outbound_msgs, "order": order, "finished": order_placed}


def maybe_route_to_tools(state: OrderState) -> str:
    """Route between chat and tool nodes if a tool call is made."""
    if not (msgs := state.get("messages", [])):
        raise ValueError(f"No messages found when parsing state: {state}")

    msg = msgs[-1]

    if state.get("finished", False):
        # When an order is placed, exit the app. The system instruction indicates
        # that the chatbot should say thanks and goodbye at this point, so we can exit
        # cleanly.
        return END

    elif hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0:
        # Route to `tools` node for any automated tool calls first.
        if any(
            tool["name"] in tool_node.tools_by_name.keys() for tool in msg.tool_calls
        ):
            return "tools"
        else:
            return "ordering"

    else:
        return "human"
    
# Auto-tools will be invoked automatically by the ToolNode
auto_tools = [get_menu]
tool_node = ToolNode(auto_tools)

# Order-tools will be handled by the order node.
order_tools = [add_to_order, confirm_order, get_order, clear_order, place_order]

# The LLM needs to know about all of the tools, so specify everything here.
llm_with_tools = llm.bind_tools(auto_tools + order_tools)


graph_builder = StateGraph(OrderState)

# Nodes
graph_builder.add_node("chatbot", chatbot_with_tools)
graph_builder.add_node("human", human_node)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("ordering", order_node)

# Chatbot -> {ordering, tools, human, END}
graph_builder.add_conditional_edges("chatbot", maybe_route_to_tools)
# Human -> {chatbot, END}
graph_builder.add_conditional_edges("human", maybe_exit_human_node)

# Tools (both kinds) always route back to chat afterwards.
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("ordering", "chatbot")

graph_builder.add_edge(START, "chatbot")
graph_with_order_tools = graph_builder.compile()

config = {"recursion_limit": 100}
state = graph_with_order_tools.invoke({"messages": []}, config)