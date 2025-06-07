# ☕ LumaBot — Cafe Ordering Assistant

LumaBot is an interactive cafe ordering assistant built using **LangGraph** and **LangChain**, powered by Google's **Gemini 2.0 Flash** model.

## 🧠 Overview

LumaBot guides users through the process of placing an order at a virtual cafe. It maintains a conversational state, integrates structured tools for menu access and order management, and follows strict behavioral rules via system prompts.

### 🔧 Key Features

- **Conversational Memory:** Maintains full message history during interaction.
- **Tool-Driven Actions:**
  - `get_menu` — shows available drinks and modifiers.
  - `add_to_order`, `clear_order`, `get_order`, `confirm_order`, `place_order` — manage the ordering flow.
- **LLM Integration:** Uses Gemini 2.0 Flash via `langchain_google_genai`.
- **Structured State:** Tracks order items and completion flag in a custom `OrderState` type.
- **Routing Graph:** LangGraph manages chatbot → tools → human interaction seamlessly via conditional nodes.

### 🏁 Conversation Flow

1. Welcome message from LumaBot.
2. User requests drinks or info.
3. LumaBot verifies items via menu, builds order.
4. User confirms order before submission.
5. Final order is placed and ETA is returned.
