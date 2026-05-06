Conversation RAG Bot 
1. What this project does

This project takes conversations.csv, where each row is one day’s conversation, and turns it into a chatbot.

It does the following:
- Splits each row into individual messages in correct time order.
- Builds topic checkpoints: segments of consecutive messages where the topic stays similar, each with its own summary.
- Builds 100‑message checkpoints: summaries for every 100 messages, independent of topics.
- Extracts a persona JSON for the user: habits, personality traits, and communication style, based only on real messages.
- Provides a Gradio chatbot that:
  - For normal questions, uses RAG (retrieves topic summaries and message snippets from your data).
  - For persona questions like “What kind of person is this user?” or “How do they talk?”, uses the persona JSON plus some retrieved example messages. [en.wikipedia](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)



2. How to run (from zero)

Step 1: Folder setup

Create a folder called conversation-rag-bot and put these files inside:

conversations.csv  
requirements.txt  
rag_chatbot.py  

Step 2: requirements.txt

Put this content inside requirements.txt:

pandas  
numpy  
sentence-transformers  
scikit-learn  
tqdm  
gradio  
pyarrow  

Step 3: Create and activate virtual environment

Open VS Code terminal in the conversation-rag-bot folder and run:

python -m venv .venv  

On Windows:

.venv\Scripts\activate  

On Linux or macOS:

source .venv/bin/activate  

Then install dependencies:

pip install -r requirements.txt  

Step 4: Run the chatbot

With the virtual environment active (you should see (.venv) in the terminal prompt), run:

python rag_chatbot.py  

On first run it will:
- Download the sentence-transformers model from Hugging Face.
- Split conversations into messages.
- Build embeddings, topics, checkpoints, and persona. [milvus](https://milvus.io/ai-quick-reference/what-is-cosine-similarity-and-how-is-it-used-with-sentence-transformer-embeddings-to-measure-sentence-similarity)

At the end it will print a local URL like:

Running on local URL: http://127.0.0.1:7860  

Open that URL in your browser to use the chatbot. [gradio](https://www.gradio.app/guides/quickstart)



3. How the logic works (short explanation)

3.1 Topic checkpoints

The code converts each row into messages and gives each message a global_id to keep exact chronological order.

For topic detection:
- It creates a sentence embedding for each message using the all-MiniLM-L6-v2 model.
- It computes cosine similarity between each message and the next one.
- When the similarity drops below a threshold (for example 0.6), it marks that point as a topic change. [github](https://github.com/saeedabc/llm-text-tiling)

Each topic segment is the continuous block of messages between two topic boundaries. For each segment, it:
- Uses TF‑IDF to find the top keywords.
- Builds a short summary using those keywords and the first message as an example. [stackoverflow](https://stackoverflow.com/questions/68459166/python-using-tf-idf-to-summarise-dataframe-text-column)

3.2 100‑message checkpoints

Independently of topics, the code also:
- Takes messages with global_id 0 to 99, 100 to 199, 200 to 299, and so on.
- For each 100‑message block, it builds a summary using the same TF‑IDF keyword logic.

These are the 100‑message checkpoints required by the task.

3.3 Persona JSON

The code scans all messages and looks for patterns:

- Habits:
  It counts messages that mention late-night times, food words, and study or work words. If counts are high, it writes habit sentences like “Possibly a late sleeper”, “Frequently talks about food”, or “Often discusses study or work”.

- Personal facts examples:
  It collects messages containing words like “birthday”, “wedding”, “college” and keeps them as example lines (evidence), not as guesses.

- Personality traits:
  It counts emoji usage, apology words, and anger words. Based on thresholds, it adds traits like “Expressive or playful”, “Polite or considerate”, or “Sometimes expresses frustration or anger”.

- Communication style:
  It measures average message length and how often messages contain a question mark. From this it decides whether the user mostly sends short messages or long detailed ones, and whether they ask many questions.

All of this is saved to data/persona.json as a structured JSON object.

3.4 Retrieval and answering (RAG)

For any query:
- The code embeds the query using the same sentence-transformers model.
- It compares the query embedding with:
  - All message embeddings.
  - All topic summary embeddings.
- It uses cosine similarity to get the most relevant messages and topics. [en.wikipedia](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)

For normal questions:
- It returns an answer text that includes:
  - The top related topic summaries with their message ranges.
  - The top related message snippets with similarity scores.
- This shows exactly which parts of the conversation were used.

For persona-type questions:
- It uses simple keyword checks on the query (for phrases like “what kind of person”, “what are their habits”, “how do they talk”).
- If matched, it returns a persona-style answer built from persona.json plus a few related messages as context.

This satisfies:
- Correct chronological topic splitting using embeddings and similarity.
- Relevant retrieval using semantic search (no random chunks).
- Persona based only on signals in the data.
- A working end-to-end chatbot that uses both RAG and persona.
