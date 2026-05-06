import os
import re
import json
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import gradio as gr


# ---------------------------
# Paths & basic config
# ---------------------------

DATA_PATH = "conversations.csv"
DATA_DIR = "data"
MESSAGES_PATH = os.path.join(DATA_DIR, "messages.parquet")
MESSAGE_EMB_PATH = os.path.join(DATA_DIR, "message_embeddings.npy")
TOPICS_PATH = os.path.join(DATA_DIR, "topics.parquet")
TOPIC_EMB_PATH = os.path.join(DATA_DIR, "topic_embeddings.npy")
CHECKPOINTS_PATH = os.path.join(DATA_DIR, "checkpoints.parquet")
PERSONA_PATH = os.path.join(DATA_DIR, "persona.json")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Topic segmentation config
TOPIC_SIM_THRESHOLD = 0.6   # lower similarity than this => new topic
MIN_TOPIC_LENGTH = 5        # minimum messages per topic

# Retrieval config
TOP_K_MESSAGES = 10
TOP_K_TOPICS = 5


# ---------------------------
# Utils
# ---------------------------

def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


# ---------------------------
# Step 1: Load & split CSV into messages
# ---------------------------

def load_raw_conversations() -> pd.DataFrame:
    """
    Load conversations.csv. If there is only one column, treat it
    as 'conversation'. If there are multiple, adjust as needed.
    """
    df = pd.read_csv(DATA_PATH)
    if df.shape[1] == 1:
        df.columns = ["conversation"]
    elif "conversation" not in df.columns:
        # Fall back: use the first column as conversation text
        df = df[[df.columns[0]]]
        df.columns = ["conversation"]
    return df


def split_conversations_to_messages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Each row in df['conversation'] is a multi-line text with user messages.
    We split by newline and try to detect "User 1:" / "User 2:" labels.
    We also create a global_id to preserve strict chronological order.
    """
    rows = []
    global_id = 0

    for conv_id, row in tqdm(df.iterrows(), total=len(df), desc="Splitting messages"):
        text = str(row["conversation"])
        # Split on newlines
        lines = re.split(r"\r?\n+", text.strip())
        local_id = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect "User 1: message" format
            m = re.match(r"^(User\s*\d+)\s*:\s*(.*)$", line, flags=re.IGNORECASE)
            if m:
                speaker = m.group(1).strip()
                content = m.group(2).strip()
            else:
                speaker = "Unknown"
                content = line

            if not content:
                continue

            rows.append(
                {
                    "global_id": global_id,
                    "conv_id": conv_id,
                    "local_id": local_id,
                    "speaker": speaker,
                    "text": content,
                }
            )
            global_id += 1
            local_id += 1

    messages_df = pd.DataFrame(rows)
    return messages_df


def build_or_load_messages() -> pd.DataFrame:
    ensure_data_dir()
    if os.path.exists(MESSAGES_PATH):
        return pd.read_parquet(MESSAGES_PATH)

    raw_df = load_raw_conversations()
    messages_df = split_conversations_to_messages(raw_df)
    messages_df.to_parquet(MESSAGES_PATH, index=False)
    return messages_df


# ---------------------------
# Step 2: Embeddings for messages
# ---------------------------

def build_or_load_message_embeddings(messages_df: pd.DataFrame, model: SentenceTransformer) -> np.ndarray:
    ensure_data_dir()
    if os.path.exists(MESSAGE_EMB_PATH):
        return np.load(MESSAGE_EMB_PATH)

    texts = messages_df["text"].tolist()
    embeddings_list = []

    for i in tqdm(range(0, len(texts), 64), desc="Encoding messages"):
        batch = texts[i : i + 64]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        embeddings_list.append(emb)

    embeddings = np.vstack(embeddings_list)
    np.save(MESSAGE_EMB_PATH, embeddings)
    return embeddings


# ---------------------------
# Step 3: Topic segmentation
# ---------------------------

def detect_topic_boundaries(message_embeddings: np.ndarray) -> List[int]:
    """
    Use cosine similarity between consecutive messages.
    If similarity < TOPIC_SIM_THRESHOLD, we start a new topic at that index.
    Returns list of boundary indices (start indices of topics).
    """
    sims = cosine_similarity(message_embeddings[:-1], message_embeddings[1:])
    sims = sims.diagonal()
    boundaries = [0]  # first topic starts at 0

    for i, s in enumerate(sims, start=1):
        if s < TOPIC_SIM_THRESHOLD:
            boundaries.append(i)

    # Ensure sorted & unique
    boundaries = sorted(set(boundaries))
    return boundaries


def build_topic_segments(messages_df: pd.DataFrame, message_embeddings: np.ndarray) -> pd.DataFrame:
    ensure_data_dir()
    if os.path.exists(TOPICS_PATH):
        return pd.read_parquet(TOPICS_PATH)

    boundaries = detect_topic_boundaries(message_embeddings)
    boundaries = sorted(boundaries)

    segments = []
    n = len(messages_df)

    # Add end boundary
    if boundaries[-1] != n:
        boundaries.append(n)

    topic_id = 0
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]  # exclusive
        length = end - start
        if length < MIN_TOPIC_LENGTH:
            # merge small segments into previous if possible
            if segments:
                segments[-1]["end_global_id"] = messages_df.iloc[end - 1]["global_id"]
            else:
                # if first is small, still keep it
                pass
            continue

        start_gid = messages_df.iloc[start]["global_id"]
        end_gid = messages_df.iloc[end - 1]["global_id"]

        segment_texts = messages_df.iloc[start:end]["text"].tolist()
        summary, keywords = summarize_segment_with_tfidf(segment_texts)

        segments.append(
            {
                "topic_id": topic_id,
                "start_global_id": int(start_gid),
                "end_global_id": int(end_gid),
                "summary": summary,
                "keywords": ", ".join(keywords),
            }
        )
        topic_id += 1

    topics_df = pd.DataFrame(segments)
    topics_df.to_parquet(TOPICS_PATH, index=False)
    return topics_df


def summarize_segment_with_tfidf(texts: List[str], top_k_keywords: int = 5) -> Tuple[str, List[str]]:
    """
    Very lightweight summarization:
    - Use TF-IDF to get top keywords for the segment.
    - Summary = first sentence + list of keywords.
    """
    if not texts:
        return "", []

    # Use all texts as a single "document" for keyword extraction
    joined = " ".join(texts)

    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    tfidf_matrix = vectorizer.fit_transform([joined])
    scores = tfidf_matrix.toarray()[0]
    feature_names = np.array(vectorizer.get_feature_names_out())

    top_indices = np.argsort(scores)[::-1][:top_k_keywords]
    keywords = feature_names[top_indices].tolist()

    first_sentence = texts[0]
    summary = f"Mainly about: {', '.join(keywords)}. Example: \"{first_sentence}\""
    return summary, keywords


def build_or_load_topic_embeddings(topics_df: pd.DataFrame, model: SentenceTransformer) -> np.ndarray:
    ensure_data_dir()
    if os.path.exists(TOPIC_EMB_PATH):
        return np.load(TOPIC_EMB_PATH)

    texts = topics_df["summary"].tolist()
    if not texts:
        embeddings = np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
        np.save(TOPIC_EMB_PATH, embeddings)
        return embeddings

    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    np.save(TOPIC_EMB_PATH, embeddings)
    return embeddings


# ---------------------------
# Step 4: 100-message checkpoints
# ---------------------------

def build_or_load_checkpoints(messages_df: pd.DataFrame) -> pd.DataFrame:
    ensure_data_dir()
    if os.path.exists(CHECKPOINTS_PATH):
        return pd.read_parquet(CHECKPOINTS_PATH)

    checkpoints = []
    max_gid = messages_df["global_id"].max()
    checkpoint_id = 0

    # Message IDs are 0..max_gid (assuming continuous)
    start_gid = 0
    while start_gid <= max_gid:
        end_gid = start_gid + 99
        mask = (messages_df["global_id"] >= start_gid) & (messages_df["global_id"] <= end_gid)
        chunk = messages_df[mask]
        if chunk.empty:
            break

        texts = chunk["text"].tolist()
        summary, keywords = summarize_segment_with_tfidf(texts)

        checkpoints.append(
            {
                "checkpoint_id": checkpoint_id,
                "start_global_id": int(start_gid),
                "end_global_id": int(chunk["global_id"].max()),
                "summary": summary,
                "keywords": ", ".join(keywords),
            }
        )

        checkpoint_id += 1
        start_gid += 100

    checkpoints_df = pd.DataFrame(checkpoints)
    checkpoints_df.to_parquet(CHECKPOINTS_PATH, index=False)
    return checkpoints_df


# ---------------------------
# Step 5: Persona extraction (rule-based)
# ---------------------------

def extract_persona(messages_df: pd.DataFrame) -> Dict:
    """
    Very simple heuristic-based persona extraction.
    You can tune the keyword lists for your data.
    """
    texts = messages_df["text"].astype(str).tolist()
    lower_texts = [t.lower() for t in texts]

    # Habits
    late_night_keywords = ["2 am", "3 am", "late night", "midnight"]
    food_keywords = ["biryani", "pizza", "burger", "coffee", "tea", "breakfast", "lunch", "dinner"]
    study_work_keywords = ["interview", "coding", "office", "exam", "study", "project"]

    late_night_count = sum(any(k in t for k in late_night_keywords) for t in lower_texts)
    food_count = sum(any(k in t for k in food_keywords) for t in lower_texts)
    study_work_count = sum(any(k in t for k in study_work_keywords) for t in lower_texts)

    habits = []
    if late_night_count > 3:
        habits.append("Possibly a late sleeper (many mentions of late night timings).")
    if food_count > 3:
        habits.append("Frequently talks about food (biryani, tea, snacks, meals).")
    if study_work_count > 3:
        habits.append("Often discusses study/work topics (coding, exams, office, projects).")

    # Personal facts: look for words like "birthday", "married", "college", etc.
    personal_fact_keywords = ["birthday", "married", "wedding", "college", "joined", "resigned"]
    personal_facts = []
    for t in texts:
        lt = t.lower()
        if any(k in lt for k in personal_fact_keywords):
            personal_facts.append(t)
    personal_facts = personal_facts[:20]  # keep short

    # Personality traits via style
    emojis = ["😂", "😅", "🤣", "😊", "😁"]
    apology_words = ["sorry", "apologize", "forgive"]
    anger_words = ["angry", "frustrated", "fed up"]

    emoji_count = sum(sum(ch in msg for ch in emojis) for msg in texts)
    apology_count = sum(any(w in msg.lower() for w in apology_words) for msg in texts)
    anger_count = sum(any(w in msg.lower() for w in anger_words) for msg in texts)

    personality_traits = []
    if emoji_count > 20:
        personality_traits.append("Expressive / playful (uses a lot of emojis).")
    if apology_count > 5:
        personality_traits.append("Polite / considerate (frequent apologies).")
    if anger_count > 5:
        personality_traits.append("Sometimes expresses frustration or anger.")

    # Communication style
    lengths = [len(t.split()) for t in texts]
    avg_len = np.mean(lengths) if lengths else 0
    short_msgs_ratio = sum(l <= 5 for l in lengths) / len(lengths) if lengths else 0

    if avg_len <= 7:
        msg_length_style = "Mostly short messages."
    elif avg_len >= 20:
        msg_length_style = "Often sends long, detailed messages."
    else:
        msg_length_style = "Mix of short and medium-length messages."

    question_marks = sum("?" in t for t in texts)
    question_ratio = question_marks / len(texts) if texts else 0
    if question_ratio > 0.3:
        question_style = "Asks many questions (curious / exploratory)."
    else:
        question_style = "Does not ask questions very frequently."

    communication_style = {
        "message_length": msg_length_style,
        "question_pattern": question_style,
        "emoji_usage": f"Total emoji usage count in dataset: {emoji_count}",
    }

    persona = {
        "habits": habits,
        "personal_facts_examples": personal_facts,
        "personality_traits": personality_traits,
        "communication_style": communication_style,
    }
    return persona


def build_or_load_persona(messages_df: pd.DataFrame) -> Dict:
    ensure_data_dir()
    if os.path.exists(PERSONA_PATH):
        with open(PERSONA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    persona = extract_persona(messages_df)
    with open(PERSONA_PATH, "w", encoding="utf-8") as f:
        json.dump(persona, f, ensure_ascii=False, indent=2)
    return persona


# ---------------------------
# Step 6: RAG Bot class (retrieval + answering)
# ---------------------------

class RAGBot:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # Build/load base structures
        self.messages_df = build_or_load_messages()
        self.message_embeddings = build_or_load_message_embeddings(self.messages_df, self.model)

        self.topics_df = build_topic_segments(self.messages_df, self.message_embeddings)
        self.topic_embeddings = build_or_load_topic_embeddings(self.topics_df, self.model)

        self.checkpoints_df = build_or_load_checkpoints(self.messages_df)
        self.persona = build_or_load_persona(self.messages_df)

    # ---- Retrieval helpers ----

    def retrieve_messages(self, query: str, top_k: int = TOP_K_MESSAGES) -> pd.DataFrame:
        if len(self.message_embeddings) == 0:
            return self.messages_df.iloc[0:0]

        q_emb = self.model.encode([query], convert_to_numpy=True)
        sims = cosine_similarity(q_emb, self.message_embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        results = self.messages_df.iloc[top_idx].copy()
        results["score"] = sims[top_idx]
        results = results.sort_values("score", ascending=False)
        return results

    def retrieve_topics(self, query: str, top_k: int = TOP_K_TOPICS) -> pd.DataFrame:
        if self.topics_df.empty or len(self.topic_embeddings) == 0:
            return self.topics_df.iloc[0:0]

        q_emb = self.model.encode([query], convert_to_numpy=True)
        sims = cosine_similarity(q_emb, self.topic_embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        results = self.topics_df.iloc[top_idx].copy()
        results["score"] = sims[top_idx]
        results = results.sort_values("score", ascending=False)
        return results

    # ---- Answer construction ----

    def build_rag_answer(self, query: str) -> str:
        msg_hits = self.retrieve_messages(query)
        topic_hits = self.retrieve_topics(query)

        lines = []
        lines.append(f"Query: {query}")
        lines.append("")

        # Topic summaries
        lines.append("Top related topic checkpoints:")
        if topic_hits.empty:
            lines.append("- (No topics found.)")
        else:
            for _, row in topic_hits.iterrows():
                lines.append(
                    f"- Topic {row['topic_id']} (messages {row['start_global_id']}–{row['end_global_id']}): "
                    f"{row['summary']} (score={row['score']:.3f})"
                )
        lines.append("")

        # Message chunks
        lines.append("Top related message snippets:")
        if msg_hits.empty:
            lines.append("- (No messages found.)")
        else:
            for _, row in msg_hits.head(8).iterrows():
                lines.append(
                    f"- [{row['score']:.3f}] {row['speaker']}: {row['text']}"
                )

        lines.append("")
        lines.append(
            "This answer is built by retrieving topic summaries and message snippets that are semantically "
            "close to your query using embeddings and cosine similarity."
        )
        return "\n".join(lines)

    def build_persona_answer(self, query: str) -> str:
        """
        Use the persona JSON + a bit of retrieval context to answer
        questions like:
        - What kind of person is this user?
        - What are their habits?
        - How do they talk?
        """
        # Basic RAG context (optional)
        msg_hits = self.retrieve_messages("persona habits character style tone")
        lines = []

        lines.append("Persona overview (based on actual conversation patterns):")
        lines.append("")

        # Habits
        lines.append("Habits:")
        if self.persona["habits"]:
            for h in self.persona["habits"]:
                lines.append(f"- {h}")
        else:
            lines.append("- No strong habits detected from the messages.")

        lines.append("")
        lines.append("Personality traits:")
        if self.persona["personality_traits"]:
            for t in self.persona["personality_traits"]:
                lines.append(f"- {t}")
        else:
            lines.append("- Not enough signals to infer strong personality traits.")

        lines.append("")
        lines.append("Communication style:")
        for k, v in self.persona["communication_style"].items():
            lines.append(f"- {k}: {v}")

        if self.persona["personal_facts_examples"]:
            lines.append("")
            lines.append("Example personal fact-related messages (truncated):")
            for ex in self.persona["personal_facts_examples"][:5]:
                lines.append(f"- {ex}")

        if not msg_hits.empty:
            lines.append("")
            lines.append("Some example messages used as context for persona:")
            for _, row in msg_hits.head(5).iterrows():
                lines.append(f"- {row['speaker']}: {row['text']}")

        return "\n".join(lines)

    # ---- Main public API ----

    def answer(self, query: str) -> str:
        q_lower = query.lower()

        # Simple detection of persona questions
        persona_triggers = [
            "what kind of person",
            "what kind of a person",
            "what are their habits",
            "what are his habits",
            "what are her habits",
            "how do they talk",
            "how does he talk",
            "how does she talk",
            "describe the user",
            "describe this user",
            "persona",
        ]
        if any(t in q_lower for t in persona_triggers):
            return self.build_persona_answer(query)
        else:
            return self.build_rag_answer(query)


# ---------------------------
# Step 7: Gradio Chatbot UI
# ---------------------------

rag_bot = RAGBot()


def chat_fn(user_message, history):
    if not user_message or not user_message.strip():
        return "", history

    answer = rag_bot.answer(user_message.strip())
    history = history + [[user_message, answer]]
    return "", history


with gr.Blocks() as demo:
    gr.Markdown("# Conversation RAG Chatbot with Topics & Persona (Offline)")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Ask a question about the conversations or the user persona")
    clear_btn = gr.Button("Clear")

    msg.submit(chat_fn, [msg, chatbot], [msg, chatbot])
    clear_btn.click(lambda: ("", []), None, [msg, chatbot])


if __name__ == "__main__":
    demo.launch()