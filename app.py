from flask import Flask, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# ------------------ ENV SETUP ------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY is missing in .env file")

# ------------------ APP SETUP ------------------
app = Flask(__name__)

embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, openai_api_key=OPENAI_API_KEY)

vector_store = None
retriever = None

# Prompt template
prompt = PromptTemplate(
    template="""
    You are a helpful assistant answering questions based on YouTube video transcripts.
    If context is not enough, reply with:
    "I don't have enough context to answer this query."

    Context:
    {context}

    Query:
    {query}
    """,
    input_variables=["context", "query"]
)

parser = StrOutputParser()

# ------------------ ROUTES ------------------

@app.route("/load_video", methods=["POST"])
def load_video():
    """
    Loads a YouTube video's transcript into FAISS vector store
    """
    global vector_store, retriever

    data = request.get_json()
    video_id = data.get("video_id")

    if not video_id:
        return jsonify({"error": "video_id is required"}), 400

    try:
        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en", "hi"])
        transcript = " ".join(chunk.text for chunk in transcript_list)
    except TranscriptsDisabled:
        return jsonify({"error": "No transcript available"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Split transcript
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents([transcript])

    # Create vector store + retriever
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    return jsonify({"message": "Transcript loaded successfully", "chunks": len(chunks)})


@app.route("/chat", methods=["POST"])
def chat():
    """
    Takes user query and returns LLM response based on transcript
    """
    global retriever

    if retriever is None:
        return jsonify({"error": "No video loaded. Call /load_video first."}), 400

    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "query is required"}), 400

    # Retrieve context
    retrieved_docs = retriever.get_relevant_documents(query)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Format prompt
    final_prompt = prompt.format(context=context, query=query)

    # Run model
    try:
        response = llm.invoke(final_prompt)
        return jsonify({"answer": response.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------ RUN ------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
