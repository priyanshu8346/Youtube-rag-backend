# YouTube RAG Backend

This is the backend for the YouTube RAG (Retrieval-Augmented Generation) project. It provides API endpoints for transcript retrieval and AI-powered question answering using Flask and OpenAI.

**Frontend:** [Live Demo](https://priyanshu8346.github.io/Youtube-rag-frontend/)

---

## Features
- Fetches YouTube video transcripts (if available)
- Embeds and stores transcript chunks for semantic search
- Answers user questions using OpenAI LLMs
- REST API endpoints for easy integration

## How It Works
1. The frontend sends a YouTube video URL to the backend.
2. The backend fetches the transcript and prepares it for semantic search.
3. User questions are answered using the transcript and OpenAI.

## API Endpoints
- `POST /load_video` — Loads a YouTube video's transcript
- `POST /chat` — Ask a question about the loaded video

## Getting Started

1. Clone this repository:
	```sh
	git clone https://github.com/priyanshu8346/Youtube-rag-backend.git
	cd Youtube-rag-backend/backend
	```
2. Install dependencies:
	```sh
	pip install -r requirements.txt
	```
3. Set your OpenAI API key in a `.env` file:
	```env
	OPENAI_API_KEY=sk-...
	```
4. Run the Flask app:
	```sh
	python app.py
	```

## Deployment
You can deploy this backend to any cloud platform (Render, Heroku, AWS, etc.) for public access.

## Tech Stack
- Python
- Flask
- youtube-transcript-api
- langchain
- OpenAI

## Demo Section

> **See the full project in action with the [Frontend Live Demo](https://priyanshu8346.github.io/Youtube-rag-frontend/)!**

---

## License
[MIT](LICENSE)