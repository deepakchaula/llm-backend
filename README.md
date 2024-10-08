# LLM Interaction API

## Overview

This application provides an API for interacting with a Language Model (LLM) using OpenAI's GPT-3.5 Turbo. It allows you to create conversations, send prompts, and receive responses from the LLM.

## Requirements

- Python >= 3.8
- FastAPI
- Pydantic
- Beanie
- MongoDB (using Docker)
- Docker
- OpenAI Python Client

## Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/deepakchaula/llm-backend.git
   cd llm-backend

## Setup Instructions

1. **Set Up Environment Variables**

   - Create a `.env` file in the project root:
     ```
     OPENAI_API_KEY=your-api-key-here
     ```
   - Ensure `.env` is listed in `.gitignore`.

2. **Install Dependencies**

   - Install required Python packages:
     ```bash
     pip install -r requirements.txt
     ```

3. **Run the Application**

   - Start the FastAPI server:
     ```bash
     virtualenv --system-site-packages -p python3 ./venv
      python3 -m venv env
      source env/bin/activate
     uvicorn main:app --reload
     ```

4. **Access the API**

   - Visit `http://localhost:8000/docs` to access the interactive API documentation.

## Security Notice

- **Do Not Share Your API Key:** Keep your OpenAI API key confidential. Do not share it in public forums, commit it to version control, or include it in client-side code.
- **Use Environment Variables:** Always load sensitive information like API keys from environment variables or secured configuration files.

## API Usage

- POST /conversations: Create a new conversation.
- GET /conversations: Retrieve all conversations.
- GET /conversations/{id}: Retrieve a specific conversation with its history.
- PUT /conversations/{id}: Update conversation properties.
- DELETE /conversations/{id}: Delete a conversation.

### Creating a Conversation

- Endpoint: `POST /conversations`
- Request Body:

  ```json
  {
    "name": "Your Conversation Name",
    "params": {}
  }