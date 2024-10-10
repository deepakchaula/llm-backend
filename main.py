# main.py
# type: ignore
from fastapi import FastAPI, HTTPException, Path, Query, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import ValidationError
from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from models import Conversation, Prompt, QueryRoleType, SUPPORTED_MODELS
from openai import OpenAI
import os
from dotenv import load_dotenv
import uuid
from openai import OpenAI, OpenAIError, AsyncOpenAI
import logging

logger = logging.getLogger("uvicorn.error")

load_dotenv()

app = FastAPI()

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can change this to allow specific origins only
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, PUT, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Initialize OpenAI API key

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)

# Initialize Beanie and MongoDB
@app.on_event("startup")
async def app_init():
    # Initialize database connection
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    await init_beanie(database=client.db_name, document_models=[Conversation])

# Exception handlers
@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "code": 422,
            "message": "Request validation error",
            "details": exc.errors()
        },
    )

@app.exception_handler(ValidationError)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "code": 422,
            "message": "Validation error",
            "details": exc.errors()
        },
    )

# Define SUPPORTED_MODELS if not defined in models.py
SUPPORTED_MODELS = ['gpt-4', 'gpt-3.5-turbo']

# Endpoint to create a new conversation
@app.post(
    "/conversations",
    response_model=dict,
    status_code=201,
    summary="Creates a new Conversation with an LLM model",
    description="""
    A Conversation describes a series of interactions with an LLM model.
    It also contains the properties that will be used to send individual queries
    to the LLM. Chat queries will be anonymized and logged for audit purposes.
    """,
    tags=["Conversations"],
    responses={
        201: {"description": "Successfully created resource with ID"},
        400: {"description": "Invalid parameters provided"},
        500: {"description": "Internal server error"},
    },
)
async def create_conversation(conversation_data: Conversation):
    conversation = Conversation(
        id=uuid.uuid4(),
        name=conversation_data.name,
        params=conversation_data.params,
        tokens=0,
        messages=[]
    )
    await conversation.insert()
    return {"id": str(conversation.id)}  # Ensure id is returned as a string

# Define the new endpoint to retrieve all conversations
@app.get(
    "/conversations",
    response_model=list[Conversation],
    status_code=200,
    summary="Retrieve a list of all conversations",
    description="Retrieves all the conversations created by the user.",
    tags=["Conversations"],
    responses={
        200: {"description": "Successfully retrieved list of conversations"},
        500: {"description": "Internal server error"},
    },
)
async def get_conversations():
    try:
        conversations = await Conversation.find_all().to_list()
        return conversations
    except Exception as e:
        logger.exception("An error occurred while retrieving conversations.")
        raise HTTPException(status_code=500, detail="Internal server error")


# Define the new endpoint to update conversation properties
@app.put(
    "/conversations/{id}",
    response_model=Conversation,
    status_code=200,
    summary="Updates the properties of a conversation",
    description="Update the parameters and properties of a specific conversation.",
    tags=["Conversations"],
    responses={
        200: {"description": "Successfully updated conversation"},
        404: {"description": "Conversation not found"},
        500: {"description": "Internal server error"},
    },
)
async def update_conversation(
    id: str = Path(..., description="Unique conversation ID", regex="^[0-9a-fA-F-]{36}$"),
    conversation_data: Conversation = ...
):
    conversation = await Conversation.get(id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conversation.name = conversation_data.name
    conversation.params = conversation_data.params

    try:
        await conversation.save()
        return conversation
    except Exception as e:
        logger.exception("An error occurred while updating the conversation.")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get(
    "/conversations/{id}",
    response_model=Conversation,
    status_code=200,
    summary="Retrieves the Conversation History",
    description="Retrieves the entire conversation history with the LLM.",
    tags=["Conversations"],
    responses={
        200: {"description": "Successfully retrieved the conversation"},
        404: {"description": "Conversation not found"},
        500: {"description": "Internal server error"},
    },
)
async def get_conversation(
    id: str = Path(..., description="Unique conversation ID", regex="^[0-9a-fA-F-]{36}$")
):
    conversation = await Conversation.get(id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    try:
        return conversation
    except Exception as e:
        logger.exception("An error occurred while retrieving the conversation.")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.delete(
    "/conversations/{id}",
    status_code=204,
    summary="Deletes the Conversation",
    description="Deletes the entire conversation history with the LLM Model.",
    tags=["Conversations"],
    responses={
        204: {"description": "Successfully deleted the conversation"},
        404: {"description": "Conversation not found"},
        500: {"description": "Internal server error"},
    },
)
async def delete_conversation(
    id: str = Path(..., description="Unique conversation ID", regex="^[0-9a-fA-F-]{36}$")
):
    conversation = await Conversation.get(id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    try:
        await conversation.delete()
        return  # 204 No Content is returned automatically by FastAPI when no body is provided
    except Exception as e:
        logger.exception("An error occurred while deleting the conversation.")
        raise HTTPException(status_code=500, detail="Internal server error")


# Endpoint to send a prompt and receive a response from the LLM
@app.post(
    "/queries/{id}",
    response_model=Prompt,
    status_code=201,
    summary="Creates a new Prompt query",
    description="""
    This action sends a new Prompt query to the LLM and returns its response.
    If any errors occur when sending the prompt to the LLM, then a 422 error is raised.
    """,
    tags=["LLM Queries"],
    responses={
        201: {"description": "Successfully created resource with ID"},
        400: {"description": "Invalid parameters provided"},
        404: {"description": "Conversation not found"},
        422: {"description": "Unable to create resource"},
        500: {"description": "Internal server error"},
    },
)
async def create_prompt(
    id: str = Path(..., description="A unique conversation ID", regex="^[0-9a-fA-F-]{36}$"),
    prompt: Prompt = ...,
    model_name: str = Query('gpt-3.5-turbo', description="The name of the model to use")
):
    logger.debug(f"Received request for conversation ID: {id}")

    client = AsyncOpenAI()

    if model_name not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail="Unsupported model selected.")

    conversation = await Conversation.get(id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conversation.messages.append(prompt)

    try:
        response = await client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt.content}])
    except OpenAIError as e:
        logger.error(f"OpenAIError occurred: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("An unexpected error occurred")
        raise HTTPException(status_code=500, detail=str(e))

    assistant_reply = response.choices[0].message.content
    assistant_message = Prompt(role=QueryRoleType.assistant, content=assistant_reply)
    conversation.messages.append(assistant_message)
    conversation.tokens += response.usage.total_tokens
    await conversation.save()

    return assistant_message
