# Google Gen AI SDK Technical Guide

This repository contains examples and templates for using the **Google Gen AI SDK** (`google-genai`) to interact with Gemini models on Vertex AI. This guide summarizes the technical concepts, setup procedures, and advanced capabilities demonstrated in the provided notebooks.

## 1. Prerequisites & Setup

Before running the code, ensure the following environment requirements are met:

*   **Google Cloud Object**: A generic GCP project with billing enabled.
*   **APIs**: Enable the Vertex AI API (`aiplatform.googleapis.com`).
*   **Python Version**: Tested on Python 3.10+.
*   **Authentication**:
    *   **Colab/Workbench**: Often automated via the environment.
    *   **Local**: Use `gcloud auth application-default login`.

### Installation
Install the SDK and common dependencies:
```bash
pip install --upgrade google-genai pandas
```

## 2. Initialization

To use the SDK with Vertex AI, initialize the client with your Project ID and Location.

```python
from google import genai
import os

PROJECT_ID = "your-project-id"
LOCATION = "us-central1"

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
```

## 3. Core Capabilities

### Text Generation
The most basic usage is sending a text prompt to a model (e.g., `gemini-2.5-flash`).

```python
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What's the largest planet in our solar system?"
)
print(response.text)
```

### Multimodal Prompts
Gemini models can process images, PDFs, and video alongside text. You can pass images as `PIL` objects or GCS URIs.

```python
from google.genai.types import Part

# Using a Cloud Storage URI
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        Part.from_uri(
            file_uri="gs://cloud-samples-data/.../image.png",
            mime_type="image/png"
        ),
        "Write a blog post based on this picture."
    ]
)
```

### System Instructions
You can "steer" the model's behavior/persona using system instructions.

```python
from google.genai.types import GenerateContentConfig

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="I like bagels.",
    config=GenerateContentConfig(
        system_instruction="You are a French translator."
    )
)
```

### Model Configuration
Fine-tune response generation using parameters like `temperature`, `top_p`, `top_k`, and `max_output_tokens`.

```python
config = GenerateContentConfig(
    temperature=0.4,
    top_p=0.95,
    top_k=20,
    candidate_count=1,
    stop_sequences=["STOP!"]
)
```

### Safety Settings
Control the filter thresholds for various harm categories (Hate Speech, Harassment, etc.).

```python
from google.genai.types import SafetySetting, HarmCategory, HarmBlockThreshold

safety_settings = [
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
    )
]
```

## 4. Advanced Features

### Multi-turn Chat
Maintain conversation history automatically using the Chat session object.

```python
chat = client.chats.create(model="gemini-2.5-flash")
response = chat.send_message("Hello!")
response = chat.send_message("Follow up question...")
```

### Function Calling (Tools)
Connect the model to external code tools. The model outputs a **function call** instead of text, which you can then execute.

```python
from google.genai.types import FunctionDeclaration, Tool

get_weather = FunctionDeclaration(name="get_weather", ...)
weather_tool = Tool(function_declarations=[get_weather])

response = client.models.generate_content(
    ...,
    config=GenerateContentConfig(tools=[weather_tool])
)
# Check response.candidates[0].content.parts[0].function_call
```

### Controlled Output (JSON)
Force the model to output structured data (JSON) adhering to a specific schema, defined via Pydantic models or Python dictionaries.

```python
from pydantic import BaseModel

class Recipe(BaseModel):
    name: str
    ingredients: list[str]

response = client.models.generate_content(
    ...,
    config=GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=Recipe
    )
)
```

### Streaming
Receive response chunks as they are generated to improve perceived latency.

```python
for chunk in client.models.generate_content_stream(...):
    print(chunk.text)
```

### Asynchronous Requests
Use `client.aio` for non-blocking calls, useful for high-throughput applications.

```python
response = await client.aio.models.generate_content(...)
```

## 5. Efficiency & Production

### Token Counting
Calculate token usage before sending requests to manage costs and context windows.

*   `count_tokens()`: Returns total token count.
*   `compute_tokens()`: Returns detailed token info (IDs).

### Context Caching
For large inputs (like books or long PDFs) sent repeatedly, use **Context Caching** to reduce costs and latency. You create a cache with a TTL (Time To Live).

```python
cached_content = client.caches.create(
    model="gemini-2.5-flash",
    config=CreateCachedContentConfig(
        contents=[large_pdf_part],
        ttl="3600s"
    )
)

# Use the cache
response = client.models.generate_content(
    ...,
    config=GenerateContentConfig(cached_content=cached_content.name)
)
```

### Batch Prediction
Process large volumes of non-latency-sensitive requests efficiently using Batch jobs. Results are output to Cloud Storage (GCS) or BigQuery.

```python
batch_job = client.batches.create(
    model="gemini-2.5-flash",
    src="gs://bucket/inputs.jsonl",
    config=CreateBatchJobConfig(dest="gs://bucket/outputs/")
)
```

### Embeddings
Generate vector embeddings for text, useful for semantic search and RAG applications.

```python
response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=["Hello world"],
    config=EmbedContentConfig(output_dimensionality=128)
)
```
