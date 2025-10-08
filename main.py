import base64
import json
import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI()


# Initialize Gemini client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

class ImageRequest(BaseModel):
    image_url: str


@app.post("/analyze_image")
def analyze_image(req: ImageRequest):
    try:
        # Download the image from URL
        response = requests.get(req.image_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download image")

        image_bytes = response.content

        # Convert image to Base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Prepare Gemini input
        model = "gemini-2.5-pro"
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part(text=(
                        "For the following image, identify the animal in it, give its scientific and common name, "
                        "give a very brief description of the animal,"
                        " and on a scale of 1 to 10, give the level of threat it represents to man. "
                        "Return a JSON array where each item includes the fields: "
                        "`common_name`, `scientific_name`, `description`, `threat`."
                    )),
                    types.Part(inline_data=types.Blob(
                        mime_type="image/jpeg",
                        data=image_bytes,
                    )),
                    types.Part(text="Only output valid JSON — no explanations or extra text.")
                ],
            ),
        ]

        # Send to Gemini API
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            ),
        )

        # Parse and return the JSON
        data = json.loads(response.text)
        return {"result": data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
