import base64
import json
import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI()

#Try git
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
                        "Pour l'image suivante, identifie l'animal qui s'y trouve, indique son nom scientifique" 
                        " et son nom commun, donne une très brève description de l'animal, et, " 
                        "sur une échelle de 0.1 à 1.0 (donne une valeur décimale), indique le niveau de menace qu'il " 
                        "représente pour l'homme. Donne aussi un label correspondant pour le niveau de dangerosité pamis les"
                        "options suivantes : 'faible'(0.0-0.3), 'modéré'(0.4-0.6), 'élevé'(0.7-0.9), 'critique'(0.9-1.0). " 
                        "Propose également une méthode de lutte appropriée contre ce nuisible."
                        "Retourne un tableau JSON où chaque élément comprend les champs suivants :"
                        "common_name, scientific_name, description, threat, label, control."
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
