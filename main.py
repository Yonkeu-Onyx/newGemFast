import base64
import json
import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types
# from dotenv import load_dotenv

app = FastAPI()

# load_dotenv()

#Try git
# Initialize Gemini client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

maps_api_key = os.environ.get("MAPS_API_KEY")
maps_base_url = "https://maps.googleapis.com/maps/api/geocode/json?"
headers = {
    "Content-Type": "application/json",
}

class ImageRequest(BaseModel):
    image_url: str
    
class LocationRequest(BaseModel):
    latitude: float
    longitude: float


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

@app.post("/getcity")
def get_city(location : LocationRequest):
    print(f'Latitude: {location.latitude}')
    print(f'Longitude: {location.longitude}')
    response = requests.get(f'{maps_base_url}latlng={location.latitude},{location.longitude}&key={maps_api_key}', headers=headers)
    data = response.json()
    city_name = extract_locality_long_name(data)

    return city_name


def extract_locality_long_name(data):
    """
    Extracts the long_name from address_components where types contains 
    both 'locality' and 'political'.
    Returns the first match found or None if not found.
    """
    for result in data.get("results", []):
        for component in result.get("address_components", []):
            types = component.get("types", [])
            if "locality" in types and "political" in types:
                return component.get("long_name")
    return None