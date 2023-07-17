import json
import os

import requests
from flask import Flask, request

app = Flask(__name__)


@app.get("/")
def index():
    # Returns index.html
    return app.send_static_file("index.html")


@app.post("/api")
def api():
    # Get the Image Prompt from the request body
    image_prompt = request.json["image_prompt"]
    response = requests.post(
        "https://api.segmind.com/v1/sd2.1-txt2img",
        headers={
            "x-api-key": os.getenv("API_KEY"),
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "prompt": image_prompt,
            "negative_prompt": "NONE",
            "samples": 1,
            "scheduler": "UniPC",
            "num_inference_steps": 25,
            "guidance_scale": 7.5,
        }),
    )

    # Response is image
    return response.content


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
