import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()
model_path = "modelex.h5"  # EDIT THIS

classes = ["normal", "phonecall_left", "phonecall_right", "phonehold_left", "phonehold_right"]
model = tf.keras.models.load_model(model_path)


@app.get("/")
async def root():
    return HTMLResponse("""
    <h1>Imagine Texting & Driving API Server</h1>
    Use the endpoint /api/predict?path=<path> to try the API.
    <ul>Examples:
    <li>/api/predict?path=data/test/normal/img_20579.jpg</li>
    <li>/api/predict?path=data/test/normal/img_65547.jpg</li>
    <li>/api/predict?path=data/test/phonecall_left/img_6313.jpg</li>
    </ul>
    In an actual scenario, image URLs would be passed in as a query parameter, not a file path.
    """)


@app.get("/api/predict")
async def predict(path: str):
    img = tf.keras.utils.load_img(path, target_size=(256, 256))
    img_array = tf.expand_dims(tf.keras.utils.img_to_array(img), 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    return {"class": classes[np.argmax(score)], "confidence": 100 * np.max(score)}

