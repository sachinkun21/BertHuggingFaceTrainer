from fastapi import FastAPI
from torch_predict import predict


app = FastAPI(title="NER with HuggingFace and BERT")


@app.get("/")
async def home():
    return "<h2> Connected to BertNER API</h2>"


@app.get("/predictTorch")
async def get_entities_pos(text: str):
    result = predict(text)
    return result


@app.get("/predictONNX")
async def get_entities_pos(text: str):
    return text
