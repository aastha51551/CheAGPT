from fastapi import FastAPI, Request
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = FastAPI()
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

@app.post("/CheAGPT")
async def chat(request: Request):
    data = await request.json()
    input_text = data['text']
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}


