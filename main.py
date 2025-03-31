from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Tải mô hình và tokenizer
tokenizer = T5Tokenizer.from_pretrained('transformer_chatbot_tokenizer')
model = T5ForConditionalGeneration.from_pretrained('transformer_chatbot_model')

# Khởi tạo ứng dụng FastAPI
app = FastAPI(title="Chatbot API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

# Định nghĩa schema đầu vào
class ChatRequest(BaseModel):
    message: str

# Hàm xử lý chatbot
def get_response(user_input: str):
    input_ids = tokenizer.encode(user_input, return_tensors='pt', padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=512,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2 
        )
    
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    if response.startswith("answer:"):
        response = response[len("answer:"):].strip()
    
    return response

# API endpoint để chatbot phản hồi
@app.post("/chat")
async def chat(request: ChatRequest):
    response = get_response(request.message)
    return {"response": response}

# Chạy ứng dụng FastAPI với Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
