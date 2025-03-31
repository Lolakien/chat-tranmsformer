from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Tải mô hình và tokenizer
tokenizer = T5Tokenizer.from_pretrained('transformer_chatbot_tokenizer')
model = T5ForConditionalGeneration.from_pretrained('transformer_chatbot_model')

def chatbot():
    print("Chào bạn! Tôi là chatbot AI hỗ trợ ôn luyện TOEIC.")
    while True:
        user_input = input("Bạn: ").strip()
        if user_input.lower() == 'thoát':
            print("Chatbot: Tạm biệt!")
            break
        
        # Thêm tiền tố cho câu hỏi
        input_text = user_input
        
        # Tokenize câu hỏi của người dùng
        input_ids = tokenizer.encode(input_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        # Kiểm tra token hóa
        print(f"Tokenized input: {tokenizer.convert_ids_to_tokens(input_ids[0])}")
        
        # Tạo câu trả lời từ mô hình
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_length=512,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=2 
            )

        print(f"Output IDs: {output_ids}")
        
        # Giải mã câu trả lời
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Kiểm tra giải mã
        print(f"Decoded response: {response}")
        
        # Loại bỏ tiền tố "answer:" nếu có
        if response.startswith("answer:"):
            response = response[len("answer:"):].strip()
        
        print(f"Chatbot: {response}")

# Chạy chatbot
chatbot()