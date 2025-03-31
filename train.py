from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from torch.utils.data import DataLoader, Dataset
import json

# Tải tokenizer và mô hình viT5
tokenizer = T5Tokenizer.from_pretrained("VietAI/vit5-base")
model = T5ForConditionalGeneration.from_pretrained("VietAI/vit5-base")

# Tải dữ liệu đã định dạng
with open('formatted_data1.json', 'r', encoding='utf-8') as f:
    formatted_data = json.load(f)

# Chuẩn bị dữ liệu
inputs = []
targets = []

for intent in formatted_data['intents']:
    for pattern in intent['patterns']:
        inputs.append(pattern)  # Câu hỏi
        targets.append(intent['responses'][0])  # Chọn câu trả lời đầu tiên

# Tokenize dữ liệu
input_encodings = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True, max_length=512)
target_encodings = tokenizer(targets, return_tensors='pt', padding=True, truncation=True, max_length=512)

# Huấn luyện mô hình
class ChatbotDataset(Dataset):
    def __init__(self, input_encodings, target_encodings):
        self.input_encodings = input_encodings
        self.target_encodings = target_encodings

    def __len__(self):
        return len(self.input_encodings['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_encodings['input_ids'][idx],
            'attention_mask': self.input_encodings['attention_mask'][idx],
            'labels': self.target_encodings['input_ids'][idx]
        }

dataset = ChatbotDataset(input_encodings, target_encodings)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Chuyển mô hình sang chế độ huấn luyện
model.train()

# Huấn luyện mô hình
for epoch in range(3):  # Huấn luyện trong 3 epoch
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(input_ids=batch['input_ids'], 
                        attention_mask=batch['attention_mask'], 
                        labels=batch['labels'])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f'Epoch: {epoch}, Loss: {loss.item()}')

# Lưu mô hình
model.save_pretrained('transformer_chatbot_model')
tokenizer.save_pretrained('transformer_chatbot_tokenizer')