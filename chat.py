from transformers import pipeline

def chat_with_model(model_id, prompt, temperature=0.7, max_tokens=256):
    chatbot = pipeline("text-generation", model=model_id)
    result = chatbot(prompt, max_length=max_tokens, temperature=temperature)
    return result[0]['generated_text']