from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def generate_response(prompt, model, tokenizer, max_length=50, temperature=0.9, top_k=50, top_p=0.9):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = input_ids.clone().detach()
    outputs = model.generate(
        input_ids, 
        max_length=max_length, 
        num_return_sequences=1, 
        pad_token_id=tokenizer.eos_token_id,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        attention_mask=attention_mask
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def chat():
    print("Chatbot: Hello! How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Chatbot: Goodbye!")
            break
        response = generate_response(user_input, model, tokenizer)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chat()