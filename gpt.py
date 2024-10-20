# llama_api.py
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the GPT-J-6B model and tokenizer
model_name = "EleutherAI/gpt-j-6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)



def generate_text(input_text):    
    # Create a more specific prompt structure for question answering
    prompt = f"{input_text}"
    
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate response with increased token limit and sampling settings
    outputs = model.generate(
        **inputs, 
        max_new_tokens=500,  # Increase the length of generated tokens
        pad_token_id=tokenizer.eos_token_id, 
        do_sample=True, 
        top_p=0.9, 
        top_k=50, 
        temperature=0.8,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Decode the generated response and clean up the output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    
    # Post-process to clean up newlines and spaces
    response = " ".join(response.split())  # Remove multiple spaces/newlines
    response = response.removeprefix(prompt)
    return response

output = generate_text("What is golang?")
print(output)
