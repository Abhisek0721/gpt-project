from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the GPT-2 model and tokenizer (try using a larger model like gpt2 or gpt2-medium)
# Load light weight model "distilgpt2"
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)


app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate_text():
    input_text = request.json.get("input_text")
    if not input_text:
        return jsonify({"message": "input_text is missing in payload!"})

    # Create a more specific prompt structure for question answering
    prompt = f"{input_text}"
    
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate response with increased token limit and sampling settings
    outputs = model.generate(
        **inputs, 
        max_new_tokens=100,  # Increase the length of generated tokens (Recommended between 100 - 500)
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
    return jsonify({"response": response})

if __name__ == "__main__":
    print("Running on port: 5000")
    app.run(host="0.0.0.0", port=5000)


