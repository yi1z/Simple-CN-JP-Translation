from modelscope import AutoModelForCausalLM, AutoTokenizer
import os

model_name_or_path = "Hunyuan-MT-7B"

print(f"Loading model from {model_name_or_path}")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

print("Model loaded")

input_text = input("Enter the text to translate: ")

messages = [
    {"role": "user", "content": f"Translate the following segment into Japanese, without additional explanation.\n\n{input_text}"},
]

print("Tokenizing input")
tokenized_chat = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    return_tensors="pt"
)

print("Generating output")

outputs = model.generate(tokenized_chat.to(model.device), max_new_tokens=2048)
output_text = tokenizer.decode(outputs[0])

print("Output:")
# format output
output_text = output_text.replace("<|eos|>", "").replace("<|startoftext|>", "").replace("<|extra_0|>", "").replace("<|extra_4|>", "")
print(output_text)