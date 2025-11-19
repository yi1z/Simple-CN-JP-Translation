from modelscope import AutoModelForCausalLM, AutoTokenizer
import os
import torch


language = input("Enter the language you want to translate to: ")
language = language.lower()

model_name_or_path = input("Enter the model name or path: ")
if model_name_or_path == "":
    model_name_or_path = "Hunyuan-MT-7B"
    print("Using default model: Hunyuan-MT-7B")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model from {model_name_or_path} on {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)

print("Model loaded")

input_text = ""
curr_text = input("Enter the text to translate (type 'eos' to end): ")
while curr_text != "eos":
    input_text += f"\n{curr_text}"
    curr_text = input("Enter the text to translate (type 'eos' to end): ")

messages = [
    {"role": "user", "content": f"Translate the following segment into {language}, without additional explanation.\n\n{input_text}"},
]

print("Tokenizing input")
tokenized_chat = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    return_tensors="pt"
)

print("Generating output")

tokenized_chat = tokenized_chat.to(device)

outputs = model.generate(tokenized_chat, max_new_tokens=2048)
output_text = tokenizer.decode(outputs[0])

print("Output:")
# output only the translated text
output_text = output_text.split("<|extra_0|>")[1].split("<|eos|>")[0]
print(output_text)