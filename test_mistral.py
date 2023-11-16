from transformers import AutoModelForCausalLM, AutoTokenizer
from valley.conversation import conv_templates
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("/mnt/bn/luoruipu-disk/weight_pretrained/Mistral-7B-Instruct-v0.1/")
tokenizer = AutoTokenizer.from_pretrained("/mnt/bn/luoruipu-disk/weight_pretrained/Mistral-7B-Instruct-v0.1/")

messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

conv = conv_templates['mistral']
conv.messages = []
for i,msg in enumerate(messages):
    conv.append_message(conv.roles[0] if i%2 == 0 else conv.roles[1], msg['content'])

prompt = conv.get_prompt()
encodeds = tokenizer.encode(prompt,return_tensors='pt')

model_inputs = encodeds.to(device)
model.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])