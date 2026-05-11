from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("distilgpt2")
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")

model.save_pretrained("./model/distilgpt2")
tokenizer.save_pretrained("./model/distilgpt2")

print("saved!")
