import torch

def get_attention(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    tokens = [t.replace("Ġ", " ").strip() for t in tokens]
    attentions = [layer[0].numpy() for layer in outputs.attentions]
    
    return tokens, attentions