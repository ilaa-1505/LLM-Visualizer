import torch
import torch.nn.functional as F

def get_next_token_probs(text, model, tokenizer, temperature=1.0, top_k=15, rep_penalty=1.0, output_attention=False):
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"][0]
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=output_attention)
    
    logits = outputs.logits[0, -1, :]
    
    for token_id in set(input_ids.tolist()):
        if logits[token_id] > 0:
            logits[token_id] /= rep_penalty
        else:
            logits[token_id] *= rep_penalty
    
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    
    k = int(min(top_k, probs.shape[-1]))
    top_probs, top_ids = torch.topk(probs, k)
    
    top_tokens = [tokenizer.decode(id) for id in top_ids]
    entropy = -(probs * torch.log(probs + 1e-9)).sum().item()
    
    return top_tokens, top_probs.numpy(), entropy