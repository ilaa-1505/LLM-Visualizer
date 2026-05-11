def tokenize_text(text, tokenizer):
    token_ids = tokenizer.encode(text)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    types = []
    for i, token in enumerate(tokens):
        clean = token.replace("Ġ", "")  
        
        if any(c.isdigit() for c in clean):
            types.append("number")
        elif token.startswith("Ġ") or i == 0:
            types.append("full word")
        elif any(c.isalpha() for c in clean):
            types.append("subword")
        else:
            types.append("punctuation")

    tokens = [t.replace("Ġ", " ").strip() for t in tokens]
    
    return tokens, token_ids, types