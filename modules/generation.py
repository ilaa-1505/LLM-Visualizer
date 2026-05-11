from modules.next_token import get_next_token_probs


def get_next_token(text, model, tokenizer, temperature, rep_penalty):
    tokens, probs, entropy = get_next_token_probs(
        text, model, tokenizer, temperature, top_k=4, rep_penalty=rep_penalty
    )
    high_prob = [tokens[0],probs[0]]
    alternatives = list(zip(tokens[1:], probs[1:]))
    return high_prob, alternatives, entropy
