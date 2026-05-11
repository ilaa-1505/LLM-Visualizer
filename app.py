import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import os
import time


st.set_page_config(
    page_title="LLM Visualizer",
    page_icon="",
    layout="wide"
)

st.title(" LLM Visualizer")
st.caption("See exactly what happens inside GPT-2")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Embedding Space",
    "Tokenizer Explorer",
    "Next Token Probability",
    "Generation Walkthrough",
    "Attention Heatmap"
])

@st.cache_resource
def load_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if not os.path.exists(".model/"):
        AutoTokenizer.from_pretrained("distilgpt2").save_pretrained(".model/")
        AutoModelForCausalLM.from_pretrained("distilgpt2").save_pretrained(".model/")

    tokenizer = GPT2Tokenizer.from_pretrained(".model/")
    model = GPT2LMHeadModel.from_pretrained(".model/", output_attentions=True)
    model.eval()
    return tokenizer, model

if "model_loaded" not in st.session_state:
    st.markdown("""
        <div style="display:flex; justify-content:center; 
        align-items:center; height:80vh; flex-direction:column; gap:10px">
            <h1>LLM Visualizer</h1>
            <p style="font-size:18px">Loading GPT-2 into memory...</p>
            <p style="color:gray">This only happens once per session</p>
        </div>
    """, unsafe_allow_html=True)
    tokenizer, model = load_model()
    st.session_state.model_loaded = True
    st.session_state.show_success = True
    st.rerun()
else:
    tokenizer, model = load_model()

if st.session_state.get("show_success"):
    success = st.success("Model ready!")
    time.sleep(8)
    success.empty()
    st.session_state.show_success = False
    

with tab1:
    st.header("Embedding Space Visualizer")
    
    from modules.embedding_viz import get_word_embeddings, reduce_dimensions
    import plotly.express as px
    import pandas as pd

    method = st.radio("Reduction method", ["UMAP", "PCA"], horizontal=True)
    dimensions = st.radio("View", ["3D", "2D"], horizontal=True)

    if st.button("Generate", key="embed_generate"):
        with st.spinner("Computing embeddings..."):
            words, categories, vectors = get_word_embeddings(model, tokenizer)
            n = 3 if dimensions == "3D" else 2
            coords = reduce_dimensions(vectors, method, n_components=n)

        if dimensions == "3D":
            df = pd.DataFrame({
                "word": words, "category": categories,
                "x": coords[:, 0], "y": coords[:, 1], "z": coords[:, 2]
            })
            fig = px.scatter_3d(df, x="x", y="y", z="z",
                color="category", text="word", title="Word Embeddings in 3D Space")
        else:
            df = pd.DataFrame({
                "word": words, "category": categories,
                "x": coords[:, 0], "y": coords[:, 1]
            })
            fig = px.scatter(df, x="x", y="y",
                color="category", text="word", title="Word Embeddings in 2D Space")

        fig.update_traces(textposition="top center", marker=dict(size=6))
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    from modules.tokenizer_explorer import tokenize_text

    text = st.text_input("Type anything", "What are pre-trained models with respect to Transformers?",key="tok_input")

    if text:
        tokens, ids, types = tokenize_text(text, tokenizer)
        color_map = {
    "full word": "🟦",
    "subword": "🟨", 
    "punctuation": "🟥",
    "number": "🟩"
}
        cols = st.columns(len(tokens))
        for i, col in enumerate(cols):
            col.markdown(f"{color_map[types[i]]} **{tokens[i]}**")
            col.caption(str(ids[i]))
        st.markdown("🟦 Full word &nbsp; 🟨 Subword &nbsp; 🟥 Punctuation &nbsp; 🟩 Number")
        st.write(f"Total tokens: {len(tokens)} | Characters: {len(text)} | Ratio: {len(text)/len(tokens):.1f} chars/token")

with tab3:
    st.header("Next Token Probability")
    
    from modules.next_token import get_next_token_probs
    import plotly.express as px

    text = st.text_input("Enter prompt", "Once upon a time",key="prob_input")
    temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1, key="prob_temperature")
    rep_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.0, 0.1, key="prob_rep_penalty")
    st.caption("1.0 = no penalty, 2.0 = strongly avoid repeating tokens")

    if text:
        with st.spinner("Running forward pass..."):
            tokens, probs, entropy = get_next_token_probs(
                text, model, tokenizer, temperature, top_k=15, rep_penalty=rep_penalty
            )

        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig = px.bar(
                x=probs, y=tokens,
                orientation="h",
                title="Top 15 Next Token Probabilities",
                labels={"x": "Probability", "y": "Token"}
            )
            fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric("Entropy", f"{entropy:.2f}")
            st.caption("Low = confident\nHigh = uncertain")
            st.metric("Top token", tokens[0])
            st.metric("Top prob", f"{probs[0]:.2%}")

with tab4:
    st.header("Generation Walkthrough")
    
    from modules.generation import get_next_token
    import time
    import pandas as pd

    def token_color(prob):
        if prob > 0.3:
            return "green"
        elif prob > 0.1:
            return "orange"
        else:
            return "red"

    prompt = st.text_input("Enter prompt", "Once upon a time", key="gen_prompt")
    st.markdown("""
    <div style="background:#1a1a2e; border-radius:10px; padding:25px; margin-top:20px">
        <h4> How Generation Works</h4>
        <p style="color:#aaa">At each step the model:</p>
        <ol style="color:#aaa; line-height:2">
            <li>Reads your entire prompt so far</li>
            <li>Calculates probability for all 50,257 tokens</li>
            <li>Picks the next token based on temperature</li>
            <li>Adds it to the prompt and repeats</li>
        </ol>
        <br>
        <p>🟢 <b style="color:green">Green</b> = model was confident (p > 0.3)</p>
        <p>🟠 <b style="color:orange">Orange</b> = model was uncertain (p 0.1-0.3)</p>
        <p>🔴 <b style="color:red">Red</b> = model was guessing (p < 0.1)</p>
        <br>
        <p style="color:#aaa; font-size:13px">Try low temperature (0.1) for focused output vs high temperature (1.5) for creative/chaotic output</p>
                
    </div>
                <br><br>
""", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        max_tokens = st.slider("Max tokens", 5,100, 20, 1, key="gen_max_tokens")
    with col2:
        temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1, key="gen_temperature")
    with col3:
        rep_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.3, 0.1, key="gen_rep_penalty")

    if st.button("Generate", key="gen_generate"):
        sentence = prompt
        sentence_tokens = [prompt]
        sentence_probs = [1.0]
        history_data = []

        sentence_placeholder = st.empty()
        table_placeholder = st.empty()

        for i in range(int(max_tokens)):
            token, alternatives, entropy = get_next_token(
                sentence, model, tokenizer, temperature, rep_penalty
            )

            sentence += token[0]
            sentence_tokens.append(token[0])
            sentence_probs.append(float(token[1]))

            history_data.append({
                "Step": i + 1,
                "Chosen Token": token[0].strip(),
                "Probability": f"{token[1]:.2f}",
                "Alternatives": " | ".join([f"{t.strip()}({p:.2f})" for t, p in alternatives])
            })

            colored_tokens = []
            for j, (tok, prob) in enumerate(zip(sentence_tokens, sentence_probs)):
                if j == len(sentence_tokens) - 1:
                    color = token_color(prob)
                    colored_tokens.append(f'<span style="color:{color}"><b>{tok}</b></span>')
                else:
                    colored_tokens.append(tok)

            sentence_placeholder.markdown(
    f'<div style="font-size:28px; font-weight:bold; line-height:1.6">{" ".join(colored_tokens)}</div>',
    unsafe_allow_html=True
)

            df = pd.DataFrame(history_data)
            table_placeholder.dataframe(
    df, 
    use_container_width=True,
    height=(len(history_data) + 1) * 35 + 3  
)

            time.sleep(1)
with tab5:
    st.header("Attention Heatmap")
    
    from modules.attention import get_attention
    import plotly.express as px
    import numpy as np

    text = st.text_input("Enter sentence (max 20 words)", "The cat sat on the mat", key="att_input")

    st.markdown("""
    <div style="background:#1a1a2e; border-radius:10px; padding:25px; margin-top:20px">
        <h4>How Attention Works</h4>
        <p style="color:#aaa">When GPT-2 processes a sentence, every token "looks at" every other token with different levels of focus. This is called <b>self-attention</b>.</p>
        <br>
        <p style="color:#aaa; line-height:2">
            <b>Darker red</b> = stronger attention between two tokens<br>
            <b>White</b> = token is ignoring that position<br>
            <b>Upper triangle is always empty</b> — GPT-2 can only look backwards, never at future tokens<br>
            <b>First column dark</b> — most tokens attend strongly to the first word
        </p>
        <br>
        <h5>What to try:</h5>
        <p style="color:#aaa; line-height:2">
            🔹 <b>Layer 1-2</b> — basic syntax, nearby word relationships<br>
            🔹 <b>Layer 3-4</b> — grammar, subject-verb patterns<br>
            🔹 <b>Layer 5-6</b> — semantics, meaning level patterns<br>
            🔹 <b>Average all heads</b> — see the overall picture across all 12 perspectives
        </p>
        <br>
        <p style="color:#aaa; font-size:13px"> Try "The cat sat on the mat" and see how "cat" gets attended to by everything after it</p>
    </div>
    <br><br>
""", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        layer = st.slider("Layer", 1, 6, 1, 1, key="att_layer")
    with col2:
        head = st.slider("Head", 1, 12, 1, 1, key="att_head")
    
    avg_heads = st.checkbox("Average all heads")

    if st.button("Show Attention", key="att_button"):
        with st.spinner("Computing attention..."):
            tokens, attentions = get_attention(text, model, tokenizer)
        
        # attentions[layer] shape = (12, seq_len, seq_len)
        layer_attention = attentions[layer - 1]  # select layer
        
        if avg_heads:
            matrix = layer_attention.mean(axis=0)  # average across heads
        else:
            matrix = layer_attention[head - 1]  # select specific head

        fig = px.imshow(
            matrix,
            x=tokens,
            y=tokens,
            color_continuous_scale="Reds",
            title=f"Attention - Layer {layer} {'(avg heads)' if avg_heads else f'Head {head}'}",
            labels={"x": "Attends To", "y": "Token"}
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # interesting pattern callouts
        st.subheader("What to look for")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(" Diagonal = each token attending to itself")
        with col2:
            st.info("Last row = final token attends to everything")
        with col3:
            st.info(" Try different layers — early=syntax, late=semantics")

    