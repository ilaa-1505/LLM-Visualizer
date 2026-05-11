# LLM Visualizer

Most people use LLMs every day but have no idea what's actually happening inside. 
This project makes the invisible visible: every token, every probability, every 
attention weight, laid out in front of you interactively.

Built with distilgpt2 running fully locally. No API calls. No black box.

---

## Demo

>![LLM Visualizer Demo](images/demo.gif)

---

## What's inside

### Embedding Space
Words are stored as vectors in a 768-dimensional space. Similar words end up close 
together, not because we told the model to, but because it learned this from 
reading billions of sentences.

This tab lets you see that. 50 words from 7 categories plotted in 3D (or 2D) space 
using PCA or UMAP dimensionality reduction. Rotate it, zoom in, see where "king" 
and "queen" land relative to each other.

> ![alt text](images/embedding.png)
> *3D embedding space showing word clusters :— places, animals, royalty and emotions 
> grouping naturally*

---

### Tokenizer Explorer
GPT-2 doesn't read words. It reads tokens. And tokenization is weirder than you think.

"Unfortunately" becomes 4 tokens. "1234" splits character by character. The same word 
gets a different token ID depending on whether there's a space before it.

Type anything here and watch it get torn apart.

> ![alt text](images/token.png)
> *"pre-trained" splitting into 3 separate tokens with their IDs*

---

### Next Token Probability
Every time GPT-2 generates a word, it's actually calculating a probability 
distribution over all 50,257 tokens in its vocabulary. You just never get to see it.

Until now. Type a prompt, move the temperature slider, watch the bars shift in 
real time. Crank temperature to 2.0 and watch the model go from focused to chaotic.

> ![alt text](images/next_token.png)
> *Top 15 next token probabilities with temperature at 0.3 vs 1.5 : notice how 
> the distribution flattens*

---

### Generation Walkthrough
This is the one that makes it click.

Watch the model build a sentence one token at a time. Each word is colored by 
confidence — green means it was sure, red means it was basically guessing. Below, 
a live decision log shows exactly what it considered and rejected at every step.

> ![alt text](images/generation-1.png)
> ![alt text](images/generation-2.png)
> *Sentence building word by word with decision log showing alternatives at each step*

---

### Attention Heatmap
The attention mechanism is what makes transformers work. Every token looks at every 
other token and decides how much to focus on it. This tab shows those decisions as 
a heatmap.

Switch between layers and heads. Layer 1 captures syntax. Layer 6 captures meaning. 
12 different heads, each specializing in something different, the model learned all 
of this on its own.

> ![alt text](images/attention-1.png)
> ![alt text](images/attention-2.png)
> *Layer 3 averaged attention on "The cat sat on the mat", notice how every token 
> attends back to "The" and "cat"*

---

## Why distilgpt2

Deliberately chose a small 82M parameter model over something larger. The goal 
here is transparency, not performance. distilgpt2 runs fully on CPU, loads in 
seconds, and exposes every internal value without any API restrictions.

A larger model would just be a bigger black box. That defeats the point.

---

## Stack

- **Model** — distilgpt2 (HuggingFace transformers, runs locally)
- **UI** — Streamlit
- **Visualizations** — Plotly
- **Dimensionality reduction** — UMAP, PCA (scikit-learn)
- **Deep learning** — PyTorch

---

## Run it yourself

```bash
git clone https://github.com/ilaa-1505/llm-visualizer
cd llm-visualizer
pip install -r requirements.txt
```

Download the model once:
```bash
python save_model.py
```

Run the app:
```bash
streamlit run app.py
```

First load takes ~10 seconds while the model loads into memory. 
Every reload after that is instant.

---

## Things I learned building this

- Token embeddings in GPT-2 are optimized for prediction, not similarity, 
  which is why clusters are messier than Word2Vec
- The upper triangle of every attention matrix is always empty.
  causal masking means the model literally cannot see future tokens
- Temperature doesn't change what the model knows, 
  it just changes how decisive it is

---

## What's next

- Add support for larger models (GPT-2 medium, GPT-2 large)
- Side by side comparison of different models on same prompt