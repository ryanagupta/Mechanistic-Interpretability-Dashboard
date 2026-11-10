"""A local Streamlit explorer for a Pythia forward pass."""

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "EleutherAI/pythia-70m"
MAX_TOKENS = 96

st.set_page_config(page_title="Pythia Explorer", layout="wide")


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, use_safetensors=True)
    return tokenizer, model.eval()


st.title("Pythia Explorer")
st.caption("Observe a single forward pass through EleutherAI’s Pythia-70M.")
prompt = st.text_area("Prompt", "The capital of France is", max_chars=1000)
if not prompt.strip():
    st.info("Enter a prompt to inspect.")
    st.stop()

try:
    with st.spinner("Loading Pythia-70M"):
        tokenizer, model = load_model()
except OSError:
    st.error("Could not load the model. Check your connection and Hugging Face cache.")
    st.stop()

inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
with torch.inference_mode():
    outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    probabilities = outputs.logits[0, -1].float().softmax(dim=-1)
    scores, ids = probabilities.topk(10)

st.subheader("Next-token probabilities")
st.dataframe(
    [
        {"Token": repr(tokenizer.decode([token_id])), "ID": token_id, "Probability": score}
        for token_id, score in zip(ids.tolist(), scores.tolist())
    ],
    hide_index=True,
    column_config={"Probability": st.column_config.NumberColumn(format="percent")},
)

st.subheader("Hidden-state channels")
token_ids = inputs.input_ids[0].tolist()
st.caption(f"Inspecting {len(token_ids)} token(s). Input is capped at {MAX_TOKENS} tokens.")
left, right = st.columns(2)
with left:
    layer = st.selectbox("Hidden-state index", range(len(outputs.hidden_states)))
with right:
    position = st.selectbox(
        "Token position",
        range(len(token_ids)),
        index=len(token_ids) - 1,
        format_func=lambda pos: f"{pos} · {tokenizer.decode([token_ids[pos]])!r}",
    )
hidden = outputs.hidden_states[layer][0, position].float()
channels = hidden.abs().topk(16).indices.tolist()
st.bar_chart(
    [{"Channel": str(channel), "Activation": hidden[channel].item()} for channel in channels],
    x="Channel",
    y="Activation",
)
st.caption("Shows the 16 largest channels by absolute value. Index 0 is the input embedding state.")
