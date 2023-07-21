import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DistilBertForTokenClassification

st.title("NLI Graph Spacy")
st.markdown('NLIGRAPHSPACY - A simple tool to create knowledge graphs in **NLP** using a pre-trained model modelled on custom dataset created using spaCy library')

form = st.form("nligraphspacy - form")
text = form.text_input(label='Enter your own sentence')
submit = form.form_submit_button("Submit")

with st.spinner('Wait for the model to load, please click Submit once you see Done status'):
    m_name = "vishnun/kg_model"
    gtokenizer = AutoTokenizer.from_pretrained(m_name)
    gmodel = DistilBertForTokenClassification.from_pretrained(m_name)
st.success('Done!', icon="✅")

if submit:
    inputs = gtokenizer(text, return_tensors="pt")
    tokens = gtokenizer.tokenize(text)

    with torch.no_grad():
        logits = gmodel(**inputs).logits

    predictions = torch.argmax(logits, dim=2)
    predicted_token_class = [gmodel.config.id2label[t.item()] for t in predictions[0][1:-1]]

    entities = []
    for label, text in zip(predicted_token_class, tokens):
        js_dict = {}
        js_dict[text] = label
        entities.append(js_dict)

    st.json(entities)
