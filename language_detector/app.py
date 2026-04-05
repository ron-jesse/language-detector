import gradio as gr
import pickle

# Load saved files
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def predict_language(text):
    text = text.lower()
    vector = vectorizer.transform([text])

    probs = model.predict_proba(vector)[0]
    classes = model.classes_

    top2 = probs.argsort()[-2:][::-1]

    if probs[top2[1]] > 0.30:
        return f"Mixed: {classes[top2[0]]} + {classes[top2[1]]}"

    return classes[top2[0]]

# Gradio UI
app = gr.Interface(
    fn=predict_language,
    inputs=gr.Textbox(lines=2, placeholder="Enter text here..."),
    outputs="text",
    title="Language Identification System",
    description="Detect Swahili, English, Sheng, or Luo"
)

app.launch()