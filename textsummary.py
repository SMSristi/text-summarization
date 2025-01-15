import torch
import gradio as gr

# Use a pipeline as a high-level helper
model_path = ("../Models/models--sshleifer--distilbart-cnn-12-6/snapshots/a4f8f3ea906ed274767e9906dbaede7531d660ff")
from transformers import pipeline

text_summary = pipeline("summarization", model=model_path,torch_dtype=torch.bfloat16)

# text= "A member of the wealthy South African Musk family, Musk was born in Pretoria and briefly attended the University of Pretoria. At the age of 18 he immigrated to Canada, acquiring its citizenship through his Canadian-born mother, Maye. Two years later, he matriculated at Queen's University in Canada. Musk later transferred to the University of Pennsylvania and received bachelor's degrees in economics and physics. He moved to California in 1995 to attend Stanford University but never enrolled in classes, and with his brother Kimbal co-founded the online city guide software company Zip2. The startup was acquired by Compaq for $307 million in 1999. That same year, Musk co-founded X.com, a direct bank. X.com merged with Confinity in 2000 to form PayPal. In 2002, Musk acquired United States citizenship, and that October eBay acquired PayPal for $1.5 billion. Using $100 million of the money he made from the sale of PayPal, Musk founded SpaceX, a spaceflight services company, in 2002."
# print(text_summary(text));

def summary (input):
    output = text_summary(input)
    return output[0]['summary_text']

gr.close_all()

# Define custom CSS for styling
custom_css = """
    .gradio-container {
        font-family: 'Roboto', sans-serif;
        background: linear-gradient(to right, #ece9e6, #ffffff);
        color: #000000 !important;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 10px 20px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #34495e;
        text-align: center;
        font-weight: bold;
    }
    .input-textbox textarea {
        border: 2px solid #1abc9c !important;
        border-radius: 10px !important;
        font-size: 16px;
        padding: 10px;
        color: #2c3e50;
        background: #ecf9f7 !important;
    }
    .output-textbox textarea {
        border: 2px solid #3498db !important;
        border-radius: 10px !important;
        font-size: 16px;
        padding: 10px;
        color: #2c3e50;
        background: #ebf5ff !important;
    }
    .btn-primary {
        background-color: #1abc9c !important;
        border-color: #1abc9c !important;
        color: black;
        font-size: 16px;
        font-weight: bold;
        padding: 12px 20px !important;
        border-radius: 8px !important;
    }
    .btn-primary:hover {
        background-color: #16a085 !important;
    }
    .gradio-container .description {
        color: #000000 !important; 
    }
"""

# Create the Gradio Interface
demo = gr.Interface(
    fn=summary,
    inputs=[
        gr.Textbox(
            label="Input Text",
            placeholder="Paste your text here to summarize...",
            lines=10,
            elem_classes=["input-textbox"]
        )
    ],
    outputs=[
        gr.Textbox(
            label="Summarized Text",
            placeholder="Your summary will appear here...",
            lines=5,
            elem_classes=["output-textbox"]
        )
    ],
    title="🌟 Modern AI Text Summarizer 🌟",
    description=(
        "Generate concise summaries of your long texts with this powerful AI tool. "
        "Simply enter your text in the input box and press 'Submit'!"
    ),
    theme="compact",
    css=custom_css  # Apply the custom CSS
)

demo.launch()