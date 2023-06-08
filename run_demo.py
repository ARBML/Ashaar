import gradio as gr
from transformers import pipeline

from transformers import AutoTokenizer, AutoModelForCausalLM

gpt_tokenizer = AutoTokenizer.from_pretrained('Zaid/ashaar_tokenizerv2')
model = AutoModelForCausalLM.from_pretrained('Zaid/Ashaar_modelv2')

import json 
theme_to_token = json.load(open("theme_tokens.json", "r"))
token_to_theme = {t:m for m,t in theme_to_token.items()}
print(theme_to_token)
meter_to_token = json.load(open("meter_tokens.json", "r"))
token_to_meter = {t:m for m,t in meter_to_token.items()}

def generate(meter="", theme= "", qafiyah = "", top_p = 3):

    prompt = f"{meter_to_token[meter]} {qafiyah} {theme_to_token[theme]}"
    encoded_input = gpt_tokenizer(prompt, return_tensors='pt')
    output = model.generate(**encoded_input, max_length = 512, top_p = 3, do_sample=True)

    result = ""
    prev_token = ""

    for i, beam in enumerate(output[:, len(encoded_input.input_ids[0]):]):
        for token in beam:
            decoded = gpt_tokenizer.decode(token)
            if 'meter' in decoded or 'theme' in decoded:
                break
            if decoded == "<|vsep|>":
                result += "*** "
            elif decoded in ["<|bsep|>", "</|bsep|>"]:
                result += "\n"
            elif decoded in ['<|psep|>', '</|psep|>']:
                pass
            else:
                result += decoded
            prev_token = decoded
        else:
            break
    return theme+" "+ f"من بحر {meter} مع قافية بحر ({qafiyah})" + "\n" +result

examples = [
    ["الكامل", "قصيدة حزينه", "ك"],
]

demo = gr.Interface(
    fn=generate,
    inputs=[
            gr.Dropdown(list(meter_to_token.keys()), label="Meter"),
            gr.Dropdown(list(theme_to_token.keys()), label="Theme"),
            gr.inputs.Textbox(lines=1, label="Qafiyah"), 
            gr.Slider(1, 5, value=3, step = 1, label="Probability", info="Probability for sampling next token")],
    outputs=gr.outputs.Textbox(label="Generated Poem", dir= 'rtl'),
    examples=examples
)

demo.launch(server_name = "0.0.0.0", share = True, debug = True)