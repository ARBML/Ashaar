import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import gradio as gr
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from Ashaar.utils import get_output_df, display_highlighted_patterns, get_highlighted_patterns_html
from Ashaar.bait_analysis import BaitAnalysis
from langs import *
import sys
import json
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--lang', type = str, default = 'ar', required=True)
args = arg_parser.parse_args()
lang = args.lang

if lang == 'ar':
    TITLE = TITLE_ar
    DESCRIPTION = DESCRIPTION_ar
    textbox_trg_text = textbox_trg_text_ar
    textbox_inp_text = textbox_inp_text_ar
    btn_trg_text = btn_trg_text_ar
    btn_inp_text = btn_inp_text_ar
    css = """ #textbox{ direction: RTL;}"""

else:
    TITLE = TITLE_en
    DESCRIPTION = DESCRIPTION_en
    textbox_trg_text = textbox_trg_text_en
    textbox_inp_text = textbox_inp_text_en
    btn_trg_text = btn_trg_text_en
    btn_inp_text = btn_inp_text_en
    css = ""

gpt_tokenizer = AutoTokenizer.from_pretrained('arbml/ashaar_tokenizer')
model = AutoModelForCausalLM.from_pretrained('arbml/Ashaar_model')

theme_to_token = json.load(open("extra/theme_tokens.json", "r"))
token_to_theme = {t:m for m,t in theme_to_token.items()}
meter_to_token = json.load(open("extra/meter_tokens.json", "r"))
token_to_meter = {t:m for m,t in meter_to_token.items()}

analysis = BaitAnalysis()
meter, theme, qafiyah = "", "", ""

def analyze(poem):
    global meter,theme,qafiyah
    shatrs = poem.split("\n")
    baits = [' # '.join(shatrs[2*i:2*i+2]) for i in range(len(shatrs)//2)]
    output = analysis.analyze(baits,override_tashkeel=True)
    meter = output['meter']
    qafiyah = output['qafiyah'][0]
    theme = output['theme'][-1]
    
    df = get_output_df(output)
    return get_highlighted_patterns_html(df)

def generate(inputs, top_p = 3):
    baits = inputs.split('\n')
    print(baits)
    poem = ' '.join(['<|bsep|> '+baits[i]+' <|vsep|> '+baits[i+1]+' </|bsep|>' for i in range(0, len(baits), 2)])
    print(poem)
    prompt = f"""
    {meter_to_token[meter]} {qafiyah} {theme_to_token[theme]}
    <|psep|>
    {poem}
    """.strip()
    print(prompt)
    encoded_input = gpt_tokenizer(prompt, return_tensors='pt')
    output = model.generate(**encoded_input, max_length = 512, top_p = 3, do_sample=True)

    result = ""
    prev_token = ""
    line_cnts = 0
    for i, beam in enumerate(output[:, len(encoded_input.input_ids[0]):]):
        if line_cnts >= 10:
            break
        for token in beam:
            if line_cnts >= 10:
                break
            decoded = gpt_tokenizer.decode(token)
            if 'meter' in decoded or 'theme' in decoded:
                break
            if decoded in ["<|vsep|>", "</|bsep|>"]:
                result += "\n"
                line_cnts+=1
            elif decoded in ['<|bsep|>', '<|psep|>', '</|psep|>']:
                pass
            else:
                result += decoded
            prev_token = decoded
        else:
            break
    # return theme+" "+ f"من بحر {meter} مع قافية بحر ({qafiyah})" + "\n" +result
    return result
examples = [
    [
"""القلب أعلم يا عذول بدائه
وأحق منك بجفنه وبمائه"""
    ],
    [
"""ألا ليت شعري هل أبيتن ليلة
بجنب الغضى أزجي الغلاص النواجيا"""
    ],
]

with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    with gr.Row():
        with gr.Column():
            gr.HTML(TITLE)
            gr.HTML(DESCRIPTION)

    with gr.Row():
        with gr.Column():
            textbox_output = gr.Textbox(lines=10, label=textbox_trg_text, elem_id="textbox")
        with gr.Column():
            inputs = gr.Textbox(lines=10, label=textbox_inp_text, elem_id="textbox")


    with gr.Row():
        with gr.Column():
            trg_btn = gr.Button(btn_trg_text)
        with gr.Column():
            inp_btn = gr.Button(btn_inp_text)

    with gr.Row():
        html_output = gr.HTML()
        
    if lang == 'en':
        gr.Examples(examples, textbox_output)
        inp_btn.click(generate, inputs = textbox_output, outputs=inputs)
        trg_btn.click(analyze, inputs = textbox_output, outputs=html_output)
    else:
        gr.Examples(examples, inputs)
        trg_btn.click(generate, inputs = inputs, outputs=textbox_output)
        inp_btn.click(analyze, inputs = inputs, outputs=html_output)

demo.launch(server_name = "0.0.0.0", share = True, debug = True)