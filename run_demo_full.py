import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import gradio as gr
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from Ashaar.utils import get_output_df, display_highlighted_patterns, get_highlighted_patterns_html
# gpt_tokenizer = AutoTokenizer.from_pretrained('Zaid/ashaar_tokenizerv2')
# model = AutoModelForCausalLM.from_pretrained('Zaid/Ashaar_modelv2')
import json 
theme_to_token = json.load(open("theme_tokens.json", "r"))
token_to_theme = {t:m for m,t in theme_to_token.items()}
# print(theme_to_token)
meter_to_token = json.load(open("meter_tokens.json", "r"))
token_to_meter = {t:m for m,t in meter_to_token.items()}
from Ashaar.bait_analysis import BaitAnalysis
import sys
server_path = '/home/g201080740/Arabic_Diacritization'
if server_path not in sys.path:
    sys.path.append(server_path)

analysis = BaitAnalysis()

def analyze(poem):
    shatrs = poem.split("\n")[1:-1]
    baits = [' # '.join(shatrs[2*i:2*i+2]) for i in range(len(shatrs)//2)]
    output = analysis.analyze(baits,override_tashkeel=True)
    df = get_output_df(output)
    # return display_highlighted_patterns(df)
    return get_highlighted_patterns_html(df)

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
    [
        """
القلب أعلم يا عذول بدائه
وأحق منك بجفنه وبمائه
مهلا فإن العذل من أسقامه
وترفقا فالسمع من أعضائه
        """
    ],
]

# demo = gr.Interface(
#     fn=analyze,
#     inputs=[
#             # gr.Dropdown(list(meter_to_token.keys()), label="Meter"),
#             # gr.Dropdown(list(theme_to_token.keys()), label="Theme"),
#             gr.inputs.Textbox(lines=5, label="Poem"), 
#             # gr.Slider(1, 5, value=3, step = 1, label="Probability", info="Probability for sampling next token")
#             ],
#     # outputs=gr.Dataframe(label="Results of analysis"),
#     outputs = gr.HTML(),
#     examples=examples
# )

with gr.Blocks() as demo:
    with gr.Row():
        inputs = gr.inputs.Textbox(lines=5, label="Poem")

    with gr.Row():
        analyze_btn = gr.Button("analyze")

    with gr.Row():
        outputs = gr.HTML()
    
    gr.Examples(examples, inputs)
    # inputs.render()
    analyze_btn.click(analyze, inputs=inputs, outputs=outputs)

demo.launch(server_name = "0.0.0.0", share = True, debug = True)