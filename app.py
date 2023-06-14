import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import gradio as gr
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from Ashaar.utils import get_output_df, display_highlighted_patterns, get_highlighted_patterns_html
from Ashaar.bait_analysis import BaitAnalysis
import sys
import json

TITLE="""<h1 style="font-size: 30px;" align="center">Ashaar: Arabic Poetry Analysis and Generation</h1>"""
DESCRIPTION = """
<p align = 'center'>
<img src='https://raw.githubusercontent.com/ARBML/Ashaar/master/images/ashaar_icon.png' width='150px' alt='logo for Ashaar'/>
</p>
"""

DESCRIPTION +="""
The demo provides a way to generate analysis for poetry and also complete the poetry. 
The generative model is a character-based conditional GPT-2 model. The pipeline contains many models for 
classification, diacritization and conditional generation. Check our [GitHub](https://github.com/ARBML/Ashaar) for more techincal details
about this work. In the demo we have two basic pipelines. `Analyze` which predicts the meter, era, theme, diacritized text, qafiyah and, arudi style.
The other module `Generate` which takes the input text, meter, theme and qafiyah to generate the full poem. 
"""
if (SPACE_ID := os.getenv('SPACE_ID')) is not None:
    DESCRIPTION = DESCRIPTION.replace("</p>", " ")
    DESCRIPTION += f'or <a href="https://huggingface.co/spaces/{SPACE_ID}?duplicate=true"><img style="display: inline; margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate the Space"/></a> and upgrade to GPU in settings.</p>'
else:
    DESCRIPTION = DESCRIPTION.replace("either", "")


gpt_tokenizer = AutoTokenizer.from_pretrained('Zaid/ashaar_tokenizerv2')
model = AutoModelForCausalLM.from_pretrained('Zaid/Ashaar_modelv2')

theme_to_token = json.load(open("extra/theme_tokens.json", "r"))
token_to_theme = {t:m for m,t in theme_to_token.items()}
meter_to_token = json.load(open("extra/meter_tokens.json", "r"))
token_to_meter = {t:m for m,t in meter_to_token.items()}

analysis = BaitAnalysis(config_yml = "/home/g201080740/Ashaar/Ashaar/config/test.yml", 
                        pretrained_model= "/home/g201080740/Arabic_Diacritization/log_dir_ashaar")
meter, theme, qafiyah = "", "", ""

def analyze(poem):
    global meter,theme,qafiyah
    shatrs = poem.split("\n")[1:-1]
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
وأحق منك بجفنه وبمائه
مهلا فإن العذل من أسقامه
وترفقا فالسمع من أعضائه"""
    ],
]
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Row():
        with gr.Column():
            gr.HTML(TITLE)
            gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column():
            gr.Markdown("Write the input, one verse in each line")
            inputs = gr.inputs.Textbox(lines=10, label="Input Poem")
        with gr.Column():
            gr.Markdown("Output generated poem")
            outputs_2 = gr.inputs.Textbox(lines=10, label="Generated Poem")

    with gr.Row():
        with gr.Column():
            analyze_btn = gr.Button("Analyze")
        with gr.Column():
            generate_btn = gr.Button("Generate")

    with gr.Row():
        outputs_1 = gr.HTML()
    
    gr.Examples(examples, inputs)
    analyze_btn.click(analyze, inputs=inputs, outputs=outputs_1)
    generate_btn.click(generate, inputs = inputs, outputs=outputs_2)

demo.launch(server_name = "0.0.0.0", share = True, debug = True)