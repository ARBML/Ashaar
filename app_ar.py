import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import gradio as gr
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from Ashaar.utils import get_output_df, display_highlighted_patterns, get_highlighted_patterns_html
from Ashaar.bait_analysis import BaitAnalysis
import sys
import json

TITLE="""<h1 style="font-size: 30px;" align="center">أَشْعــَـار: تحليل وإنشاء الشعر العربي</h1>"""
DESCRIPTION = """
<p align = 'center'>
<img src='https://raw.githubusercontent.com/ARBML/Ashaar/master/images/ashaar_icon.png' width='150px' alt='logo for Ashaar'/>
</p>
"""

DESCRIPTION +=""" <p dir='rtl'>
هذا البرنامج يتيح للمستخدم تحليل وإنشاء الشعر العربي.
يتم إنشاء الشعر بواسطة نموذج GPT تم تدريبه على الحروف العربية. 
يحتوي البرنامج على نماذج لتصنيف الحقبة الزمنية والعاطفة والبحر و كذلك تشكيل الشعر العربي بالإضافة إلى إكمال الشعر.
قمنا بتوفير الشفرة البرمجية كلها على 
<a href ='https://github.com/ARBML/Ashaar'> GitHub</a>.

</p>
"""

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
وأحق منك بجفنه وبمائه
مهلا فإن العذل من أسقامه
وترفقا فالسمع من أعضائه"""
    ],
    [
"""ألا ليت شعري هل أبيتن ليلة
بجنب الغضى أزجي الغلاص النواجيا"""
    ],
]
css = """
#input-textbox, #output-textbox{
  direction: RTL;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    with gr.Row():
        with gr.Column():
            gr.HTML(TITLE)
            gr.HTML(DESCRIPTION)

    with gr.Row():
        with gr.Column():
            outputs_2 = gr.Textbox(lines=10, label="القصيدة المنشئة", elem_id="output-textbox")

        with gr.Column():
            inputs = gr.Textbox(lines=10, label="القصيدة المدخلة", elem_id="input-textbox")


    with gr.Row():
        with gr.Column():
            generate_btn = gr.Button("إنشاء")
        with gr.Column():
            analyze_btn = gr.Button("تحليل")

    with gr.Row():
        outputs_1 = gr.HTML()
    
    gr.Examples(examples, inputs)
    generate_btn.click(generate, inputs = inputs, outputs=outputs_2)
    analyze_btn.click(analyze, inputs=inputs, outputs=outputs_1)

demo.launch(server_name = "0.0.0.0", share = True, debug = True)