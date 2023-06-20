IMG = """<p align = 'center'>
<img src='https://raw.githubusercontent.com/ARBML/Ashaar/master/images/ashaar_icon.png' width='150px' alt='logo for Ashaar'/>
</p>

"""
TITLE_ar="""<h1 style="font-size: 30px;" align="center">أَشْعــَـار: تحليل وإنشاء الشعر العربي</h1>"""
DESCRIPTION_ar = IMG

DESCRIPTION_ar +=""" <p dir='rtl'>
هذا البرنامج يتيح للمستخدم تحليل وإنشاء الشعر العربي.
لإنشاء الشعر العربي تم تدريب نموج يقوم بإستخدام البحر والقافية والعاطفة لإنشاء أكمال للقصيدة بناء على هذه الشورط.
بالإضافة إلى نموذج إنشاء الشعر يحتوي البرنامج على نماذج لتصنيف الحقبة الزمنية والعاطفة والبحر و كذلك تشكيل الشعر العربي بالإضافة إلى إكمال الشعر.
قمنا بتوفير الشفرة البرمجية كلها على 
<a href ='https://github.com/ARBML/Ashaar'> GitHub</a>.

</p>
"""

TITLE_en="""<h1 style="font-size: 30px;" align="center">Ashaar: Arabic Poetry Analysis and Generation</h1>"""
DESCRIPTION_en = IMG

DESCRIPTION_en +="""
The demo provides a way to generate analysis for poetry and also complete the poetry. 
The generative model is a character-based conditional GPT-2 model. The pipeline contains many models for 
classification, diacritization and conditional generation. Check our <a src='https://github.com/ARBML/Ashaar'>GitHub</a> for more techincal details
about this work. In the demo we have two basic pipelines. Analyze which predicts the meter, era, theme, diacritized text, qafiyah and, arudi style.
The other module, Generate which takes the input text, meter, theme and qafiyah to generate the full poem. 
"""

btn_trg_text_ar = "إنشاء"
btn_inp_text_ar = "تحليل"

btn_inp_text_en = "Generate"
btn_trg_text_en = "Analyze"

textbox_inp_text_ar = "القصيدة المدخلة"
textbox_trg_text_ar = "القصيدة المنشئة"

textbox_trg_text_en = "Input Poem"
textbox_inp_text_en = "Generated Poem"



