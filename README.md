## Ashaar

Arabic poetry analysis generation and analysis. 
<p align = 'center'>
<img src='https://raw.githubusercontent.com/ARBML/Ashaar/master/images/ashaar_icon.png' width='150px' alt='logo for Ashaar'/>
</p>

## Demo 

Start the demo using the following command

```
gradio app.py
```

## Generation 

### Training 

Training the character based gpt-2 model. Our model was trained on A100 for around 500k steps. 

```
python run_clm.py \
	    --model_type gpt2 \
	    --config_overrides="n_layer=10,vocab_size=100" \
	    --dataset_name Zaid/ashaar_datasetv2 \
	    --tokenizer_name Zaid/ashaar_tokenizerv2 \
	    --per_device_train_batch_size 16 \
	    --per_device_eval_batch_size 4 \
	    --do_train \
	    --do_eval \
	    --num_train_epochs=100 \
        --eval_steps=500 \
        --logging_steps=1 \
        --save_steps=500 \
        --logging_strategy='steps' \
        --evaluation_strategy='steps' \
        --save_strategy='steps' \
	    --output_dir <output-dir> \
	    --report_to="wandb" \
        --overwrite_output_dir \
	    --push_to_hub \
	    --hub_model_id=<model-hub-id>
```

### Inference 

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

prompt = "enter your prompt here"
gpt_tokenizer = AutoTokenizer.from_pretrained('Zaid/ashaar_tokenizerv2')
model = AutoModelForCausalLM.from_pretrained('Zaid/Ashaar_modelv2')

encoded_input = gpt_tokenizer(prompt, return_tensors='pt')
output = model.generate(**encoded_input, max_length = 512, top_p = 3, do_sample=True)
```

## Analysis

```python
from Ashaar.bait_analysis import BaitAnalysis

prompt ="ألا ليت شعري هل أبيتن ليلة # بجنب الغضى أزجي القلاص النواجيا"
analysis = BaitAnalysis()
output = analysis.analyze(prompt, override_tashkeel=True)
```

Sample output 

```
{'arudi_style'     : [['ألاليت شعري هل أبيتنن ليلتن بجنب لغضى أزجلقلاص ننواجيا',
                     '1101011010101101011011011010110101011010110110']],
 'closest_baits'   : [[('ألاليت شعري هل أبيتن ليلة # بجنب الغضى أزجي القِلاص '
                     'النواجيا',
                     [0.38896721601486206])]],
 'closest_patterns': [('1101011010101101011011011010110101011010110110',
                       1.0,
                       'فعولنْ مفاعيلنْ فعولنْ مفاعلنْ # فعولنْ مفاعيلنْ '
                       'فعولنْ مفاعلنْ')],
 'diacritized'     : ['أَلَالَيْتُ شِعْرِي هَلْ أَبِيتَنَّ لَيْلَةً # بِجَنْبِ '
                     'الْغَضَى أَزْجِي الْقِلَاصَ النَّوَاجِيَا'],
 'era'             : ['العصر الحديث', 'العصر العثماني'],
 'meter'           : 'الطويل',
 'qafiyah'         : ('ي',
                     'قافية بحرف الروي: ي ،  زاد لها الوصل بإشباع رويها زاد لها '
                     'التأسيس'),
 'theme'           : ['قصيدة رومنسيه', 'قصيدة شوق', 'قصيدة غزل']}
```

