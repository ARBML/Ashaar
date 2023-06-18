## Ashaar

Arabic poetry analysis and generation. 

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
        --dataset_name arbml/Ashaar_dataset \
        --tokenizer_name arbml/ashaar_tokenizer \
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
gpt_tokenizer = AutoTokenizer.from_pretrained('arbml/Ashaar_tokenizer')
model = AutoModelForCausalLM.from_pretrained('arbml/Ashaar_model')

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
{'diacritized': ['أَلا لَيْتَ شِعْرِي هَلْ أَبَيْتَنَّ لَيْلَةً # بِجَنْبِ الْغَضَى أَزُجَيَّ الْقَلَاصِ النَّوَاجِيَا'],
 'arudi_style': [('ألا ليت شعري هل أبيتنن ليلتن', '10110110101011010110110'),
  ('بجنب لغضى أزجيي لقلاص ننواجيا', '1101011011101011010110110')],
 'patterns_mismatches': ['G1G0R1G1G0G1G1G0G1G0G1G0G1G1G0G1G0G1G1G0G1G1G0',
  'G1G1G0G1G0G1G1G0R1R1G1G0G1G0G1G1G0G1G0G1G1G0G1G1G0'],
 'qafiyah': ('ي',
  'قافية بحرف الروي: ي ،  زاد لها الوصل بإشباع رويها زاد لها التأسيس'),
 'meter': 'الطويل',
 'closest_baits': [[('فَيَا لَيْتَنِي لَمْ أَعْنِ بِالْمُلْكِ سَاعَةً # وَلَمْ أَلْهُ فِي لَذَّاتِ عَيْشٍ نَوَاضِرِ',
    [0.8884615898132324])]],
 'era': ['العصر الجاهلي', 'العصر الإسلامي', 'العصر الأموي', 'قبل الإسلام'],
 'closest_patterns': [('1010110101011010110110',
   0.9777777777777777,
   'عولنْ مفاعيلنْ فعولنْ مفاعلنْ'),
  ('11010110101011010110110',
   0.9583333333333334,
   'فعولنْ مفاعيلنْ فعولنْ مفاعلنْ')],
 'theme': ['قصيدة رومنسيه', 'قصيدة شوق', 'قصيدة غزل']}
```

