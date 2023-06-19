import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import json 
from transformers import AutoTokenizer, AutoModelForCausalLM
from Ashaar.models import create_transformer_model, char2idx
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix
import json
import random
import argparse
import numpy as np

def run(args):
    top_accuracy = args.top_accuracy
    max_permeter = args.max_permeter
    theme_to_token = json.load(open("extra/theme_tokens.json", "r"))
    token_to_theme = {t:m for m,t in theme_to_token.items()}
    meter_to_token = json.load(open("extra/meter_tokens.json", "r"))
    token_to_meter = {t:m for m,t in meter_to_token.items()}


    gpt_tokenizer = AutoTokenizer.from_pretrained('arbml/ashaar_tokenizer')
    gpt_model = AutoModelForCausalLM.from_pretrained('checkpoint-450000')

    meter_model = create_transformer_model()
    meter_model.load_weights("deep-learning-models/meters_model/cp.ckpt")

    with open('extra/labels.txt', 'r') as f:
        label2name = f.readlines()
        label2name = [name.replace('\n', '') for name in label2name]

    with open('extra/labels_ar.txt', 'r') as f:
        label2name_ar = f.readlines()
        label2name_ar = [name.replace('\n', '') for name in label2name_ar]

    def predict_meter(bayt, meter, top_accuracy = 1):
        x = [[char2idx[char] for char in sent if char in char2idx] for sent in bayt]
        x = pad_sequences(x, padding='post', value=0, maxlen = 128)
        logits = meter_model.predict(x, verbose = 0)
        most_common = Counter([label2name_ar[logit.argmax()]for logit in logits]).most_common(5)
        out_meters = []
        for (pred_meter, rep) in most_common:
            if pred_meter in meters_to_eval:
                out_meters.append(pred_meter)
        
        if len(out_meters) == 0:
            # if all possible meters is empty, return a random pick for the meter
            return random.choice(meters_to_eval)

        if meter in out_meters[:top_accuracy]:
            return meter
        else:
            return out_meters[0]

    def generate(meter, top_p = 3):
        prompt = f"""{meter_to_token[meter]}""".strip()
        encoded_input = gpt_tokenizer(prompt, return_tensors='pt')
        output = gpt_model.generate(**encoded_input, max_length = 512, top_p = top_3, do_sample=True, pad_token_id=gpt_tokenizer.eos_token_id)

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
                if 'meter' in decoded:
                    break
                if decoded in ["<|vsep|>", "</|bsep|>"]:
                    result += "\n"
                    line_cnts+=1
                elif decoded in ['<|bsep|>', '<|psep|>', '</|psep|>']:
                    pass
                elif 'theme' in decoded:
                    pass
                else:
                    result += decoded
                prev_token = decoded
            else:
                break
        result = result[1:]
        bayts = result.split('\n')
        bayts = [verse.strip() for verse in bayts]
        bayts = [' # '.join(bayts[i:i+2]) for i in range(0, len(bayts), 2)]
        return bayts

    true_meters = []
    pred_meters = []
    meters_to_eval = ['الطويل', 'الكامل', 'البسيط', 'الخفيف', 'الوافر', 'الرجز', 'الرمل', 'المتقارب', 'السريع', 'المنسرح', 'المديد', 'الهزج', 'المتدارك', 'المقتضب', 'المضارع']
    pbar = tqdm(total = len(meters_to_eval) * max_permeter)

    max_gens = 10
    poems = []
    for meter in meters_to_eval:
        for i in range(max_permeter):
            
            generated_bayts = []
            for i in range(max_gens):
                generated_bayts = generate(meter = meter,)
                
                pred_meter =predict_meter(generated_bayts, meter, top_accuracy = top_accuracy)
                if len(generated_bayts) >= 5: #only report for poems
                    true_meters.append(meters_to_eval.index(meter))
                    pred_meters.append(meters_to_eval.index(pred_meter))
                    poems.append('\n'.join(generated_bayts))
                    break
            pbar.update(1)


    accuracy = sum(np.array(pred_meters)==np.array(true_meters))/len(true_meters)
    cf_matrix = confusion_matrix(true_meters, pred_meters)
    labels = [label2name[label2name_ar.index(meter)] for meter in meters_to_eval]

    results = {
        'accuracy':accuracy,
        'true_meters': true_meters,
        'pred_meters': pred_meters,
        'labels': labels,
        'poems' : poems
    }
    with open(f'eval_results/meter_eval_results_{top_accuracy}_{max_permeter}.json', 'w') as fout:
        json.dump(results, fout, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_accuracy', type=int, default = 1)
    parser.add_argument('--max_permeter', type=int, default = 5)
    args = parser.parse_args()

    run(args)

