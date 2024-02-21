# -*- coding: utf-8 -*- 
import re
import torch
import time
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from datasets import load_dataset

def load(model_name, peft_model_name = None):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",
                                                 quantization_config=quantization_config,
                                                 trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)

    if peft_model_name != None:
        config = PeftConfig.from_pretrained(peft_model_name)
        model = PeftModel.from_pretrained(model, peft_model_name)
        
    generation_config = GenerationConfig(
        max_new_tokens=32,
        do_sample=True,
        temperature=0.3,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    return model, generation_config, tokenizer

def run(model, config, tokenizer, ptext):
    inp = tokenizer(ptext, return_tensors="pt")
    inputs = inp.to(model.device)
    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            generation_config = config, 
        )
        output = tokenizer.decode(tokens[0], skip_special_tokens=True)
    return output

quiz_file = "quiz.json"
data = load_dataset("json", data_files=quiz_file)

model_id = 'Orion-14B-qlora-out'
model, cfg, tokenizer = load(model_id)

quiz_base = quiz_file.split('.')[0]
output = "result2"
filename = f"{output}.log"
print(filename)

st = time.time()
js = []
with open(filename, 'w', encoding="utf-8") as f:
    f.write(f"BASE model : {model_id}\n")
    f.write(f"QUIZ file  : {quiz_file}\n")
    f.write(f"-------------------------------------------------\n")
    count_all = 0
    count_ans = 0
    for data_point in data['train']:
        q = data_point['instruction']
        o = data_point['output']
        
        ptext = q
        res = run(model, cfg, tokenizer, ptext)
        count_all = count_all + 1

        # Extract everything after "回答："
        answer = re.search(r'### 回答：\n(.+)', res)

        hit = False
        if answer:
            out_str = answer.group(1)
            match = re.search(r"(\d+)：", out_str)
            if match:
                ans = match.group(1)
                x = (o[0] == ans)
                if x:
                    count_ans = count_ans + 1
                    hit = True
                
                l = f"{res} / {x} : {float(count_ans)/float(count_all)}"
            else:
                l = f"{res} / strange response!!"            
        else:
            l = ""
                
        f.write(l + "\n")
        f.write("\n---\n")
        f.flush()
        print(l)
        print("\n")        

        id = count_all
        new_data = {}
        new_data['ID']   = id
        new_data['instruction'] = q
        new_data['prompt'] = ptext
        new_data['output'] = res
        new_data['correct_answer'] = o
        new_data['out_str'] = out_str
        new_data['hit'] = hit
        js.append(new_data)

js.append({'elapsed_time': time.time() - st})
filename = f"{output}.json"        
with open(filename,'w', encoding="utf-8") as f:
    json.dump(js,f,indent=2, ensure_ascii=False)
