import requests
from tqdm import tqdm
import json

def ner_requests(text):
    res = requests.get('https://voidful.tech/zhner', params={
        'input': text
    })
    return res.json()

if __name__ == "__main__":
    g = open('ner_example.jsonl', 'a')

    with open('data/kkday_dataset/train_title.txt', 'r') as f, open('data/kkday_dataset/train_template_voidful.txt', 'w') as tmp:
        for line in tqdm(f.readlines()):
            line = line.strip()
            output = ner_requests(line)
            if 'result' in output:
                tokens = line.split(' ')
                cleaned = []
                ners = [e[0] for e in output['result']] 
                for t in tokens:
                    found = 0
                    for n in ners:
                        if t in n or n in t:
                            found = 1
                            break

                    if found == 1:
                        cleaned.append('##')
                    else:
                        cleaned.append(t)
                tmp.write(' '.join(cleaned)+'\n')
                g.write(json.dumps({'input': line, 'output': output['result']})+'\n')

            else:
                tmp.write('\n')
    g.close()