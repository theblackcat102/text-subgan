import os, glob
import fasttext



def pretrain_fasttext():
    source_dir = 'data/kkday_dataset/user_data/prod_desc_after/'
    g = open('data/kkday_dataset/corpus.txt', 'w')
    for filename in glob.glob(source_dir+'*.txt'):
        target = os.path.basename(filename)
        with open(filename, 'r') as f:
            for line in f.readlines():
                g.write(line.strip()+'\n')

    source_dir = 'data/kkday_dataset/user_data/prod_title_after/'
    for filename in glob.glob(source_dir+'*.txt'):
        target = os.path.basename(filename)
        with open(filename, 'r') as f:
            for line in f.readlines():
                g.write(line.strip().replace('\u3000', ' ')+'\n')
    g.close()
    model = fasttext.train_unsupervised('data/kkday_dataset/corpus.txt', 
        model='skipgram', dim=128, min_count=1, epoch=20)
    model.save_model('data/kkday_dataset/model-128.bin')
def is_alpha(word):
    try:
        return word.encode('ascii').isalpha()
    except:
        return False

def match_template():

    match = {}
    english_token = ['天鵝堡', 'é', '山麓', '守夜人', '樂園', '鐵塔', '墓穴', '交通', '瘋馬', '高鐵']
    free_token = [
        'JR', 'PASS', 'WiFi', 'Spa', 'G', 'BBQ', 'Hotel', 'airport', 'easy', 'buffet', 'GB', 'CARD', 'IG', 'Ticket', 'Game', '4G'
    ]
    
    for dataset in ['test', 'train', 'valid']:
        with open(f'data/kkday_dataset/{dataset}_title.txt', 'r') as f, open(f'data/kkday_dataset/{dataset}_template.txt', 'r') as t:
            title_lines = f.readlines()
            template_lines = t.readlines()
            
            for (title, template) in zip(title_lines, template_lines):
                title = title.strip()
                template = template.strip()
                match[title] = template
                new_token = []
                for t in template.split(' '):
                    if (is_alpha(t) or t in english_token) and t not in free_token:
                        english_token.append(t)
                        new_token.append('##')
                    else:
                        new_token.append(t)
                template = ' '.join(new_token)
                match[title] = template

    source_dir = 'data/kkday_dataset/user_data/prod_title_after/'
    os.makedirs('data/kkday_dataset/user_data/prod_template_after/', exist_ok=True)
    match_cnt, cnt = 0, 0
    for filename in glob.glob(source_dir+'*.txt'):
        target = os.path.basename(filename)
        output_filename = os.path.join('data/kkday_dataset/user_data/prod_template_after/', target)
        with open(filename, 'r') as f:
            lines = []
            for line in f.readlines():
                lines.append(line.strip().replace('\u3000', ' '))
            cnt += 1
            title = ''.join(lines)
        if title in match:
            with open(output_filename, 'w') as f:
                f.write(match[title])
            match_cnt += 1
    print(f'fit {match_cnt}/{cnt}')

if __name__ == "__main__":
    match_template()