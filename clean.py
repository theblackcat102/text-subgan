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

    '''
    What need to be clean: place, time(season), item
    '''
    match = {}
    english_token = ['天鵝堡', 'é', '山麓', '守夜人', '樂園', '鐵塔', '墓穴', '交通', '瘋馬', '高鐵', '101', '門票', '蜜月島', '夕陽',  '稻荷', '華欣泰式',
        '遊艇', '沙巴', '台灣', '韓國', '東京', '環島', 'ATV', '紅谷', '秋', '紅樹林', 'Disneyland', 'Selvatica', 'Playa', 'Carmen', 'del', '文化',
        'Disneyland', '喜劇', 'COAST', 'Cenote', "A'DAM", 'Dinawan', 'Island', 'Discovery', '巴登汗蒸幕', '青海湖', '大象', '珊瑚礁', '潛艇', '郵輪',
        '新加坡水', '武士刀', '长隆欢乐', '广州亲子同乐', '叢林', '野戰', '龍舌蘭酒', 'SeaWorld', '®', '晴空塔', '石門山', '薄荷島', '叢林', '諾', '皇帝島',
        '珊瑚島', '浮潛', '清邁泰式', '山野溪','響尾蛇','山打根','仙台拼車', '櫻花','賞櫻', '袋鼠', '潑水節', '摩天輪', '外國人', '墨西哥城', '雪梨塔',
        '大洋路', '巨巖','廣州塔','港臺','長鼻猴','漢堡', '企鵝', '菲利浦島','馬蹄灣','羚羊','富國島', '九', '族', 'Segway', '科羅拉多河', '彩穴','瀑布',
        '泰國菜', '琵琶湖', '越南版', '龍船', '南麓', '法拉利', '半島', '宇登呂', '溫泉','川湯', '菲利普島', '皮皮島', '蛋', '麻辣', '鍋', '鱒魚', '冰釣',
        '簸箕船', '滑雪團', '後裔', '葡萄酒', '明洞', '三十三', '巧克力', '耶', '爾', '森林','基督城','金沙城','浮潛','占婆塔','大峡谷南缘', '音樂劇', '環球',
        '美甲美睫', '大峡谷南缘','玫瑰谷','海釣','釣魚樂','電視', '尼克號', '泰坦', '人妖秀', '原住民', '寺廟', '雲石寺', '大鹿', '冬季', '烏岩角', '冰魚',
        '破冰船', '卡迪灣', '極光號', '海蝕洞', '1250', '台灣', '暹粒','銀杏','清邁泰式','大倉山', '咖啡豆', '觀光塔', '°', '360', '180','空叻瑪榮水', '捷運線',
        '單程', '車票', '櫻花','大阪卡', '一日券','中式', '點心', '午餐', '套餐','野花', 'Mulle-gil', '十分', '黑潮', 'Spa', '按摩', '泰式', '音樂劇', '獅子王'
        '梅花鹿' , '雙嶼水', '獨木舟', '立槳', '巖頭鎮', '芬多精', '清水寺', '高台寺', '飛驒', '威尼斯人', '貢', '藏王', '樹冰', '叮叮', '車', '按摩', '賽車', '魔術秀']
    free_token = [
        'JR', 'PASS', 'WiFi', 'Spa', 'G', 'BBQ', 'Hotel', 'airport', 'easy', 'buffet', 'GB', 'CARD', 'IG', 'Ticket', 'Game', '4G', 'VS'
    ]
    
    no_mask = 0
    for dataset in ['test', 'train', 'valid']:
        with open(f'data/kkday_dataset/{dataset}_title.txt', 'r') as f, open(f'data/kkday_dataset/{dataset}_template.txt', 'r') as t:
            title_lines = f.readlines()
            template_lines = t.readlines()
            
            for (title, template) in zip(title_lines, template_lines):
                title = title.strip()
                template = template.strip()
                if '##' not in template:
                    no_mask += 1
                match[title] = template
                new_token = []
                for t in template.split(' '):
                    if (is_alpha(t) or t in english_token or '洞' in t or '寺' in t or '谷' in t or '岩' in t or '湖' in t or '山' in t or '島' in t) and t not in free_token:
                        # english_token.append(t)
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
    print(f'No mask template {no_mask}')

if __name__ == "__main__":
    match_template()