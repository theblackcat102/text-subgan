import logging
import os
from collections import defaultdict
from logging.handlers import RotatingFileHandler
import torch
from tokenizer import WordTokenizer
import random
from tqdm import tqdm
FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)

logger = logging.getLogger()


from opts import preprocess_opts

def preprocess_data(src_text, tgt_text, template_txt, user_data, cache_path, corpus_type):

    logger.info("Build user data mapping")
    user2id = defaultdict(int)
    prod2id = defaultdict(int)
    neg_user2product = defaultdict(list)
    user2product, product2user = defaultdict(list), defaultdict(list)
    user2product_mask = {}
    if corpus_type not in ('valid', 'test' ): # load validation user2product mapping and mask from training 
        for name in ['valid.pt', 'test.pt']:
            valid_data = torch.load(os.path.join(cache_path, name))
            
            for e in valid_data:
                if len(e[3]) > 0:
                    for (prod, user) in e[3]:
                        if user not in user2product_mask:
                            user2product_mask[user] = {}
                        if prod not in user2product_mask[user]:
                            user2product_mask[user][prod] = 1

    title2product = defaultdict(list)

    with open(os.path.join(user_data, 'user_records.txt'), 'r') as f, open('found_title.txt', 'w') as g:
        for line in f.readlines():
            user_id, products = line.strip().split(',', maxsplit=1)
            if user_id not in user2id:
                user2id[user_id] = len(user2id)

            for prod_id in products.split(','):
                if len(prod_id) == 0:
                    continue

                if prod_id not in prod2id:
                    prod2id[prod_id] = len(prod2id)

                uid = user2id[user_id]
                pid = prod2id[prod_id]

                if (uid not in user2product_mask) or (uid in user2product_mask and pid not in user2product_mask[uid]):
                    user2product[ uid ].append( pid )
                    product2user[ pid ].append( uid )

                if os.path.exists(os.path.join(user_data, 'prod_title_after/prod_title_'+prod_id+'.txt' )):
                    with open(os.path.join(user_data, 'prod_title_after/prod_title_'+prod_id+'.txt' ), 'r') as f:
                        title = f.readline()
                        title = title.strip().replace('\u3000', ' ')
                        # if title in title2product:
                        #     logger.warning('exist before %s, %s' % (prod_id, title2product[title]))
                        title2product[title].append(pid)
                        g.write(title+'\n')
    hit = 0
    logger.info('total title2product %d, total products %d' % (len(title2product), len(prod2id)))  
    data = []
    with open(src_text, 'r') as src_f, open(tgt_text, 'r') as tgt_f, open(template_txt, 'r') as tmp_f:
        for (src, tgt, tmt ) in zip( src_f.readlines(), tgt_f.readlines(), tmp_f.readlines() ):
            user_prod_pair = []
            src, tgt, tmt = src.strip(), tgt.strip(), tmt.strip()
            if tgt in title2product:
                for prod in title2product[tgt]:
                    if prod in product2user:
                        for user in product2user[ prod ]:
                            hit += 1
                            user_prod_pair.append( [prod, user] )
            data.append( [ src, tgt, tmt,  user_prod_pair ] )

    print('user2product : ', len(user2product))
    print('product2user : ', len(product2user))
    print('hit rate:    ', hit)

    torch.save(data, os.path.join(cache_path, corpus_type+'.pt'))
    print('user size : ', len(user2id))
    print('prod size : ', len(prod2id))

    if corpus_type == 'train':
        for user_id, products in tqdm(user2product.items(), dynamic_ncols=True):
            neg_prods = []
            start = 0
            while len(neg_prods) < 100:
                rand_prod = random.randint(0, len(prod2id))
                # print(rand_prod, products)
                if rand_prod not in products:
                    neg_prods.append(rand_prod)
            neg_user2product[user_id] = neg_prods


        id_mapping = {
            'user2id': user2id,
            'prod2id': prod2id,
            'user2product': user2product,
            'product2user': product2user,
            'neg_user2product': neg_user2product,
            'title2product': title2product,
        }
        torch.save(id_mapping, os.path.join(cache_path, 'id_mapping.pt'))

        corpus = []

        for d in data:
            corpus.append(d[0])
            corpus.append(d[1])
            corpus.append(d[2])
        tokenizer = WordTokenizer(corpus=corpus, postfix='tempest_word')
        


def _get_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    preprocess_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    preprocess_data(opt.valid_src, opt.valid_tgt, opt.valid_template, opt.user_data, 
        cache_path=opt.save_data, corpus_type='valid')
    preprocess_data(opt.test_src, opt.test_tgt, opt.test_template, opt.user_data, 
        cache_path=opt.save_data, corpus_type='test')
    preprocess_data(opt.train_src, opt.train_tgt, opt.train_template, opt.user_data, 
        cache_path=opt.save_data, corpus_type='train')
