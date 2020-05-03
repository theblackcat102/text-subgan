import os, glob
import fasttext


if __name__ == "__main__":
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
