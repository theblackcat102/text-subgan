import requests
from tqdm import tqdm
import json
from NLP_tools import rouge, bleu

bleu = bleu.BLEU()
rouge = rouge.Rouge()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate tools')
    parser.add_argument('-r','--ref', type=str)
    parser.add_argument('-g','--gen', type=str)

    args = parser.parse_args()
    ref_file = args.ref
    system_file = args.gen
    rouge.print_score(ref_file, system_file)

    # for complete version of rouge score
    rouge.print_all(ref_file, system_file)

    # for bleu score
    # None, sm1~sm7 denotes smoothing function type
    bleu.print_score(ref_file, system_file, "sm3")
    bleu.print_score(ref_file, system_file, "sm5")