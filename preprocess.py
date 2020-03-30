import re
import numpy as np

from transfer.constant import Constants, EMOJI_PATTERN


def handle_emojis(tweet):
    '''
        手動清楚一些emoji??
    '''
    post_emo = '_{}_'.format(Constants.SIMLE_WORD)
    sad_emo = '_{}_'.format(Constants.SAD_WORD)
    xd_emo = '_{}_'.format(Constants.XD_WORD)
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\)|\(:\s?D|:-D)',
                   post_emo, tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(x-?D|X-?D|XD)',  xd_emo, tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)',  post_emo, tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', sad_emo, tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', sad_emo, tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()',  sad_emo, tweet)
    return tweet


def clean_emoji(texts):
    return EMOJI_PATTERN.sub(r'', texts)


def clean_text(texts):
    texts = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', '_{}_'.format(
        Constants.URL_WORD), texts)
    texts = handle_emojis(texts)
    return texts


def segment_text(texts):
    first_pass = texts.split('_')
    second_pass = []
    for section in first_pass:
        if len(section) <= 0:
            continue
        if len(section) > 8:
            second_pass += list(section)
        elif section[-1] == '>' and section[:2] == '</':
            second_pass.append(section)
        else:
            second_pass += list(section)
    return second_pass


def pad_sequence(encoded, chunk_size):
    encoded = encoded[:chunk_size - 2]
    encoded = np.concatenate(([Constants.BOS], encoded, [Constants.EOS]))
    encoded = np.pad(
        encoded,
        pad_width=[0, chunk_size - len(encoded)],
        mode='constant',
        constant_values=Constants.PAD)

    return encoded


def sort_by_length(seq, length, label):
    length, indices = length.sort(descending=True)
    seq = seq[indices]
    label = label[indices]
    return seq, length, label


if __name__ == "__main__":
    PTT_texts = [
        "[店家]捷運六張犁站-Huaguo花果美甲室",
        "[醫療/重疾/殘障] 59歲女/36歲女/還本終身/富邦",
        "[閒聊] (雷)萬福 第十二週71 總會有辦法的！",
    ]
    DCARD_texts = [
        "大一就變成邊緣人是正常的嗎",
        "自然捲也可以有日系捲髮(文長燙髮過程)"
    ]
    for text in PTT_texts:
        texts = clean_text(text)
        segmented = segment_text(texts)
        print(segmented)
    for text in DCARD_texts:
        texts = clean_text(text)
        segmented = segment_text(texts)
        print(segmented)