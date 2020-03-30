import inspect
import re
import textwrap

WRITER_PATTERN = textwrap.dedent(
    '''
    `%s`

    `%s`

    ---
    ''')

CACHE_DIR = "./cache"
MAX_LENGTH = 50

PTT_DATA = './data/批踢踢實業坊_title_only/*.txt'
PTT_WHITE_LIST = [
    '批踢踢實業坊_MakeUp_板.txt',
    '批踢踢實業坊_Boy-Girl_板.txt',
    '批踢踢實業坊_Gossiping_板.txt',
    '批踢踢實業坊_StupidClown_板.txt',
]

DCARD_DATA = './data/Dcard_title_only/*.txt'
DCARD_WHITE_LIST = [
    'makeup.txt',
    'mood.txt',
    'talk.txt',
    'funny.txt',
    'relationship.txt',
]

EMOJI_PATTERN = re.compile(
    u"["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    u"]+", flags=re.UNICODE)


class Constants:
    PAD = 0
    UNK = 1
    BOS = 2
    EOS = 3
    URL = 4
    SIMLE = 5
    XD = 6
    SAD = 7

    PAD_WORD = '</PAD>'
    UNK_WORD = '</UNK>'
    BOS_WORD = '</BOS>'
    EOS_WORD = '</EOS>'
    URL_WORD = '</URL>'
    SIMLE_WORD = '</SIMLE>'
    XD_WORD = '</XD>'
    SAD_WORD = '</SAD>'

    PAD_EX = ''
    UNK_EX = '涆'
    BOS_EX = ''
    EOS_EX = ''
    URL_EX = 'https://github.com/w86763777/deeplearning-final-project'
    SIMLE_EX = ':)'
    XD_EX = 'XD'
    SAD_EX = ':('

    @classmethod
    def word2idx(clazz):
        attrs = inspect.getmembers(
            clazz, lambda attr: not(inspect.isroutine(attr)))
        word2idx = {}
        for name, v in attrs:
            if not name.endswith('__') and not name.startswith('__'):
                if name.endswith('_WORD') or name.endswith('_EX'):
                    continue
                word2idx['</%s>' % name] = v
        return word2idx

    @classmethod
    def idx2word(clazz):
        return {v: k for k, v in clazz.word2idx().items()}

    @classmethod
    def idx2example(clazz):
        attrs = inspect.getmembers(
            clazz, lambda attr: not(inspect.isroutine(attr)))
        attrs = dict(list(attrs))
        idx2example = {}
        for name, v in attrs.items():
            if not name.endswith('__') and not name.startswith('__'):
                if name.endswith('_WORD') or name.endswith('_EX'):
                    continue
                idx2example[v] = attrs['%s_EX' % name]
        return idx2example


if __name__ == "__main__":
    print(Constants.word2idx())
    print(Constants.idx2word())
    print(Constants.idx2example())