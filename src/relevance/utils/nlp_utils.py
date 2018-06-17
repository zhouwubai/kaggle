import re
import nltk
from bs4 import BeautifulSoup
from relevance.features.replacer import CsvWordReplacer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from relevance.config import config

# stop words
stopwords = nltk.corpus.stopwords.words("english")
stopwords = set(stopwords)


# stemming
if config.stemmer_type == "porter":
    english_stemmer = nltk.stem.PorterStemmer()
elif config.stemmer_type == "snowball":
    english_stemmer = nltk.stem.SnowballStemmer('english')


def stem_tokens(tokens, stemmer):
    stemmed = []
    for token in tokens:
        stemmed.append(stemmer.stem(token))
    return stemmed


# pos tag
token_pattern = r"(?u)\b\w\w+\b"


def pos_tag_text(line,
                 token_pattern=token_pattern,
                 exclude_stopword=config.cooccurrence_word_exclude_stopword,
                 encode_digit=False):
    token_pattern = re.compile(token_pattern, flags=re.UNICODE | re.LOCALE)
    for name in ["query", "product_title", "product_description"]:
        l_val = line[name]
        # tokenize
        tokens = [x.lower() for x in token_pattern.findall(l_val)]
        # stem
        tokens = stem_tokens(tokens, english_stemmer)
        if exclude_stopword:
            tokens = [x for x in tokens if x not in stopwords]
        tags = pos_tag(tokens)
        tags_list = [t for w, t in tags]
        tags_str = " ".join(tags_list)
        # print tags_str
        line[name] = tags_str
    return line


# TF-IDF
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


token_pattern = r"(?u)\b\w\w+\b"
tfidf__norm = "l2"
tfidf__max_df = 0.75
tfidf__min_df = 3


def getTFV(token_pattern=token_pattern,
           norm=tfidf__norm,
           max_df=tfidf__max_df,
           min_df=tfidf__min_df,
           ngram_range=(1, 1),
           vocabulary=None,
           stop_words='english'):
    tfv = StemmedTfidfVectorizer(
        min_df=min_df, max_df=max_df, max_features=None,
        strip_accents='unicode', analyzer='word', token_pattern=token_pattern,
        ngram_range=ngram_range, use_idf=1, smooth_idf=1, sublinear_tf=1,
        stop_words=stop_words, norm=norm, vocabulary=vocabulary)
    return tfv


# BOW
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(CountVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


token_pattern = r"(?u)\b\w\w+\b"
bow__max_df = 0.75
bow__min_df = 3


def getBOW(token_pattern=token_pattern,
           max_df=bow__max_df,
           min_df=bow__min_df,
           ngram_range=(1, 1),
           vocabulary=None,
           stop_words='english'):
    bow = StemmedCountVectorizer(
        min_df=min_df, max_df=max_df, max_features=None,
        strip_accents='unicode', analyzer='word', token_pattern=token_pattern,
        ngram_range=ngram_range,
        stop_words=stop_words, vocabulary=vocabulary)
    return bow


# Text Clean ##
# synonym replacer
replacer = CsvWordReplacer('%s/synonyms.csv' % config.data_folder)
# other replace dict
# such dict is found by exploring the training data
replace_dict = {
    "nutri system": "nutrisystem",
    "soda stream": "sodastream",
    "playstation's": "ps",
    "playstations": "ps",
    "playstation": "ps",
    "(ps 2)": "ps2",
    "(ps 3)": "ps3",
    "(ps 4)": "ps4",
    "ps 2": "ps2",
    "ps 3": "ps3",
    "ps 4": "ps4",
    "coffeemaker": "coffee maker",
    "k-cups": "k cup",
    "k-cup": "k cup",
    "4-ounce": "4 ounce",
    "8-ounce": "8 ounce",
    "12-ounce": "12 ounce",
    "ounce": "oz",
    "button-down": "button down",
    "doctor who": "dr who",
    "2-drawer": "2 drawer",
    "3-drawer": "3 drawer",
    "in-drawer": "in drawer",
    "hardisk": "hard drive",
    "hard disk": "hard drive",
    "harley-davidson": "harley davidson",
    "harleydavidson": "harley davidson",
    "e-reader": "ereader",
    "levi strauss": "levi",
    "levis": "levi",
    "mac book": "macbook",
    "micro-usb": "micro usb",
    "screen protector for samsung": "screen protector samsung",
    "video games": "videogames",
    "game pad": "gamepad",
    "western digital": "wd",
    "eau de toilette": "perfume",
}


def clean_text(line, drop_html_flag=False):
    names = ["query", "product_title", "product_description"]
    for name in names:
        l_val = line[name]
        if drop_html_flag:
            l_val = drop_html(l_val)
        l_val = l_val.lower()
        # replace gb
        for vol in [16, 32, 64, 128, 500]:
            l_val = re.sub("%d gb" % vol, "%dgb" % vol, l_val)
            l_val = re.sub("%d g" % vol, "%dgb" % vol, l_val)
            l_val = re.sub("%dg " % vol, "%dgb " % vol, l_val)
        # replace tb
        for vol in [2]:
            l_val = re.sub("%d tb" % vol, "%dtb" % vol, l_val)

        # replace other words
        for k, v in replace_dict.items():
            l_val = re.sub(k, v, l_val)
        l_val = l_val.split(" ")

        # replace synonyms
        l_val = replacer.replace(l_val)
        l_val = " ".join(l_val)
        line[name] = l_val
    return line


def drop_html(html):
    return BeautifulSoup(html, "html5lib").get_text(separator=" ")
