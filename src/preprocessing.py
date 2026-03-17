import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)


class TextPreprocessor:

    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.fillers = ["um", "uh", "you know", "like", "basically", "actually"]

    def basic_cleaning(self, text):
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text

    def remove_fillers(self, text):
        for filler in self.fillers:
            text = text.replace(filler, "")
        return text

    def tokenize_sentences(self, text):
        return sent_tokenize(text)

    def tokenize_words(self, text):
        return word_tokenize(text)

    def remove_stopwords(self, words):
        return [
            word for word in words
            if word.isalpha() and word not in self.stop_words
        ]

    def full_preprocess(self, text):
        text = self.basic_cleaning(text)
        text = self.remove_fillers(text)
        words = self.tokenize_words(text)
        filtered_words = self.remove_stopwords(words)
        return " ".join(filtered_words)
