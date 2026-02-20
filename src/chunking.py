import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt", quiet=True)


class TextChunker:

    def __init__(self, max_words=200):
        self.max_words = max_words

    def split_sentences(self, text):
        return sent_tokenize(text)

    def create_chunks(self, text):
        sentences = self.split_sentences(text)

        chunks = []
        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            word_count = len(sentence.split())

            if current_word_count + word_count <= self.max_words:
                current_chunk.append(sentence)
                current_word_count += word_count
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_word_count = word_count

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks