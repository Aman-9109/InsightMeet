from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


class TopicModeler:

    def __init__(self, num_topics=2, num_keywords=5):
        self.num_topics = num_topics
        self.num_keywords = num_keywords

    # 🔹 TF-IDF Keyword Extraction
    def extract_keywords(self, text):
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=1000
        )

        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()

        scores = tfidf_matrix.toarray()[0]

        word_scores = list(zip(feature_names, scores))
        word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)

        top_keywords = [word for word, score in word_scores[:self.num_keywords]]

        return top_keywords

    # 🔹 LDA Topic Modeling
    def extract_topics(self, text):
        vectorizer = CountVectorizer(stop_words="english")
        count_matrix = vectorizer.fit_transform([text])

        lda = LatentDirichletAllocation(
            n_components=self.num_topics,
            random_state=42
        )

        lda.fit(count_matrix)

        words = vectorizer.get_feature_names_out()
        topics = []

        for topic in lda.components_:
            top_words = [
                words[i]
                for i in topic.argsort()[-self.num_keywords:]
            ]
            topics.append(top_words)

        return topics

    def structured_output(self, text):
        return {
            "keywords": self.extract_keywords(text),
            "topics": self.extract_topics(text)
        }