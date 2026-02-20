from transformers import pipeline


class SentimentAnalyzer:

    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.analyzer = pipeline("sentiment-analysis", model=model_name)

    def analyze_text(self, text):
        return self.analyzer(text)[0]

    def analyze_chunks(self, chunks):
        results = []

        for i, chunk in enumerate(chunks):
            sentiment = self.analyzer(chunk)[0]

            results.append({
                "chunk_id": i,
                "label": sentiment["label"],
                "score": float(sentiment["score"])
            })

        return results

    def aggregate_sentiment(self, chunk_results):
        positive = 0
        negative = 0

        for result in chunk_results:
            if result["label"] == "POSITIVE":
                positive += 1
            else:
                negative += 1

        if positive > negative:
            overall = "POSITIVE"
        elif negative > positive:
            overall = "NEGATIVE"
        else:
            overall = "NEUTRAL"

        return {
            "overall_sentiment": overall,
            "positive_chunks": positive,
            "negative_chunks": negative
        }