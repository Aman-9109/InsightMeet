from transformers import pipeline


class MeetingSummarizer:

    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        self.summarizer = pipeline(
            "summarization",
            model=model_name
        )

    def summarize_chunk(self, text):
        summary = self.summarizer(
            text,
            max_length=120,
            min_length=30,
            do_sample=False
        )
        return summary[0]["summary_text"]

    def summarize_chunks(self, chunks):
        chunk_summaries = []

        for i, chunk in enumerate(chunks):
            summary = self.summarize_chunk(chunk)

            chunk_summaries.append({
                "chunk_id": i,
                "summary": summary
            })

        return chunk_summaries

    def refine_summary(self, chunk_summaries):
        combined_text = " ".join(
            [c["summary"] for c in chunk_summaries]
        )

        final_summary = self.summarizer(
            combined_text,
            max_length=150,
            min_length=40,
            do_sample=False
        )

        return final_summary[0]["summary_text"]