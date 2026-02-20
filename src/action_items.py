import re


class ActionItemExtractor:

    def __init__(self):
        self.action_keywords = [
            "will",
            "need to",
            "should",
            "must",
            "assign",
            "responsible",
            "follow up",
            "complete",
            "deadline",
            "by"
        ]

        self.decision_keywords = [
            "decided",
            "agreed",
            "approved",
            "finalized",
            "confirmed"
        ]

    def extract_sentences(self, text):
        sentences = re.split(r'[.!?]', text)
        return [s.strip() for s in sentences if len(s.strip()) > 0]

    def extract_action_items(self, text):
        sentences = self.extract_sentences(text)

        action_items = []

        for sentence in sentences:
            for keyword in self.action_keywords:
                if keyword in sentence.lower():
                    action_items.append(sentence)
                    break

        return action_items

    def extract_decisions(self, text):
        sentences = self.extract_sentences(text)

        decisions = []

        for sentence in sentences:
            for keyword in self.decision_keywords:
                if keyword in sentence.lower():
                    decisions.append(sentence)
                    break

        return decisions

    def structured_output(self, text):
        return {
            "action_items": self.extract_action_items(text),
            "decisions": self.extract_decisions(text)
        }