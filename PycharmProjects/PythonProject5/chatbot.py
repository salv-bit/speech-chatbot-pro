# chatbot.py — Q/A pairing so the bot matches only questions and returns their answers
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def simple_analyzer(text: str):
    if not text:
        return []
    return re.findall(r"\b\w+\b", text.lower())


def parse_qa(corpus_text: str):
    """
    Parse Q/A pairs from corpus_text with lines like:
    Q: question text
    A: answer text
    (blank lines allowed between pairs)

    Returns (questions, answers). If no pairs found, returns ([], []).
    """
    questions, answers = [], []
    q, a = None, None

    for raw in (corpus_text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lower().startswith("q:"):
            if q is not None and a is not None:
                questions.append(q)
                answers.append(a)
            q = line[2:].strip()
            a = None
        elif line.lower().startswith("a:"):
            a = line[2:].strip()
        else:
            if a is not None:
                a += " " + line
            elif q is not None:
                q += " " + line

    if q is not None and a is not None:
        questions.append(q)
        answers.append(a)

    return questions, answers


class SimpleChatbot:
    """
    Retrieval-based FAQ bot:
    - Builds TF-IDF only over FAQ **questions**
    - Returns the paired **answer** with the highest similarity
    """
    def __init__(self, corpus_text: str):
        qs, ans = parse_qa(corpus_text)
        if qs and ans and len(qs) == len(ans):
            self.questions = qs
            self.answers = ans
            self.vectorizer = TfidfVectorizer(analyzer= simple_analyzer)
            self.q_matrix = self.vectorizer.fit_transform(self.questions)
            self.mode = "qa"
        else:
            self.mode = "sentences"
            sentences = self._split_sentences(corpus_text or "")
            if not sentences:
                sentences = [(corpus_text or "I have no data yet.").strip()]
            self.sentences = sentences
            self.vectorizer = TfidfVectorizer(analyzer= simple_analyzer)
            self.q_matrix = self.vectorizer.fit_transform(self.sentences)

    def reply(self, user_text: str) -> str:
        user_text = (user_text or "").strip()
        if not user_text:
            return "Say something and I'll try to help!"

        user_vec = self.vectorizer.transform([user_text])
        sims = cosine_similarity(user_vec, self.q_matrix).flatten()
        idx = int(np.argmax(sims))
        score = float(sims[idx])

        if score < 0.10:
            return "I'm not sure I understood. Could you rephrase?"

        if self.mode == "qa":
            return self.answers[idx]
        else:
            return self.sentences[idx]

    @staticmethod
    def _split_sentences(text: str):
        text = re.sub(r"\s+", " ", text.strip())
        if not text:
            return []
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p for p in parts if p]
