"""Presence classification from the shared person detector output."""


class PresenceClassifier:
    def predict(self, people):
        return {
            "state": "PRESENT" if people else "ABSENT",
            "person_count": len(people),
            "confidence": max((person["confidence"] for person in people), default=None),
        }
