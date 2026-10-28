"""Live posture stage, reusing the existing conservative COCO geometry."""
from .inference import posture


class PostureDetector:
    def predict(self, people):
        results = []
        for index, person in enumerate(people):
            state, score = posture(person)
            results.append(dict(person_index=index, posture=state, heuristic_score=score))
        return results
