"""Live skeleton stage, reusing the existing COCO joint/segment generator."""
from .skeleton import geometry


class SkeletonGenerator:
    def predict(self, people, width, height):
        return [dict(person_index=index, **geometry(person, width, height))
                for index, person in enumerate(people)]
