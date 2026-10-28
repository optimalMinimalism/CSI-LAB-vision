"""Client for camera recordings and visual analysis."""

import urllib.request

from framegrabber.protocol import request


class Client:
    def __init__(self, base, token):
        self.base, self.token = base.rstrip("/"), token

    def call(self, path, method="GET", data=None):
        return request(self.base, self.token, "/api/" + path, method, data, timeout=600)

    def start(self, title="", camera_enabled=True):
        return self.call(
            "recordings", "POST", dict(title=title, camera_enabled=camera_enabled)
        )

    def stop(self, identifier):
        return self.call(f"recordings/{identifier}/stop", "POST", {})


    def analyze(self, identifier):
        return self.call(f"recordings/{identifier}/inference", "POST", {})

    def skeletons(self, identifier):
        return self.call(f"recordings/{identifier}/skeletons")


    def review(self, identifier, label_id, activity, status, reviewer, note):
        return self.call(
            f"recordings/{identifier}/review",
            "POST",
            dict(
                label_id=label_id,
                activity=activity,
                status=status,
                reviewer=reviewer,
                note=note,
            ),
        )

    def export(self, identifier, destination):
        req = urllib.request.Request(
            self.base + f"/api/recordings/{identifier}/export",
            headers={"Authorization": "Bearer " + self.token},
        )
        with (
            urllib.request.urlopen(req, timeout=600) as response,
            open(destination, "wb") as output,
        ):
            while block := response.read(65536):
                output.write(block)
