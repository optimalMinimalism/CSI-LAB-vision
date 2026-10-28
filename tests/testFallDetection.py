import unittest

from backend.fall_detection import FallDetector
from backend.vision import VisionWorker
from framegrabber.protocol import request
from testVision import FakePose, VisionPipeline, wait_until, TOKEN
from backend.server import Receiver
from backend.service import Service
from backend.storage import Store
import tempfile
import threading
import numpy as np


class FallTests(unittest.TestCase):
    def frame(self, detector, seconds, posture='STANDING', box=(0, 0, 80, 200), source='camera'):
        return detector.predict([dict(bbox=box)], [dict(posture=posture)],
                                dict(monotonic_ns=int(seconds*1e9), source_id=source))['state']

    def test_rapid_drop_requires_sustained_lying(self):
        detector = FallDetector()
        self.assertEqual(self.frame(detector, 1), 'MONITORING')
        self.assertEqual(self.frame(detector, 1.5, 'LYING_DOWN', (0, 150, 200, 230)), 'UNKNOWN')
        self.assertEqual(self.frame(detector, 1.9, 'LYING_DOWN', (0, 150, 200, 230)), 'SUSPECTED_FALL')
        detector.reset()
        self.assertEqual(self.frame(detector, 2, 'LYING_DOWN', (0, 150, 200, 230)), 'UNKNOWN')

    def test_discontinuities_and_ambiguity_reset_history(self):
        for change in ('gap', 'source', 'unknown', 'multiple', 'reverse'):
            detector = FallDetector()
            self.frame(detector, 1)
            if change == 'unknown':
                self.frame(detector, 1.1, 'UNKNOWN')
            elif change == 'multiple':
                detector.predict([], [], dict(monotonic_ns=1_100_000_000))
            stamp = 4 if change == 'gap' else .5 if change == 'reverse' else 1.5
            source = 'other' if change == 'source' else 'camera'
            self.assertEqual(self.frame(detector, stamp, 'LYING_DOWN', (0, 150, 200, 230), source), 'UNKNOWN')

    def test_fall_overlay_is_visible_only_when_enabled(self):
        from backend.vision import annotate
        from unittest.mock import patch
        import cv2
        image, people, analysis = VisionPipeline(FakePose()).predict(np.zeros((240, 320, 3), np.uint8))
        metadata = dict(timestamp_ns=1_700_000_000_000_000_000)
        with patch('backend.vision.cv2.putText', wraps=cv2.putText) as draw:
            annotate(image, people, analysis, metadata, 'none')
            self.assertFalse(any('Cadute' in c.args[1] for c in draw.call_args_list))
            analysis['fall_detection'] = dict(enabled=True, state='SUSPECTED_FALL')
            annotate(image, people, analysis, metadata, 'none')
            self.assertTrue(any('POSSIBILE CADUTA' in c.args[1] for c in draw.call_args_list))
        self.assertFalse(image.any())

    def test_worker_toggle_clears_results_and_history(self):
        worker = VisionWorker(lambda: VisionPipeline(FakePose()))
        def submit(index):
            worker.submit(dict(frame_id=index, timestamp_ns=index, monotonic_ns=index),
                          b'', np.zeros((240, 320, 3), np.uint8))
        try:
            submit(1)
            wait_until(lambda: worker.snapshot()[0] is not None)
            self.assertEqual(worker.snapshot()[0][2]['fall_detection']['state'], 'DISABLED')
            worker.set_fall_detection(True)
            self.assertIsNone(worker.snapshot()[0])
            submit(2)
            wait_until(lambda: worker.snapshot()[0] is not None)
            self.assertEqual(worker.snapshot()[0][2]['fall_detection']['state'], 'MONITORING')
            worker.set_fall_detection(False)
            self.assertIsNone(worker.snapshot()[0])
            self.assertIsNone(worker.fall_detector.upright)
            submit(3)
            wait_until(lambda: worker.snapshot()[0] is not None)
            self.assertEqual(worker.snapshot()[0][2]['fall_detection']['state'], 'DISABLED')
        finally:
            worker.close()

    def test_authenticated_toggle_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            service = Service(store)
            service.enable_live(lambda: VisionPipeline(FakePose()))
            server = Receiver(('127.0.0.1', 0), service, TOKEN)
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            base = 'http://127.0.0.1:'+str(server.server_port)
            try:
                self.assertFalse(request(base, TOKEN, '/api/camera/fall-detection')['enabled'])
                for enabled in (True, False):
                    result = request(base, TOKEN, '/api/camera/fall-detection', 'POST', dict(enabled=enabled))
                    self.assertEqual(result, dict(enabled=enabled, experimental=True))
                    self.assertEqual(service.live_vision.fall_enabled, enabled)
                import urllib.request
                service.set_preview(True)
                service.latest_frame(dict(frame_id=123, timestamp_ns=1_700_000_000_000_000_000,
                    monotonic_ns=123, width=320, height=240), b'raw', np.zeros((240,320,3), np.uint8))
                wait_until(lambda: service.live_vision.snapshot()[0] is not None)
                req = urllib.request.Request(base+'/api/camera/stream?mode=all',
                    headers={'Authorization': 'Bearer '+TOKEN})
                with urllib.request.urlopen(req, timeout=3) as response:
                    self.assertIn('multipart/x-mixed-replace', response.headers['Content-Type'])
                    self.assertEqual(response.readline(), b'--csi-frame\r\n')
                    headers = {}
                    while True:
                        line = response.readline()
                        if line == b'\r\n':
                            break
                        key, value = line.decode().split(':', 1)
                        headers[key] = value.strip()
                    self.assertEqual(headers['X-Frame-Id'], '123')
                    self.assertEqual(headers['X-Timestamp-Ns'], '1700000000000000000')
                    jpeg = response.read(int(headers['Content-Length']))
                    self.assertEqual(jpeg, service.live_vision.snapshot()[0][1]['all'])
                import urllib.error
                with self.assertRaises(urllib.error.HTTPError) as error:
                    request(base, TOKEN, '/api/camera/fall-detection', 'POST', dict(enabled='true'))
                self.assertEqual(error.exception.code, 400)
            finally:
                server.shutdown()
                thread.join()
                server.server_close()
                service.close()
                store.close()
