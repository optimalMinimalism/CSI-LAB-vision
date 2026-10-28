import json
import shutil
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest.mock import Mock

from backend.server import Receiver
from backend.service import Service
from backend.storage import Store
from framegrabber.camera import CameraService
from framegrabber.protocol import request
from testSystem import FakeCamera
from testVision import FakePose
from backend.vision import VisionPipeline
import cv2
import numpy as np

TOKEN = 'video-only-test-token-123456'


class VideoDashboardTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = Store(self.directory.name)
        self.service = Service(self.store, stop_timeout=2)
        self.server = Receiver(('127.0.0.1', 0), self.service, TOKEN)
        self.base = 'http://127.0.0.1:' + str(self.server.server_port)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.camera = CameraService(self.base, TOKEN, source_factory=FakeCamera)
        self.camera_thread = threading.Thread(target=self.camera.run)
        self.camera_thread.start()

    def tearDown(self):
        self.service.close()
        self.camera.stop()
        self.camera_thread.join(4)
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()
        self.store.close()
        self.directory.cleanup()

    def api(self, path, method='GET', data=None):
        return request(self.base, TOKEN, '/api/' + path, method, data)

    def wait_for(self, check):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if check():
                return
            time.sleep(.05)
        self.fail('Timed out waiting for video')

    def test_preview_does_not_create_recordings(self):
        self.api('camera/preview', 'POST', {'enabled': True})
        self.wait_for(lambda: self.service.preview_frame is not None)
        self.assertEqual(self.store.list(), [])
        self.api('camera/preview', 'POST', {'enabled': False})
        self.assertIsNone(self.service.preview_frame)

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg required')
    def test_default_mp4_and_combination_contain_burned_overlays_and_timestamp(self):
        class OverlayCamera(FakeCamera):
            def read(self):
                return True, np.zeros((240, 320, 3), np.uint8)
        self.camera.factory = OverlayCamera
        self.service.enable_live(lambda: VisionPipeline(FakePose()))
        item = self.api('recordings', 'POST', {'title': 'Overlay export'})
        identifier = item['id']
        self.wait_for(lambda: len(self.store.frames(identifier)) >= 3)
        self.api('recordings/' + identifier + '/stop', 'POST', {})
        for query in ('', '?mode=posture,skeleton', '?mode=none'):
            req = urllib.request.Request(self.base + '/api/recordings/' + identifier + '/video' + query,
                                         headers={'Authorization': 'Bearer ' + TOKEN})
            with urllib.request.urlopen(req, timeout=30) as response:
                payload = response.read()
            path = self.directory.name + '/export.mp4'
            with open(path, 'wb') as video:
                video.write(payload)
            capture = cv2.VideoCapture(path)
            try:
                ok, frame = capture.read()
                self.assertTrue(ok)
                self.assertGreater(frame[-24:].max(), 80)
                if query != '?mode=none':
                    self.assertGreater(frame[95:105, 19:22].max(), 80)
                    self.assertGreater(frame[95:105, 49:52].max(), 80)
            finally:
                capture.release()

    def test_video_recording_and_preview_continues(self):
        self.api('camera/preview', 'POST', {'enabled': True})
        item = self.api('recordings', 'POST', {'title': 'Solo video'})
        identifier = item['id']
        self.wait_for(lambda: len(self.store.frames(identifier)) >= 3)
        with self.assertRaises(urllib.error.HTTPError):
            self.api('recordings/' + identifier + '/csi', 'POST', {})
        result = self.api('recordings/' + identifier + '/stop', 'POST', {})
        self.assertEqual(result['state'], 'STOPPED')
        self.assertTrue(result['camera_ack'])
        self.assertTrue(self.service.control()['preview_enabled'])
        self.assertEqual(len(self.api('recordings')), 1)
        if shutil.which('ffmpeg'):
            req = urllib.request.Request(self.base + '/api/recordings/' + identifier + '/video?mode=raw', headers={'Authorization': 'Bearer '+TOKEN})
            with urllib.request.urlopen(req, timeout=15) as response:
                self.assertEqual(response.headers['Content-Type'], 'video/mp4')
                payload = response.read()
                self.assertIn(b'ftyp', payload[:32])
                self.assertGreater(len(payload), 1000)
