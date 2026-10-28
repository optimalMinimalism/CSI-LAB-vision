import json
import hashlib
import io
import shutil
import tempfile
import threading
import time
import unittest
import urllib.request

import numpy as np
import cv2
from unittest.mock import patch

from backend.export import export_video
from backend.server import Receiver
from backend.service import Service
from backend.storage import Store
from backend.vision import VisionPipeline, VisionWorker, PreviewFrames, annotate
from framegrabber.protocol import encode_frame, request
from testSystem import person, FakeCamera
from framegrabber.camera import CameraService, PREVIEW

TOKEN = 'vision-test-token-123456'


class FakePose:
    def __init__(self, people=None):
        self.people = [person()] if people is None else people
        self.input = None

    def predict(self, image):
        self.input = image
        return self.people


def wait_until(check):
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(.01)
    raise AssertionError('Timed out waiting for inference')


class VisionTests(unittest.TestCase):
    def test_preview_only_renders_requested_combinations_once(self):
        image, people, analysis = VisionPipeline(FakePose()).predict(np.zeros((240,320,3),np.uint8))
        with patch('backend.vision.annotate', wraps=annotate) as render:
            previews = PreviewFrames(image, people, analysis, dict(timestamp_ns=1_700_000_000_000_000_000))
            render.assert_not_called()
            first = previews['all']
            self.assertEqual(previews['all'], first)
            self.assertEqual(render.call_count, 1)
            previews['posture,skeleton']
            self.assertEqual(render.call_count, 2)

    def test_export_renders_recorded_frame_before_encoding(self):
        from unittest.mock import Mock
        _, jpeg = cv2.imencode('.jpg', np.zeros((240, 320, 3), np.uint8))
        payload = jpeg.tobytes()
        frame = dict(frame_id=73, timestamp_ns=1_700_000_000_123_456_789,
                     aligned_timestamp_ns=1_700_000_000_123_456_789,
                     sha256=hashlib.sha256(payload).hexdigest())
        store = Mock()
        store.get.return_value = dict(state='STOPPED')
        store.frames.return_value = [frame]
        store.frame.return_value = (frame, payload)

        def encode(root, frames):
            self.assertEqual(frames, [frame])
            rendered = cv2.imread(str(root / '000000000073.jpg'))
            self.assertGreater(rendered[95:105, 49:52].max(), 80)
            self.assertGreater(rendered[-24:].max(), 80)
            (root / 'video.mp4').write_bytes(b'encoded-video')
            return dict(status='generated')

        with patch('backend.export.encode_video', side_effect=encode):
            output = io.BytesIO()
            export_video(store, 'test', output, VisionPipeline(FakePose()), 'skeleton')
        self.assertEqual(output.getvalue(), b'encoded-video')
        self.assertEqual(store.frame.return_value[1], payload)

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg required')
    def test_mp4_burns_in_selected_overlay_and_preserves_original(self):
        from unittest.mock import Mock
        image = np.zeros((240, 320, 3), np.uint8)
        _, jpeg = cv2.imencode('.jpg', image)
        payload = jpeg.tobytes()
        frame = dict(frame_id=0, timestamp_ns=1_700_000_000_123_456_789,
                     aligned_timestamp_ns=1_700_000_000_123_456_789,
                     sha256=hashlib.sha256(payload).hexdigest())
        store = Mock()
        store.get.return_value = dict(state='STOPPED')
        store.frames.return_value = [frame]
        store.frame.return_value = (frame, payload)
        with tempfile.TemporaryDirectory() as directory:
            for mode in ('presence', 'posture', 'skeleton'):
                output = io.BytesIO()
                export_video(store, 'test', output, VisionPipeline(FakePose()), mode)
                path = directory + '/' + mode + '.mp4'
                with open(path, 'wb') as video:
                    video.write(output.getvalue())
                capture = cv2.VideoCapture(path)
                try:
                    ok, decoded = capture.read()
                    self.assertTrue(ok)
                    self.assertGreater(decoded[-24:].max(), 80)  # capture timestamp
                    if mode == 'presence':
                        self.assertGreater(decoded[8:55, 8:190].max(), 80)
                    elif mode == 'posture':
                        self.assertGreater(decoded[95:105, 19:22].max(), 80)
                    else:
                        self.assertGreater(decoded[95:105, 49:52].max(), 80)
                finally:
                    capture.release()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), frame['sha256'])

    def test_three_stages_reuse_pose_and_handle_monochrome(self):
        model = FakePose()
        pipeline = VisionPipeline(model)
        image = np.zeros((240, 320), np.uint8)
        _, people, result = pipeline.predict(image)
        self.assertEqual(model.input.shape, (240, 320, 3))
        self.assertTrue(np.array_equal(model.input[:,:,0], model.input[:,:,2]))
        self.assertEqual(result['presence']['state'], 'PRESENT')
        self.assertEqual(result['presence']['person_count'], 1)
        self.assertEqual(result['postures'][0]['posture'], 'STANDING')
        self.assertEqual(len(result['skeletons'][0]['joints']), 17)
        self.assertGreater(len(result['skeletons'][0]['bones']), 0)
        self.assertEqual(image.shape, (240,320))
        self.assertEqual(len(people),1)

    def test_modes_draw_only_the_requested_feature(self):
        image, people, analysis = VisionPipeline(FakePose()).predict(np.zeros((240,320,3),np.uint8))
        metadata = dict(timestamp_ns=1_700_000_000_123_456_789)
        images = {mode: cv2.imdecode(np.frombuffer(annotate(image,people,analysis,metadata,mode),np.uint8),cv2.IMREAD_COLOR)
                  for mode in ('presence','posture','skeleton')}
        # Box edge and leg segment lie in separate regions, away from text/timestamps.
        self.assertGreater(images['posture'][95:105,19:22].max(),80)
        self.assertLess(images['posture'][95:105,49:52].max(),20)
        self.assertGreater(images['skeleton'][95:105,49:52].max(),80)
        self.assertLess(images['skeleton'][95:105,19:22].max(),20)
        self.assertLess(images['presence'][95:105,19:55].max(),20)
        self.assertGreater(images['presence'][8:55,8:190].max(),80)
        with patch('backend.vision.cv2.putText', wraps=cv2.putText) as draw:
            empty_image, empty_people, empty = VisionPipeline(FakePose([])).predict(image)
            annotate(empty_image,empty_people,empty,metadata,'presence')
            self.assertIn('NOT PRESENT',[call.args[1] for call in draw.call_args_list])
        self.assertFalse(image.any())

    def test_combined_overlays_and_all(self):
        image, people, analysis = VisionPipeline(FakePose()).predict(np.zeros((240,320,3),np.uint8))
        metadata = dict(timestamp_ns=1_700_000_000_123_456_789)
        for mode in ('posture,skeleton', 'all'):
            rendered = cv2.imdecode(np.frombuffer(annotate(image, people, analysis, metadata, mode), np.uint8), cv2.IMREAD_COLOR)
            self.assertGreater(rendered[95:105,19:22].max(), 80)
            self.assertGreater(rendered[95:105,49:52].max(), 80)
        self.assertEqual(annotate(image, people, analysis, metadata, 'all'),
                         annotate(image, people, analysis, metadata, 'skeleton,presence,posture'))
        raw = cv2.imdecode(np.frombuffer(annotate(image, people, analysis, metadata, 'none'), np.uint8), cv2.IMREAD_COLOR)
        self.assertLess(raw[8:200].max(), 20)

    def test_preview_capture_replaces_old_frame_when_upload_is_stalled(self):
        camera = CameraService('http://unused', TOKEN, fps=120, source_factory=FakeCamera)
        camera.active = PREVIEW
        camera.telemetry[PREVIEW] = dict(captured=0, queue_dropped=0, capture_errors=0)
        thread = threading.Thread(target=camera.capture)
        thread.start()
        try:
            wait_until(lambda: camera.telemetry[PREVIEW]['captured'] >= 5)
        finally:
            camera.stop()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertTrue(camera.queue.empty())
        captured = camera.telemetry[PREVIEW]['captured']
        self.assertEqual(camera.preview_pending[1]['frame_id'], captured-1)
        self.assertEqual(camera.telemetry[PREVIEW]['queue_dropped'], captured-1)

    def test_absence_has_no_postures_or_skeletons(self):
        _, _, result = VisionPipeline(FakePose([])).predict(np.zeros((32,32,3),np.uint8))
        self.assertEqual(result['presence']['state'], 'ABSENT')
        self.assertIsNone(result['presence']['confidence'])
        self.assertEqual(result['postures'], [])
        self.assertEqual(result['skeletons'], [])

    def test_worker_skips_pending_frames_and_preserves_acquisition_time(self):
        started, release = threading.Event(), threading.Event()
        seen = []
        class SlowModel(FakePose):
            def predict(self, image):
                seen.append(int(image[0,0,0]))
                if len(seen)==1:
                    started.set()
                    release.wait(3)
                return []
        worker = VisionWorker(lambda: VisionPipeline(SlowModel()))
        timestamp = 1_700_000_000_123_456_789
        def submit(sequence):
            worker.submit(dict(frame_id=sequence, timestamp_ns=timestamp+sequence,
                               monotonic_ns=123+sequence, width=32,height=32,
                               source_id='mono-test',timestamp_source='test-read'),
                          b'',np.full((32,32),sequence,np.uint8))
        try:
            submit(0)
            self.assertTrue(started.wait(3))
            for sequence in range(1,21):
                submit(sequence)
            release.set()
            wait_until(lambda:worker.snapshot()[1]['processed']==2)
            packet, status = worker.snapshot()
            self.assertEqual(seen,[0,20])
            self.assertEqual(status['skipped'],19)
            self.assertEqual(packet[2]['timestamp_ns'],timestamp+20)
            self.assertEqual(packet[2]['monotonic_ns'],143)
            self.assertEqual(packet[2]['source_id'],'mono-test')
            self.assertGreaterEqual(packet[2]['backend_ms'],packet[2]['inference_ms'])
        finally:
            release.set()
            worker.close()
        self.assertFalse(worker.thread.is_alive())

    def test_model_failure_is_unknown_not_absent(self):
        def broken():
            raise ValueError('missing weights')
        worker = VisionWorker(broken)
        try:
            wait_until(lambda:worker.snapshot()[1]['status']=='error')
            packet,status = worker.snapshot()
            self.assertIsNone(packet)
            self.assertIn('missing weights',status['error'])
        finally:
            worker.close()

    def test_http_preview_and_results_share_exact_capture_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            store=Store(directory)
            service=Service(store)
            service.enable_live(lambda:VisionPipeline(FakePose()))
            server=Receiver(('127.0.0.1',0),service,TOKEN)
            thread=threading.Thread(target=server.serve_forever)
            thread.start()
            base='http://127.0.0.1:'+str(server.server_port)
            try:
                image=np.zeros((240,320,3),np.uint8)
                timestamp=1_700_000_000_123_456_789
                metadata,payload=encode_frame(image,73,timestamp,None)
                metadata['source_id']='monochrome'
                request(base,TOKEN,'/api/camera/preview','POST',{'enabled':True})
                request(base,TOKEN,'/api/camera/preview','POST',payload=payload,metadata=metadata)
                wait_until(lambda:service.live_vision.snapshot()[1]['processed']==1)
                req=urllib.request.Request(base+'/api/camera/preview',headers={'Authorization':'Bearer '+TOKEN})
                with urllib.request.urlopen(req) as response:
                    frame=json.loads(response.headers['X-Frame-Metadata'])
                    self.assertTrue(response.read().startswith(b'\xff\xd8'))
                    self.assertEqual(frame['timestamp_ns'],str(timestamp))
                    self.assertEqual(frame['monotonic_ns'],str(metadata['monotonic_ns']))
                    self.assertEqual(frame['frame_id'],73)
                    self.assertEqual(frame['analysis']['presence']['state'],'PRESENT')
                for mode in ('presence','posture','skeleton','presence,posture','presence,skeleton','posture,skeleton','all','none'):
                    req=urllib.request.Request(base+'/api/camera/preview?mode='+mode,headers={'Authorization':'Bearer '+TOKEN})
                    with urllib.request.urlopen(req) as response:
                        selected=json.loads(response.headers['X-Frame-Metadata'])
                        self.assertEqual(selected['preview_mode'],mode)
                        self.assertEqual(selected['timestamp_ns'],str(timestamp))
                        self.assertEqual(selected['frame_id'],73)
                        self.assertTrue(response.read().startswith(b'\xff\xd8'))
                self.assertEqual(service.live_vision.snapshot()[1]['processed'],1)
                with self.assertRaises(urllib.error.HTTPError) as invalid:
                    request(base,TOKEN,'/api/camera/preview?mode=invalid')
                self.assertEqual(invalid.exception.code,400)
                result=request(base,TOKEN,'/api/camera/analysis')['result']
                self.assertEqual(result['timestamp_ns'],timestamp)
                self.assertEqual(result['frame_id'],frame['frame_id'])
                self.assertEqual(len(result['skeletons'][0]['joints']),17)
                self.assertEqual(store.list(),[])
            finally:
                service.close()
                server.shutdown()
                thread.join()
                server.server_close()
                store.close()
