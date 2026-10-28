"""Camera-only API and storage boundary regression checks."""
import io
import json
import tempfile
import unittest
import urllib.error
import zipfile

from backend.client import Client
from backend.export import export_dataset
from backend.service import Service
from backend.storage import Store
from framegrabber.protocol import request
import testSystem as fixtures


class CameraScopeTests(unittest.TestCase):
    def test_fresh_database_and_service_only_expose_camera_operations(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            service = Service(store, stop_timeout=0)
            try:
                tables = {row[0] for row in store.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                self.assertEqual(tables, {'experiments', 'frames', 'artifacts', 'audit'})
                for target, names in ((store, ('csi', 'add_csi')), (service, ('emit_csi', 'adapter', 'features', 'boundary')), (Client, ('csi', 'features'))):
                    for name in names:
                        self.assertFalse(hasattr(target, name), name)
                item = service.start(False, 'Camera session')
                service.stop(item['id'])
                archive = io.BytesIO()
                export_dataset(store, item['id'], archive)
                with zipfile.ZipFile(archive) as dataset:
                    names = {name.split('/')[-1] for name in dataset.namelist()}
                    self.assertIn('timing.json', names)
                    self.assertFalse(names & {'csi.jsonl', 'csi.csv', 'features.jsonl', 'synchronization.json'})
            finally:
                service.close()
                store.close()

    def test_radio_and_external_model_routes_are_not_available(self):
        system = fixtures.NetworkTests()
        system.setUp()
        try:
            item = system.service.start(False, 'Camera session')
            for prefix in ('recordings', 'experiments'):
                for route in ('csi', 'csiBatch', 'boundary', 'features', 'synchronization'):
                    with self.subTest(prefix=prefix, route=route):
                        for method, data in (('POST', {}), ('GET', None)):
                            with self.assertRaises(urllib.error.HTTPError) as error:
                                request(system.base, fixtures.TOKEN, f'/api/{prefix}/{item["id"]}/{route}', method, data)
                            self.assertEqual(error.exception.code, 404)
            detail = request(system.base, fixtures.TOKEN, '/api/recordings/' + item['id'])
            self.assertNotIn('csi_samples', detail)
            self.assertNotIn('adapter', detail['metadata'])
            self.assertNotIn('adapter', request(system.base, fixtures.TOKEN, '/api/health'))
        finally:
            system.tearDown()
