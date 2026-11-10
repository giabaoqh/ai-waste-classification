"""Separate offline units, isolated HTTP integration, and opt-in genuine model tests."""
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
# These are standalone evaluation scripts (no pytest test functions). One executes
# remote image downloads/model loading at import. Keep the scripts, never import them in pytest.
collect_ignore = ['test_gatekeeper_benchmark.py', 'test_ood_evaluation.py']


def pytest_addoption(parser):
    parser.addoption('--run-model-tests', action='store_true', default=False,
                     help='Run inference checks only with genuine, already-local artifacts; no downloads.')


def pytest_collection_modifyitems(config, items):
    for item in items:
        module = Path(str(item.fspath)).name
        cls = item.cls.__name__ if item.cls else ''
        name = item.originalname or item.name
        if cls in {'TestApiIntegration', 'TestApiAnalyticsIntegration', 'TestApiHistoryIntegration'} or (
            module == 'test_image_quality.py' and name.startswith('test_api_')
        ):
            item.add_marker(pytest.mark.integration)
        legacy = cls in {'TestEndToEndKnowledgeBaseIntegration', 'TestMultiObjectDetectionHandler', 'TestNonWasteDetection'}
        legacy |= module == 'test_new_features.py'
        legacy |= module == 'test_multi_object_detection.py' and name not in {'test_coco_classes_audit', 'test_safe_crop_clamping'}
        legacy |= name in {'test_predict_with_real_image', 'test_api_predict_automatically_creates_history',
                           'test_api_real_valid_image_inference', 'test_api_warning_dark_image_proceeds_to_inference',
                           'test_api_warning_blurry_image_proceeds_to_inference'}
        if legacy:
            item.add_marker(pytest.mark.model)
            item.add_marker(pytest.mark.legacy_model)
        if item.get_closest_marker('model'):
            if not config.getoption('--run-model-tests'):
                item.add_marker(pytest.mark.skip(reason='Model inference is opt-in: --run-model-tests; no model restored/downloaded.'))
            elif legacy and not (ROOT/'checkpoints/mobilenetv3_best.pth').is_file():
                item.add_marker(pytest.mark.skip(reason='Legacy checkpoint absent; restoration is outside the current direction.'))


@pytest.fixture(scope='session')
def live_api(tmp_path_factory):
    """Do not hit port 8000/user history. The child owns only its temporary DB."""
    directory = tmp_path_factory.mktemp('contract_api')
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    env = os.environ.copy()
    env['DATABASE_PATH'] = str(directory/'history.db')
    log = (directory/'api.log').open('w', encoding='utf-8')
    process = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'backend.app.main:app',
                                '--host', '127.0.0.1', '--port', str(port)],
                               cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    try:
        deadline = time.monotonic()+25
        while time.monotonic()<deadline:
            if process.poll() is not None:
                pytest.fail('Isolated API exited: '+(directory/'api.log').read_text(encoding='utf-8'))
            try:
                with urllib.request.urlopen(url+'/api/health', timeout=1) as response:
                    if response.status==200:break
            except (OSError, urllib.error.URLError):
                time.sleep(.1)
        else:
            pytest.fail('Isolated API did not start: '+(directory/'api.log').read_text(encoding='utf-8'))
        yield url
    finally:
        process.terminate()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill();process.wait(timeout=5)
        log.close()


@pytest.fixture(autouse=True)
def test_boundaries(request, monkeypatch):
    if request.node.get_closest_marker('integration'):
        monkeypatch.setattr(request.module, 'BASE_URL', request.getfixturevalue('live_api'))
    else:
        def offline(*args, **kwargs):
            raise urllib.error.URLError('Network disabled in offline unit tests')
        monkeypatch.setattr(urllib.request, 'urlopen', offline)
        import torch.hub
        def no_download(*args, **kwargs):
            pytest.fail('Tests must not download weights or datasets; provide existing local artifacts')
        monkeypatch.setattr(torch.hub,'download_url_to_file',no_download)


@pytest.fixture(scope='module')
def legacy_models():
    from backend.tests.test_multi_object_detection import setup_models
    return setup_models()


@pytest.fixture
def classifier(legacy_models):return legacy_models[0]


@pytest.fixture
def transform(legacy_models):return legacy_models[1]


@pytest.fixture
def target_classes(legacy_models):return legacy_models[2]


@pytest.fixture
def device(legacy_models):return legacy_models[3]
