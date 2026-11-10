"""Loopback-only review UI. Isolated from the product API/database."""
from pathlib import Path
import secrets

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .common import read_json
from .review import config, get_item, item_path, save_review, verify_item

WEB = Path(__file__).parent / 'web'


def create_app(workspace):
    workspace = Path(workspace).resolve()
    cfg = config(workspace)
    token = secrets.token_urlsafe(32)
    app = FastAPI(title='Local Teacher–Student Review', docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost'])

    @app.middleware('http')
    async def protect(request: Request, call_next):
        if request.method == 'POST':
            if request.headers.get('X-Review-Token') != token:
                return JSONResponse({'detail': 'Thiếu token review cùng phiên'}, status_code=403)
            origin = request.headers.get('origin')
            if origin and origin != str(request.base_url).rstrip('/'):
                return JSONResponse({'detail': 'Không nhận thay đổi từ origin khác'}, status_code=403)
            try:
                too_large = int(request.headers.get('content-length', '0')) > 2 * 1024 * 1024
            except ValueError:
                too_large = True
            if too_large:
                return JSONResponse({'detail': 'Request review quá lớn'}, status_code=413)
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; frame-ancestors 'none'; object-src 'none'"
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    @app.get('/')
    def index():
        return FileResponse(WEB / 'index.html')

    @app.get('/assets/{name}')
    def asset(name: str):
        if name not in ('review.js', 'review.css', 'geometry.mjs'):
            raise HTTPException(404)
        return FileResponse(WEB / name, media_type='text/javascript' if name.endswith(('.js', '.mjs')) else 'text/css')

    @app.get('/api/session')
    def session():
        return {'csrf_token': token, 'taxonomy': cfg['taxonomy'], 'test_only': cfg['test_only'],
                'dataset_fingerprint': cfg['dataset_fingerprint'], 'teacher_execution': 'import_only_no_inference',
                'workspace': str(workspace)}

    @app.get('/api/items')
    def items():
        records = [read_json(item_path(workspace, key)) for key in cfg['image_ids']]
        return [{'image_id': r['image_id'], 'source': r['source']['source'], 'file': r['source']['image_file'],
                 'status': r['status'], 'revision': r['revision'], 'flags': r['source']['flags'],
                 'teacher_available': r['teacher'] is not None} for r in records]

    @app.get('/api/items/{image_id}')
    def item(image_id: str):
        try:
            return get_item(workspace, image_id)
        except (ValueError, OSError, KeyError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get('/api/image/{image_id}')
    def image(image_id: str):
        try:
            record = get_item(workspace, image_id)
            _, path = verify_item(workspace, record)
        except (ValueError, OSError, KeyError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return FileResponse(path, media_type='image/jpeg')

    @app.post('/api/items/{image_id}/review')
    def review(image_id: str, payload: dict):
        try:
            return save_review(workspace, image_id, payload)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise HTTPException(409 if 'Revision' in str(exc) else 400, str(exc)) from exc

    return app
