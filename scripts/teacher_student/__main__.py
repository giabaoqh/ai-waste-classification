import argparse
import json
import sys
from pathlib import Path

from .audit import audit_dataset
from .common import read_json, write_json
from .review import export_approved, import_teacher, prepare_pilot


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='Offline Teacher–Student QA. No training/download/API calls.')
    commands = parser.add_subparsers(dest='command', required=True)
    audit = commands.add_parser('audit')
    audit.add_argument('--dataset', type=Path, required=True)
    audit.add_argument('--repo', type=Path, default=Path.cwd())
    audit.add_argument('--output', type=Path, required=True)
    pilot = commands.add_parser('pilot')
    pilot.add_argument('--audit', type=Path, required=True)
    pilot.add_argument('--workspace', type=Path, required=True)
    pilot.add_argument('--count', type=int, default=150)
    pilot.add_argument('--seed', type=int, default=42)
    teacher = commands.add_parser('import-teacher')
    teacher.add_argument('--workspace', type=Path, required=True)
    teacher.add_argument('--bundle', type=Path, required=True)
    commands.add_parser('teacher-status')
    propose = commands.add_parser('propose-local')
    propose.add_argument('--workspace', type=Path, required=True)
    propose.add_argument('--artifact', type=Path, required=True)
    propose.add_argument('--output', type=Path, required=True)
    propose.add_argument('--limit', type=int, default=5)
    propose.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    export = commands.add_parser('export')
    export.add_argument('--workspace', type=Path, required=True)
    export.add_argument('--output', type=Path, required=True)
    serve = commands.add_parser('serve')
    serve.add_argument('--workspace', type=Path, required=True)
    serve.add_argument('--port', type=int, default=8013)
    args = parser.parse_args()
    try:
        if args.command == 'audit':
            result = audit_dataset(args.dataset, args.repo)
            output = args.output.resolve()
            if output.exists() or output.is_relative_to(args.dataset.resolve()):
                raise ValueError('Audit output phải mới và nằm ngoài source dataset')
            write_json(output, result)
            print(json.dumps({'output': str(output), 'fingerprint': result['dataset_fingerprint'], 'split_stats': result['split_stats']}))
        elif args.command == 'pilot':
            if not 100 <= args.count <= 200:
                raise ValueError('Pilot thực tế cần 100–200 ảnh')
            print(json.dumps(prepare_pilot(read_json(args.audit), args.workspace, args.count, args.seed)))
        elif args.command == 'import-teacher':
            print(json.dumps(import_teacher(args.workspace, read_json(args.bundle))))
        elif args.command == 'teacher-status':
            from .local_teacher import teacher_status
            print(json.dumps(teacher_status(), ensure_ascii=False))
        elif args.command == 'propose-local':
            from .local_teacher import propose_local
            print(json.dumps(propose_local(args.workspace, args.artifact, args.output, args.limit, args.device)))
        elif args.command == 'export':
            print(json.dumps(export_approved(args.workspace, args.output)))
        elif args.command == 'serve':
            if args.port in (8000, 5173) or not 1024 <= args.port <= 65535:
                raise ValueError('Dùng cổng riêng, không thay server ứng dụng')
            import uvicorn
            from .server import create_app
            uvicorn.run(create_app(args.workspace), host='127.0.0.1', port=args.port, access_log=True)
    except (ValueError, OSError, KeyError, ImportError, RuntimeError, TypeError) as exc:
        parser.exit(2, str(exc) + '\n')


if __name__ == '__main__':
    main()
