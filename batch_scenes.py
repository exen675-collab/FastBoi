#!/usr/bin/env python3
"""Run many FastBoi scenes from a JSON storyboard file.

See docs/batch-scenes.md for the format. Example:

  python batch_scenes.py examples/scenes.example.json --dry-run
  python batch_scenes.py scenes.json --mode local --num-gpus 1
  python batch_scenes.py scenes.json --mode gradio --endpoint http://127.0.0.1:7860
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent

try:
    from backend import SIZES, Backend, validate_request
except ImportError:  # pragma: no cover - only when used outside repo
    SIZES = {
        'Poziomo · 1344 × 768': (1344, 768),
        'Pionowo · 768 × 1344': (768, 1344),
        'Kwadrat · 768 × 768': (768, 768),
    }

    def validate_request(prompt, size, seed):  # type: ignore
        raise RuntimeError('backend.py not importable; use --mode gradio or run from repo root')

    Backend = None  # type: ignore

ID_RE = re.compile(r'^[A-Za-z0-9._-]{1,64}$')
ENGINE_NUM_FRAMES = 124
ENGINE_FPS = 24
ENGINE_DURATION = round(ENGINE_NUM_FRAMES / ENGINE_FPS, 2)


def _is_http_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
    except Exception:
        return False
    return parsed.scheme in ('http', 'https') and bool(parsed.netloc)


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('Root JSON value must be an object.')
    return data


def normalize_document(doc: dict[str, Any], json_dir: Path) -> dict[str, Any]:
    version = doc.get('version')
    if version != 1:
        raise ValueError('version must be 1.')
    defaults = doc.get('defaults') or {}
    if not isinstance(defaults, dict):
        raise ValueError('defaults must be an object.')
    scenes_in = doc.get('scenes')
    if not isinstance(scenes_in, list) or not scenes_in:
        raise ValueError('scenes must be a non-empty array.')

    default_size = defaults.get('size', next(iter(SIZES)))
    default_seed = defaults.get('seed', -1)
    default_duration = defaults.get('duration_seconds', ENGINE_DURATION)
    default_fps = defaults.get('fps', ENGINE_FPS)
    default_frames = defaults.get('num_frames', ENGINE_NUM_FRAMES)

    scenes: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for i, raw in enumerate(scenes_in, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f'scenes[{i - 1}] must be an object.')
        scene_id = raw.get('id') or f'scene_{i:03d}'
        if not isinstance(scene_id, str) or not ID_RE.match(scene_id):
            raise ValueError(f'Invalid scene id {scene_id!r} (use A-Za-z0-9._- , max 64).')
        if scene_id in seen_ids:
            raise ValueError(f'Duplicate scene id: {scene_id}')
        seen_ids.add(scene_id)

        prompt = raw.get('prompt')
        size = raw.get('size', default_size)
        seed = raw.get('seed', default_seed)
        # Reuse backend validation for prompt/size/seed when available.
        try:
            prompt, _wh, seed = validate_request(prompt, size, seed)
        except ValueError:
            raise
        except RuntimeError:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f'{scene_id}: prompt required') from None
            if size not in SIZES:
                raise ValueError(f'{scene_id}: unsupported size {size!r}') from None
            prompt = prompt.strip()
            seed = int(seed)

        refs_raw = raw.get('reference_images') or []
        if not isinstance(refs_raw, list):
            raise ValueError(f'{scene_id}: reference_images must be an array.')
        refs: list[dict[str, Any]] = []
        for j, ref in enumerate(refs_raw):
            if not isinstance(ref, dict):
                raise ValueError(f'{scene_id}: reference_images[{j}] must be an object.')
            path = ref.get('path')
            url = ref.get('url')
            if not path and not url:
                raise ValueError(f'{scene_id}: reference_images[{j}] needs path or url.')
            entry: dict[str, Any] = {
                'index': int(ref.get('index', j + 1)),
            }
            if path is not None:
                if not isinstance(path, str) or not path.strip():
                    raise ValueError(f'{scene_id}: empty path in reference_images[{j}].')
                p = Path(path)
                if not p.is_absolute():
                    p = (json_dir / p).resolve()
                entry['path'] = str(p)
                entry['path_exists'] = p.is_file()
            if url is not None:
                if not isinstance(url, str) or not _is_http_url(url):
                    raise ValueError(f'{scene_id}: invalid url in reference_images[{j}].')
                entry['url'] = url
            refs.append(entry)

        declared = raw.get('reference_image_count')
        if declared is not None:
            if not isinstance(declared, int) or declared < 0:
                raise ValueError(f'{scene_id}: reference_image_count must be a non-negative int.')
            if declared != len(refs):
                raise ValueError(
                    f'{scene_id}: reference_image_count={declared} but '
                    f'reference_images has {len(refs)} entries.'
                )

        scenes.append({
            'id': scene_id,
            'prompt': prompt,
            'size': size,
            'seed': seed,
            'duration_seconds': float(raw.get('duration_seconds', default_duration)),
            'num_frames': int(raw.get('num_frames', default_frames)),
            'fps': float(raw.get('fps', default_fps)),
            'reference_image_count': len(refs) if declared is None else declared,
            'reference_images': refs,
            'notes': raw.get('notes'),
            'engine_applies': {
                'prompt': True,
                'size': True,
                'seed': True,
                'duration_seconds': False,
                'num_frames': False,
                'fps': False,
                'reference_images': False,
                'engine_duration_seconds': ENGINE_DURATION,
                'engine_num_frames': ENGINE_NUM_FRAMES,
                'engine_fps': ENGINE_FPS,
            },
        })

    return {
        'version': 1,
        'defaults': {
            'size': default_size,
            'seed': default_seed,
            'duration_seconds': default_duration,
            'fps': default_fps,
            'num_frames': default_frames,
        },
        'scenes': scenes,
        'source_warnings': [
            'FastH3 path ignores duration_* / fps / num_frames (fixed 124@24).',
            'FastH3 path ignores reference_images (text-to-video only).',
            'Multi-GPU shards one job; this runner does not fan out scenes across GPUs.',
        ],
    }


def run_local(scene: dict[str, Any], backend: Any, out_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    video, meta_path, used_seed, seconds = backend.generate(
        scene['prompt'], scene['size'], scene['seed']
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    # Copy/move is unnecessary: Backend already writes under outputs/. Link result.
    result = {
        'id': scene['id'],
        'ok': True,
        'mode': 'local',
        'video': video,
        'parameters': meta_path,
        'seed': used_seed,
        'seconds': seconds,
        'wall_seconds': round(time.monotonic() - started, 2),
        'scene': scene,
    }
    (out_dir / 'scene_result.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    return result


def run_gradio(scene: dict[str, Any], endpoint: str, out_dir: Path,
               auth: tuple[str, str] | None) -> dict[str, Any]:
    try:
        from gradio_client import Client
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            'gradio_client is required for --mode gradio (pip install gradio_client)'
        ) from exc

    started = time.monotonic()
    client = Client(endpoint, auth=auth) if auth else Client(endpoint)
    # Gradio Blocks: generate(prompt, size, seed) -> video, files, status
    video, files, status = client.predict(
        scene['prompt'],
        scene['size'],
        scene['seed'],
        api_name='/predict',
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        'id': scene['id'],
        'ok': True,
        'mode': 'gradio',
        'endpoint': endpoint,
        'video': video,
        'files': files,
        'status': status,
        'wall_seconds': round(time.monotonic() - started, 2),
        'scene': scene,
    }
    (out_dir / 'scene_result.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Batch-generate FastBoi scenes from JSON.')
    parser.add_argument('json_path', type=Path, help='Path to scenes JSON')
    parser.add_argument('--mode', choices=['local', 'gradio'], default='local')
    parser.add_argument('--endpoint', default='http://127.0.0.1:7860',
                        help='Gradio server URL for --mode gradio')
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--profile', choices=['portable', 'blackwell'], default='portable')
    parser.add_argument('--low-vram', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Validate and print plan only')
    parser.add_argument('--concurrency', type=int, default=1,
                        help='Max parallel Gradio jobs (server still queues; default 1). '
                             'Ignored for --mode local (Backend is single-flight).')
    parser.add_argument('--run-id', default=None, help='Optional batch run id (default: random hex)')
    args = parser.parse_args(argv)

    json_path = args.json_path.resolve()
    if not json_path.is_file():
        print(f'File not found: {json_path}', file=sys.stderr)
        return 2

    try:
        normalized = normalize_document(load_json(json_path), json_path.parent)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f'Invalid scenes JSON: {exc}', file=sys.stderr)
        return 2

    run_id = args.run_id or uuid.uuid4().hex[:12]
    batch_dir = ROOT / 'outputs' / f'batch_{run_id}'
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / 'normalized_scenes.json').write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    print(f'Scenes: {len(normalized["scenes"])} · run_id={run_id}')
    for warning in normalized['source_warnings']:
        print(f'NOTE: {warning}')
    for scene in normalized['scenes']:
        missing = [r for r in scene['reference_images'] if r.get('path') and not r.get('path_exists')]
        if missing:
            print(f'WARN {scene["id"]}: missing local ref paths: '
                  f'{[m["path"] for m in missing]}')

    if args.dry_run:
        print('Dry-run OK. Wrote', batch_dir / 'normalized_scenes.json')
        return 0

    auth = None
    user = os.environ.get('FASTBOI_USER')
    password = os.environ.get('FASTBOI_PASSWORD')
    if user and password:
        auth = (user, password)

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    if args.mode == 'local':
        if Backend is None:
            print('Cannot import Backend; refuse --mode local.', file=sys.stderr)
            return 2
        backend = Backend(args.num_gpus, args.profile, low_vram=args.low_vram)
        try:
            for scene in normalized['scenes']:
                scene_dir = batch_dir / scene['id']
                print(f'→ {scene["id"]} (local)…')
                try:
                    results.append(run_local(scene, backend, scene_dir))
                    print(f'  OK {scene["id"]} in {results[-1]["wall_seconds"]}s')
                except Exception as exc:
                    err = {'id': scene['id'], 'ok': False, 'error': str(exc)}
                    errors.append(err)
                    scene_dir.mkdir(parents=True, exist_ok=True)
                    (scene_dir / 'scene_result.json').write_text(
                        json.dumps(err, ensure_ascii=False, indent=2), encoding='utf-8'
                    )
                    print(f'  FAIL {scene["id"]}: {exc}', file=sys.stderr)
        finally:
            backend.close()
    else:
        concurrency = max(1, args.concurrency)

        def job(scene: dict[str, Any]) -> dict[str, Any]:
            scene_dir = batch_dir / scene['id']
            print(f'→ {scene["id"]} (gradio)…')
            return run_gradio(scene, args.endpoint.rstrip('/'), scene_dir, auth)

        if concurrency == 1:
            for scene in normalized['scenes']:
                try:
                    results.append(job(scene))
                    print(f'  OK {scene["id"]} in {results[-1]["wall_seconds"]}s')
                except Exception as exc:
                    err = {'id': scene['id'], 'ok': False, 'error': str(exc)}
                    errors.append(err)
                    print(f'  FAIL {scene["id"]}: {exc}', file=sys.stderr)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {pool.submit(job, s): s for s in normalized['scenes']}
                for fut in as_completed(futures):
                    scene = futures[fut]
                    try:
                        results.append(fut.result())
                        print(f'  OK {scene["id"]} in {results[-1]["wall_seconds"]}s')
                    except Exception as exc:
                        err = {'id': scene['id'], 'ok': False, 'error': str(exc)}
                        errors.append(err)
                        print(f'  FAIL {scene["id"]}: {exc}', file=sys.stderr)

    manifest = {
        'run_id': run_id,
        'mode': args.mode,
        'source_json': str(json_path),
        'ok_count': len(results),
        'error_count': len(errors),
        'results': results,
        'errors': errors,
        'warnings': normalized['source_warnings'],
    }
    manifest_path = batch_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Done. manifest={manifest_path} ok={len(results)} fail={len(errors)}')
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
