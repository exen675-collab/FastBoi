"""One-command, resumable Linux installation. No model imports before preflight."""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / '.runtime'
# FA2 dla gęstych warstw w profilu portable (fallback zamiast Torch SDPA).
# Pin zgodny z komentarzem w upstream pyproject.toml; budowane ze źródeł,
# więc wymaga kompilatora CUDA (nvcc) na hoście.
FLASH_ATTN_PIN = 'flash-attn==2.8.1'
# Blackwell (SM100+) nie ma kerneli FA2 — źródłowy build 2.8.1 nie może się
# udać na RTX PRO 6000 / RTX 50xx / B200 / GB200. Wtedy (i przy każdym
# mismatch nvcc vs torch, np. CUDA 12.8 vs cu130) kontynuujemy z Torch SDPA.
_BLACKWELL_HINTS = ('blackwell', 'b200', 'gb200', 'gb300', 'rtx 5090', 'rtx 5080',
                    'rtx 5070', 'rtx 5060', 'rtx pro 6000', 'rtx pro 5000',
                    'rtx pro 4000', 'dgx spark', 'gb10')


def _gpu_names():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ''
    return out


def _is_blackwell(names):
    lowered = names.lower()
    return any(hint in lowered for hint in _BLACKWELL_HINTS)


def run(*args, **kwargs):
    subprocess.run([str(a) for a in args], check=True, **kwargs)


def hardware_check(allow_low_vram=False):
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64'):
        raise RuntimeError('Ten instalator obsługuje Linux x86_64 / WSL2 z NVIDIA CUDA.')
    result = subprocess.check_output([
        'nvidia-smi', '--query-gpu=name,memory.total,driver_version',
        '--format=csv,noheader,nounits'], text=True)
    print(result, flush=True)
    rows = [row.split(',') for row in result.strip().splitlines()]
    if sum(int(row[1]) for row in rows) < 60000:
        message = ('Za mało VRAM dla standardowego FastH3 BF16. '
                   'Próg instalatora to 60 GB łącznie (nie gwarantuje działania). '
                   'Konfiguracja referencyjna autora: 4 × B200. '
                   'RTX 4060 8 GB nie jest obsługiwana przez ten profil.')
        if allow_low_vram:
            print('WARNING: bypassing 60GB VRAM gate at user request; full BF16 model will likely OOM.',
                  flush=True)
        else:
            raise RuntimeError(f'{message} Użyj --allow-low-vram, aby kontynuować na własne ryzyko.')
    return 'cu130' if min(int(row[2].strip().split('.')[0]) for row in rows) >= 580 else 'cu126'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--profile', choices=['portable', 'blackwell'], default='portable')
    parser.add_argument('--check', action='store_true', help='Tylko kontrola sprzętu')
    parser.add_argument('--allow-low-vram', action='store_true',
                        help='Pomiń próg 60 GB VRAM (kontynuacja na własne ryzyko, zwykle OOM)')
    parser.add_argument('--no-fa2', action='store_true',
                        help='Pomiń instalację FlashAttention-2 (gęste warstwy użyją Torch SDPA)')
    parser.add_argument('--low-vram', action='store_true',
                        help='Oszczędzaj VRAM w runtime (sekwencyjne ładowanie H3; wolniej, ale stabilniej na 1 GPU)')
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    if args.num_gpus < 1 or 56 % args.num_gpus:
        parser.error('Liczba GPU musi dzielić 56: 1, 2, 4, 7, 8, 14, 28, 56.')
    backend = hardware_check(allow_low_vram=args.allow_low_vram)
    if args.check:
        return
    if args.profile == 'blackwell' and backend != 'cu130':
        raise RuntimeError('Profil Blackwell wymaga sterownika obsługującego CUDA 13.')
    config = json.loads((ROOT / 'config.json').read_text())
    RUNTIME.mkdir(exist_ok=True)
    uv = shutil.which('uv')
    if not uv:
        installer = RUNTIME / 'uv-install.sh'
        run('curl', '-LsSf', 'https://astral.sh/uv/install.sh', '-o', installer)
        run('sh', installer, env={**os.environ, 'UV_INSTALL_DIR': str(RUNTIME / 'bin'),
                                 'UV_NO_MODIFY_PATH': '1'})
        uv = str(RUNTIME / 'bin' / 'uv')
    python = ROOT / '.venv/bin/python'
    if not python.exists():
        run(uv, 'venv', '--python', '3.12', '--seed', ROOT / '.venv')
    source = RUNTIME / 'FastVideo'
    revision = config['fastvideo_revision']
    if not (source / '.git').exists():
        run('git', 'clone', '--no-checkout', 'https://github.com/hao-ai-lab/FastVideo.git', source)
    run('git', '-C', source, 'checkout', '--detach', revision)
    env = {**os.environ, 'UV_TORCH_BACKEND': backend}
    stamp = RUNTIME / 'installed.json'
    want_fa2 = args.profile == 'portable' and not args.no_fa2
    signature = {'revision': revision, 'backend': backend, 'profile': args.profile,
                 'fa2': FLASH_ATTN_PIN if want_fa2 else None}
    if not stamp.exists() or json.loads(stamp.read_text()) != signature:
        package = f'{source}[fasth3]' if args.profile == 'blackwell' else str(source)
        run(uv, 'pip', 'install', '--python', python, '--no-sources-package',
            'fastvideo-kernel', '-e', package, env=env, cwd=source)
        if want_fa2:
            if _is_blackwell(_gpu_names()):
                print('Pomijam flash-attn 2.8.1: brak kerneli FA2 dla Blackwell '
                      '(SM100+); gęste warstwy użyją Torch SDPA.', flush=True)
            else:
                print(f'Instalowanie {FLASH_ATTN_PIN} (kompilacja ze źródeł, może potrwać).',
                      flush=True)
                try:
                    run(uv, 'pip', 'install', '--python', python, '--no-cache-dir',
                        FLASH_ATTN_PIN, '--no-build-isolation', env=env)
                except subprocess.CalledProcessError as exc:
                    # FA2 jest opcjonalną optymalizacją (upstream wymaga jej tylko
                    # w komentarzu; validate_profile_dependencies sprawdza FA4,
                    # nie FA2). Mismatch nvcc vs torch (np. 12.8 vs cu130) albo
                    # brak prebuilt wheel nie może blokować startu.
                    print(f'WARNING: instalacja {FLASH_ATTN_PIN} nie powiodła się ({exc}). '
                          'Kontynuuję z Torch SDPA dla gęstych warstw. '
                          'Aby nie ponawiać próby, usuń flagę albo użyj --no-fa2.',
                          flush=True)
        run(uv, 'pip', 'check', '--python', python)
        stamp.write_text(json.dumps(signature))
    env['HF_HOME'] = os.environ.get('HF_HOME', str(ROOT / '.cache/huggingface'))
    env['FASTBOI_SOURCE'] = str(source)
    # Validate actual CUDA visibility and the selected devices before downloading.
    prepare_cmd = [python, ROOT / 'prepare.py', '--num-gpus', args.num_gpus,
                   '--profile', args.profile]
    if args.allow_low_vram:
        prepare_cmd.append('--allow-low-vram')
    run(*prepare_cmd, env=env)
    # OOM w VAE decode to zwykle fragmentacja alokatora przy piku VRAM;
    # expandable_segments to zalecenie z samego komunikatu PyTorch.
    # setdefault: świadoma konfiguracja użytkownika ma pierwszeństwo.
    # Musi trafić do env przed execve, bo workery CUDA dziedziczą je przy starcie.
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    command = [str(python), str(ROOT / 'app.py'), '--host', args.host, '--port', str(args.port),
               '--num-gpus', str(args.num_gpus), '--profile', args.profile]
    if args.low_vram:
        command.append('--low-vram')
    if args.no_browser:
        command.append('--no-browser')
    os.execve(str(python), command, env)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError, FileNotFoundError) as exc:
        sys.exit(f'Błąd uruchomienia: {exc}')
