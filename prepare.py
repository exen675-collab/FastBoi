"""Check selected CUDA devices, then cache the complete pinned HF snapshot."""
import argparse
import json
import os
import shutil
from pathlib import Path


def main():
    import torch
    from huggingface_hub import HfApi, snapshot_download, try_to_load_from_cache
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-gpus', type=int, required=True)
    parser.add_argument('--profile', required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.num_gpus:
        raise RuntimeError('Brak wymaganej liczby GPU widocznych dla PyTorch/CUDA_VISIBLE_DEVICES.')
    total = 0
    for i in range(args.num_gpus):
        props = torch.cuda.get_device_properties(i)
        total += props.total_memory
        if props.major < 8:
            raise RuntimeError('Ten profil BF16 wymaga NVIDIA Ampere lub nowszej.')
        if args.profile == 'blackwell' and (props.major, props.minor) != (10, 0):
            raise RuntimeError('Profil blackwell/sm100a jest przeznaczony dla B200/GB200 (SM100).')
    if total < 60_000 * 1024**2:
        raise RuntimeError('Wybrane GPU mają mniej niż 60 GB VRAM łącznie. Zwiększ --num-gpus.')
    config = json.loads(Path('config.json').read_text())
    cache = Path(os.environ['HF_HOME']) / 'hub'
    cache.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(config['model_id'], revision=config['model_revision'], files_metadata=True)
    missing_bytes = sum(file.size or 0 for file in info.siblings
                        if not isinstance(try_to_load_from_cache(config['model_id'], file.rfilename,
                                          revision=config['model_revision']), str))
    if shutil.disk_usage(cache).free < missing_bytes + 10 * 1024**3:
        raise RuntimeError(f'Za mało miejsca w {cache}. Potrzeba około '
                           f'{missing_bytes / 1024**3 + 10:.0f} GiB na brakujące pliki i zapas.')
    print('Pobieranie modelu i wszystkich komponentów. Kolejne starty korzystają z cache.', flush=True)
    path = snapshot_download(config['model_id'], revision=config['model_revision'])
    Path('.runtime/model-path.txt').write_text(path)


if __name__ == '__main__':
    main()
