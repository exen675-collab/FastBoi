"""Thin adapter around the pinned, official FastH3 inference recipe."""
import importlib.util
import json
import os
import secrets
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SIZES = {'Poziomo · 1344 × 768': (1344, 768), 'Pionowo · 768 × 1344': (768, 1344),
         'Kwadrat · 768 × 768': (768, 768)}


def validate_request(prompt, size, seed):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('Wpisz opis wideo.')
    if len(prompt) > 8000:
        raise ValueError('Opis może zawierać maksymalnie 8000 znaków.')
    if size not in SIZES:
        raise ValueError('Nieobsługiwany rozmiar.')
    numeric = float(seed)
    if not numeric.is_integer() or not -1 <= numeric <= 2**32 - 1:
        raise ValueError('Seed musi być liczbą całkowitą od -1 do 4294967295.')
    seed = secrets.randbelow(2**32) if numeric == -1 else int(numeric)
    return prompt.strip(), SIZES[size], seed


class Backend:
    def __init__(self, num_gpus=1, profile='portable'):
        self.num_gpus = num_gpus
        self.profile = profile
        self.generator = None
        self.recipe = None
        self.args = None
        self.lock = threading.Lock()

    def load(self):
        source = Path(os.environ.get('FASTBOI_SOURCE', ROOT / '.runtime/FastVideo'))
        script = source / 'examples/inference/basic/basic_fasth3.py'
        spec = importlib.util.spec_from_file_location('fastboi_recipe', script)
        self.recipe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.recipe)
        model_path = (ROOT / '.runtime/model-path.txt').read_text().strip()
        argv = ['--model-path', model_path, '--prompt', 'init', '--num-gpus', str(self.num_gpus),
                '--steps', '5', '--no-warmup', '--repeats', '1', '--execution-backend', 'mp']
        if self.profile == 'portable':
            argv += ['--no-replicated-dit', '--vsa-kernel', 'triton', '--no-fa4',
                     '--no-inference-torch-compile', '--no-compile-vae', '--profile', 'strict']
        self.args = self.recipe.parse_args(argv)
        self.recipe.configure_environment(self.args)
        self.recipe.validate_profile_dependencies(self.args)
        self.generator = self.recipe.VideoGenerator.from_config(
            self.recipe.build_generator_config(self.args))

    def generate(self, prompt, size, seed):
        prompt, (width, height), seed = validate_request(prompt, size, seed)
        with self.lock:
            started = time.monotonic()
            if self.generator is None:
                self.load()
            folder = ROOT / 'outputs' / uuid.uuid4().hex
            folder.mkdir(parents=True)
            self.args.prompt, self.args.width, self.args.height = prompt, width, height
            requested = folder / 'video.mp4'
            try:
                result = self.generator.generate(self.recipe.build_request(self.args, requested, seed))
                actual = self.recipe._actual_output_path(result, requested).resolve()
                if not actual.is_file() or ROOT.joinpath('outputs').resolve() not in actual.parents:
                    raise RuntimeError('Silnik nie zapisał pliku MP4 w katalogu outputs.')
            except Exception:
                # A failed CUDA request can leave distributed workers unusable.
                self.close()
                raise
            metadata = {'prompt': prompt, 'width': width, 'height': height, 'seed': seed,
                        'num_frames': 124, 'fps': 24, 'transformer_forwards': 4,
                        'profile': self.profile, 'num_gpus': self.num_gpus,
                        'seconds': round(time.monotonic() - started, 2),
                        **json.loads((ROOT / 'config.json').read_text())}
            info = folder / 'parameters.json'
            info.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
            return str(actual), str(info), seed, metadata['seconds']

    def close(self):
        generator, self.generator = self.generator, None
        if generator is not None:
            generator.shutdown()
