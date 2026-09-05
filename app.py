"""FastBoi browser UI. Start with start.sh after cloning."""
import argparse
import atexit
import os
import secrets
from pathlib import Path

from backend import SIZES, Backend


def build_ui(backend):
    import gradio as gr

    def generate(prompt, size, seed, progress=gr.Progress()):  # noqa: B008 - Gradio injects progress
        progress(0, desc='Ładowanie modelu / generowanie wideo i audio…')
        try:
            video, metadata, used_seed, seconds = backend.generate(prompt, size, seed)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        except Exception as exc:
            import logging
            logging.getLogger(__name__).exception('Generowanie nie powiodło się')
            raise gr.Error('Generowanie nie powiodło się. Sprawdź terminal (szczegóły błędu CUDA/modelu). '
                           'Przy braku pamięci użyj większego GPU lub większej liczby kart.') from exc
        return video, [video, metadata], f'Gotowe · {seconds:.1f} s · seed: {used_seed}'

    with gr.Blocks(title='FastBoi · FastH3', theme=gr.themes.Soft(primary_hue='violet'),
                   css='.gradio-container {max-width: 1150px !important;}') as demo:
        gr.Markdown('# FastBoi\n### Od pomysłu do wideo z dźwiękiem\nFastH3 Preview · 4 kroki · około 5 sekund · 24 kl./s')
        with gr.Row():
            with gr.Column(scale=5):
                prompt = gr.Textbox(label='Opisz scenę', lines=8,
                                    placeholder='A cinematic close-up of rain falling on a neon-lit Tokyo street. '
                                    'Soft rainfall and distant city sounds.')
                gr.Markdown('Uwzględnij ruch kamery, oświetlenie i dźwięki. Najlepiej zacząć od opisu po angielsku.')
                size = gr.Dropdown(list(SIZES), value=next(iter(SIZES)), label='Format')
                seed = gr.Number(value=-1, precision=0, label='Seed (−1 = losowy)')
                button = gr.Button('Generuj wideo', variant='primary')
            with gr.Column(scale=6):
                video = gr.Video(label='Twoje wideo', interactive=False)
                status = gr.Textbox(value='Gotowy. Pierwsza generacja ładuje model do pamięci.',
                                    label='Status', interactive=False)
                files = gr.File(label='Pobierz MP4 i parametry', file_count='multiple', interactive=False)
        gr.Examples([
            [('A red fox walks through a snowy pine forest at dawn. Slow tracking shot. '
              'Soft footsteps in snow, wind through the trees.')],
            [('Ocean waves break against dark cliffs at sunset. Aerial cinematic shot. '
              'Crashing surf and distant seabirds.')],
        ], inputs=prompt)
        gr.Markdown('Pliki zostają w `outputs/`. Zadania wykonują się kolejno. '
                    'Nie zamykaj serwera podczas generowania.')
        button.click(generate, [prompt, size, seed], [video, files, status], concurrency_limit=1)
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--profile', choices=['portable', 'blackwell'], default='portable')
    parser.add_argument('--low-vram', action='store_true',
                        help='Oszczędzaj VRAM na 1 GPU (sekwencyjne ładowanie H3: wolniej, ale stabilniej).')
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    backend = Backend(args.num_gpus, args.profile, low_vram=args.low_vram)
    atexit.register(backend.close)
    auth = None
    if args.host not in ('127.0.0.1', 'localhost', '::1'):
        user = os.environ.get('FASTBOI_USER', 'admin')
        password = os.environ.get('FASTBOI_PASSWORD') or secrets.token_urlsafe(20)
        auth = (user, password)
        print(f'Logowanie: {user}', flush=True)
        if not os.environ.get('FASTBOI_PASSWORD'):
            print(f'Hasło na tę sesję: {password}', flush=True)
    demo = build_ui(backend)
    demo.queue(max_size=8, default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, auth=auth, share=False,
        inbrowser=not args.no_browser,
        allowed_paths=[str(Path(__file__).resolve().parent / 'outputs')])


if __name__ == '__main__':
    main()
