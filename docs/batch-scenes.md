# Batch scenes JSON (FastBoi)

Format for driving many FastBoi video generations from one file.
Another agent can author this JSON from a plot/storyboard.

## Current engine limits (read this)

FastBoi / FastH3 Preview today:

| Field in JSON | Applied now? | Notes |
| --- | --- | --- |
| `prompt` | **yes** | Required. English works best. |
| `size` | **yes** | One of the UI format labels (see below). |
| `seed` | **yes** | `-1` = random. |
| `duration_seconds` / `num_frames` / `fps` | **stored only** | Engine is fixed at **124 frames @ 24 FPS (~5.17 s)**. Values are kept in the job manifest for future models / agents. |
| `reference_images` | **stored only** | UI/backend do **not** accept image conditioning yet. Paths/URLs are validated when present and copied into the per-scene result metadata. |
| multi-GPU parallel scenes | **no** | `--num-gpus` shards **one** generation. Parallel scenes would need separate FastBoi processes (one GPU each); this runner stays sequential (or concurrent Gradio queue against one server, still one GPU job at a time). |

## File shape

```json
{
  "version": 1,
  "defaults": {
    "size": "Poziomo · 1344 × 768",
    "seed": -1,
    "duration_seconds": 5.17,
    "fps": 24,
    "num_frames": 124
  },
  "scenes": [
    {
      "id": "s01_open",
      "prompt": "A cinematic…",
      "size": "Poziomo · 1344 × 768",
      "seed": 42,
      "duration_seconds": 5.17,
      "reference_image_count": 2,
      "reference_images": [
        { "index": 1, "path": "refs/hero.png" },
        { "index": 2, "url": "https://example.com/ref2.png" }
      ],
      "notes": "optional free text for humans/agents"
    }
  ]
}
```

- `version` (number, required): must be `1`.
- `defaults` (object, optional): fallbacks for scene fields.
- `scenes` (array, required): 1…N scenes, order = run order. No hard upper limit.

### Scene object

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `id` | string | recommended | Stable id (`^[A-Za-z0-9._-]{1,64}$`). Auto `scene_NNN` if omitted. |
| `prompt` | string | **yes** | Scene description (max 8000 chars). |
| `size` | string | no | Format label; default from `defaults` or first `SIZES` key. |
| `seed` | integer | no | `-1`…`2^32-1`. Default `-1`. |
| `duration_seconds` | number | no | Desired length in seconds (forward-compat). |
| `num_frames` | integer | no | Desired frame count (forward-compat). |
| `fps` | number | no | Desired fps (forward-compat). |
| `reference_image_count` | integer | no | Declared count; if set, must equal `reference_images.length`. |
| `reference_images` | array | no | List of refs (see below). |
| `notes` | string | no | Ignored by runner; for authors. |

### Reference image object

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `index` | integer | recommended | 1-based slot number. |
| `path` | string | one of path/url | Local filesystem path relative to the JSON file or absolute. |
| `url` | string | one of path/url | `http(s)` URL. |

At least one of `path` or `url` is required per entry.

### Allowed `size` values

Must match Gradio dropdown labels exactly:

- `Poziomo · 1344 × 768`
- `Pionowo · 768 × 1344`
- `Kwadrat · 768 × 768`

## CLI

On the machine where FastBoi is installed (model ready):

```bash
# Dry-run: validate only
python batch_scenes.py scenes.json --dry-run

# Local: import Backend in-process (same as UI)
python batch_scenes.py scenes.json --mode local --num-gpus 1

# Against a running Gradio server (e.g. Vast.ai tunnel on :7860)
python batch_scenes.py scenes.json --mode gradio --endpoint http://127.0.0.1:7860
```

Outputs:

- Each scene writes under `outputs/batch_<run_id>/<scene_id>/` (MP4 + `parameters.json` + `scene_result.json`).
- A run summary is written to `outputs/batch_<run_id>/manifest.json`.

Auth for Gradio (when server uses `FASTBOI_USER` / password):

```bash
export FASTBOI_USER=admin
export FASTBOI_PASSWORD='…'
python batch_scenes.py scenes.json --mode gradio --endpoint http://127.0.0.1:7860
```

## Authoring tips for other agents

1. One scene ≈ one continuous camera beat (~5 s with current model).
2. Put motion, lighting, and sound in the English `prompt`.
3. Keep `id`s stable so regenerating one scene does not renumber the whole film.
4. Always fill `reference_images` when the story needs character consistency — even if FastBoi ignores them today, downstream tools / future backends can consume the same JSON.
5. Prefer `defaults` for shared size/seed policy; override per scene only when needed.
