# DSNN De-AI Nodes

Pinned custom-node stack for the De-AI image-processing workflow. The installer sets up the four upstream packages and applies the two supplied source-file replacements. Use it with an existing ComfyUI installation (workflow reference: ComfyUI 0.45.0).

| Workflow node | Package | Pinned commit |
| --- | --- | --- |
| `Vintage Effect` | [ComfyUI_VintageEffect](https://github.com/brucew4yn3rp/ComfyUI_VintageEffect) | `9f7c52c992a3ca7b9498799291fae2ccc53360d2` |
| `BondSaveWithCustomMetadata` | [Bond-Node-Suite](https://github.com/bondngn-studios/Bond-Node-Suite) | `8d2f6ce3abd471442bfd6785226154ec605aea0a` |
| `LCLensProfile` | [ComfyUI_LC123_nodes](https://github.com/lonecatone23/ComfyUI_LC123_nodes) | `04041dbeb32cb1f3b8c269882da1d419d1242c61` |
| `RadianceCompressionArtifacts` | [Radiance](https://github.com/FXTD-Studios/radiance) | `0174a5811b64665714a366b209cfd73d44418e91` |

The workflow's other six types are built into ComfyUI: `LoadImage`, `PreviewAny`, `PrimitiveInt`, `UpscaleModelLoader`, `ImageUpscaleWithModel`, and `ImageScaleBy`.

## GPU and performance runtime

Re-run the installer and restart ComfyUI to enable the included performance helper. It uses the existing node IDs and preserves your workflow file, effect parameters, node order, output dimensions, JPEG passes, quality and colour sampling.

- **LC Lens Profile:** batched float32 interpolation on ComfyUI's CUDA device when memory permits; bounded caching of lens grids. CPU execution is supported, with the original CPU method used after a CUDA out-of-memory error. It honours ComfyUI's CPU mode and does not install another Torch or attention wheel.
- **LC before/after previews:** the generated thumbnails have a maximum edge of 1024 pixels. The `IMAGE` output remains full resolution. This reduces PNG encoding, image transfer and browser overhead. Inspect the final Bond output for full-resolution details, or set the preview edge to 0 for original full-size previews.
- **Bond metadata:** identical EXIF reads within one save are reused, with invalidation after writes. Successful location lookups are cached for up to 24 hours (128 cities maximum); failed lookups are retried. JPEG quality 95, 4:4:4 and encoder optimization remain as supplied.
- **Vintage:** a redundant final image copy is avoided for single-image batches. Its Pillow JPEG codec, passes and image effects remain unchanged.
- **Timings:** console lines beginning `[DSNN De-AI]` report the four custom nodes, model-based upscaling and resizing when executed. Cached ComfyUI nodes do not produce a new timing.

JPEG encoding and EXIF writing remain on the CPU. The runtime does not change the 4× upscaler, move the resize, or lower quality to produce a speedup. CUDA interpolation uses the same float32 equations; CPU/GPU rounding can differ, so cross-device results are not guaranteed to be bit-identical. No real-CUDA speed claim is made by the CPU CI checks.

After an image runs, append `/dsnn/deai/performance` to your ComfyUI address to see observed devices, individual durations and a summary sorted by mean duration. On Vast's internal port this is commonly:

```bash
curl -fsS http://127.0.0.1:18188/dsnn/deai/performance | python -m json.tool
```

Look for `LCLensProfile` with `device=cuda:0` in the log. CPU fallback includes its reason. The helper checks each custom-node source path and hash before optimizing; unfamiliar versions retain their original behavior and appear under `warnings` in the report.

Optional template environment variables (restart ComfyUI after changing them):

| Variable | Default | Purpose |
| --- | --- | --- |
| `DSNN_DEAI_PERFORMANCE` | `1` | Set to `0` for the baseline behavior, including the two supplied source overrides. |
| `DSNN_DEAI_DEVICE` | `auto` | `auto`/`cuda` prefer ComfyUI's CUDA device with safe CPU fallback; `cpu` requests CPU math. |
| `DSNN_DEAI_PREVIEW_EDGE` | `1024` | Thumbnail maximum edge; `0` restores full-resolution previews. |
| `DSNN_DEAI_GRID_CACHE_MB` | `512` | Maximum lens-map cache size; `0` disables caching. |

Alternatively, install with `--no-performance` and restart to retain the baseline node behavior. The helper lives at `custom_nodes/DSNN-DeAI-Performance`; it adds no new workflow node types. Your supplied Bond and camera files stay byte-identical on disk.

## Install

Run these in your instance terminal, replacing the ComfyUI path if needed:

```bash
git clone https://github.com/simbo1005/DSNN-DeAI-Nodes.git
python DSNN-DeAI-Nodes/install.py --comfyui /workspace/ComfyUI
```

The installer uses `COMFYUI_VENV` when set, then checks ComfyUI's usual virtual-environment folders, `/venv/main`, and the current Python. Set the interpreter explicitly if your setup uses another environment:

```bash
python DSNN-DeAI-Nodes/install.py --comfyui /workspace/ComfyUI --python /venv/main/bin/python
```

For a RunPod setup using the grabber's own virtual environment, that argument can instead be `/workspace/runpod-slim/ComfyUI/.venv-cu128/bin/python`, with `--comfyui /workspace/runpod-slim/ComfyUI`. Use your instance's actual paths.

Restart ComfyUI once installation finishes, then refresh its browser page. The installer leaves a running generation alone; restart after your queue finishes.

For ComfyUI-Manager installations, `install.py` also auto-detects ComfyUI when this repository is under `ComfyUI/custom_nodes/DSNN-DeAI-Nodes`. The four packages are installed as separate sibling directories so their imports, assets and frontend extensions retain their original structure. This stack folder itself adds no extra node types.

## Included modifications

- `overrides/Bond-Node-Suite/bond_nodes.py` replaces Bond's root `bond_nodes.py`. `BondSaveWithCustomMetadata` saves `.jpg` files using JPEG quality 95, 4:4:4 colour sampling (`subsampling=0`) and optimization, then stamps metadata with ExifTool.
- `overrides/radiance/film/camera.py` replaces **`radiance/film/camera.py`**. The compression node accepts `block_size=1` with step 1. A value of 1 disables the extra mosaic/block-averaging pass while retaining JPEG compression. The node's default remains 8, matching the supplied file.

Both files are byte-identical to the supplied modifications. Their expected hashes and original upstream hashes are recorded in `nodes.lock.json`. The private workflow and reference images remain in their original private repository.

## Dependencies and existing installations

Python dependencies come from the pinned packages' requirements files. Installed Torch, torchvision, torchaudio, torchcodec, Triton, SageAttention and NumPy versions are constrained during pip resolution. A dependency conflict stops installation for review instead of replacing those existing versions. ComfyUI must already have working PyTorch in the selected Python.

Bond needs **ExifTool**. On Debian/Ubuntu, the correct package is **`libimage-exiftool-perl`**. The installer installs it with `apt-get` when run as root. Otherwise install it with your system package manager and ensure `exiftool` is on PATH before retrying.

Existing checkouts are reused by their Git origin, even when their folder name differs. A different commit, unknown local edit, duplicate checkout, or symlinked patch target stops installation. The installer preserves those files. Use a fresh installation or manage the existing checkout deliberately before retrying; it never runs `git reset --hard`.

Original replacement targets are backed up under `<ComfyUI>/.dsnn-deai/backups/`. The installation receipt is `<ComfyUI>/.dsnn-deai/installed.json`. An interrupted install may leave `install.lock` there; remove that lock only after confirming its recorded process has stopped. Re-running a completed installation is safe and retains the backups.

Advanced `--skip-deps` and `--skip-system-tools` flags are available for environments whose dependencies are managed separately; the receipt records skipped checks.

## Verify after restarting

Use the local address and port on which your ComfyUI listens:

```bash
python DSNN-DeAI-Nodes/install.py --verify http://127.0.0.1:8188
```

On Vast, the internal port may be `18188` instead of `8188`. Verification checks all ten workflow node IDs and confirms the patched Radiance `block_size=1` schema. It reports missing types rather than treating a successful clone as a successful node load.

This repository installs node code. Continue using your own De-AI workflow and its selected upscale-model file.

## Validation and provenance

CI tests patch integrity, backups/idempotence, local-edit preservation, duplicate detection, path safety and Python version constraints. The lens math is compared against the independently captured pinned LC123 method for all eight lenses and both aberration modes, including batch, dtype and alpha checks. A CUDA equivalence test is included and runs when a CUDA device is available.

A Linux CPU smoke test installs the actual four packages into pinned ComfyUI and executes the synthetic image chain twice, with performance mode disabled and enabled. It checks node registration, 4:4:4 JPEG output, final decoded-pixel agreement, helper activation and metadata-read reuse. No diffusion or upscale weights are downloaded for that test.

See [third-party notices](THIRD_PARTY.md) for upstream ownership and licensing. DSNN's installer does not rename the upstream node IDs, so existing workflow links remain compatible.
