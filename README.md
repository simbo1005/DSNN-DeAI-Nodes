# DSNN De-AI Nodes

Pinned custom-node stack for the De-AI image-processing workflow. The installer sets up the four upstream packages and applies the two supplied source-file replacements. Use it with an existing ComfyUI installation (workflow reference: ComfyUI 0.45.0).

| Workflow node | Package | Pinned commit |
| --- | --- | --- |
| `Vintage Effect` | [ComfyUI_VintageEffect](https://github.com/brucew4yn3rp/ComfyUI_VintageEffect) | `9f7c52c992a3ca7b9498799291fae2ccc53360d2` |
| `BondSaveWithCustomMetadata` | [Bond-Node-Suite](https://github.com/bondngn-studios/Bond-Node-Suite) | `8d2f6ce3abd471442bfd6785226154ec605aea0a` |
| `LCLensProfile` | [ComfyUI_LC123_nodes](https://github.com/lonecatone23/ComfyUI_LC123_nodes) | `04041dbeb32cb1f3b8c269882da1d419d1242c61` |
| `RadianceCompressionArtifacts` | [Radiance](https://github.com/FXTD-Studios/radiance) | `0174a5811b64665714a366b209cfd73d44418e91` |

The workflow's other six types are built into ComfyUI: `LoadImage`, `PreviewAny`, `PrimitiveInt`, `UpscaleModelLoader`, `ImageUpscaleWithModel`, and `ImageScaleBy`.

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

CI tests patch integrity, backups/idempotence, local-edit preservation, duplicate detection, path safety and Python version constraints. A Linux CPU smoke test installs the actual four packages into pinned ComfyUI, checks registration, and executes a synthetic image through Vintage Effect → LC Lens Profile → patched Radiance compression → patched Bond. The output must be a 4:4:4 JPEG. No diffusion or upscale weights are downloaded for that test.

See [third-party notices](THIRD_PARTY.md) for upstream ownership and licensing. DSNN's installer does not rename the upstream node IDs, so existing workflow links remain compatible.
