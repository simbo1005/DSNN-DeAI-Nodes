# Third-party notices

The original installer, manifest, documentation and tests in this repository are MIT-licensed. This does not relicense upstream node packages or the supplied modified files.

| Component | Upstream | License / distribution |
| --- | --- | --- |
| Vintage Effect | [brucew4yn3rp/ComfyUI_VintageEffect](https://github.com/brucew4yn3rp/ComfyUI_VintageEffect/tree/9f7c52c992a3ca7b9498799291fae2ccc53360d2) | No license file declared at this pin. Source is fetched directly from upstream; it is not redistributed in this repository. |
| Bond-Node-Suite | [bondngn-studios/Bond-Node-Suite](https://github.com/bondngn-studios/Bond-Node-Suite/tree/8d2f6ce3abd471442bfd6785226154ec605aea0a) | MIT, Copyright © 2026 Bond. Modified `bond_nodes.py` supplied by DSNN; upstream notice is [included](overrides/Bond-Node-Suite/LICENSE.txt). |
| LC123 Nodes | [lonecatone23/ComfyUI_LC123_nodes](https://github.com/lonecatone23/ComfyUI_LC123_nodes/tree/04041dbeb32cb1f3b8c269882da1d419d1242c61) | MIT. Full package fetched directly from upstream. The lens math in `runtime/performance.py` and the independent test reference are adapted/copied from the pinned source under its MIT license; the [notice is included](LICENSES/LC123-MIT.txt). |
| Radiance | [FXTD-Studios/radiance](https://github.com/FXTD-Studios/radiance/tree/0174a5811b64665714a366b209cfd73d44418e91) | GPL-3.0-only. Modified `film/camera.py` supplied by DSNN; the [GPL license is included](overrides/radiance/LICENSE). Original package and corresponding source are available at the pinned upstream link. |

Modifications distributed here: Bond JPEG output at quality 95 with 4:4:4 sampling; Radiance compression schema permitting a one-pixel block size. Upstream names and notices are retained. Full packages are cloned from their original public repositories at the manifest's exact commits.
