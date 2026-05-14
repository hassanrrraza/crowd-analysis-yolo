# Contributing to crowd-analysis-yolo

Thank you for your interest in this project. It accompanies research on BRT crowd analysis using YOLOv8; contributions that improve clarity, portability, and reproducibility are welcome.

## Before you open an issue

1. Read the [README](README.md) and, if relevant, [docs/detailed_setup_guide.md](docs/detailed_setup_guide.md), [docs/custom_training.md](docs/custom_training.md), and [docs/project_overview.md](docs/project_overview.md).
2. Confirm your problem is **this repository** (not Ultralytics, PyTorch, or OpenCV—those have their own trackers).
3. For **security-sensitive** reports, follow [SECURITY.md](SECURITY.md) instead of a public issue.

## Bug reports

Use [GitHub Issues](https://github.com/hassanrrraza/crowd-analysis-yolo/issues) and include as much of the following as you can:

- **OS** (e.g. Windows 10/11, Ubuntu 22.04) and **Python version** (`python --version`).
- **GPU / CUDA** if relevant (or “CPU only”).
- **Exact command** or code path (e.g. `python main.py --model ... --video ...`).
- **Full error message** and traceback.
- Whether you use a **venv** and how you installed dependencies (`pip install -r requirements.txt`).

Screenshots of the OpenCV window are helpful only when the bug is **visual**; otherwise text logs are enough.

## Feature requests and scope

This repo is aimed at **research and education** around crowd monitoring. Requests that require **large new products** (cloud dashboards, multi-tenant services, etc.) may be out of scope; opening an issue to **discuss** before a big PR is appreciated.

Respect the usage constraints in the README regarding **BRT footage** (educational/research use; do not misuse or misrepresent media).

## Pull requests

- Keep changes **focused**—one logical improvement per PR is easier to review.
- **Match** existing style (naming, imports, formatting) in the files you touch. Avoid drive-by refactors unrelated to your fix.
- Update **docs** when behavior or flags change (`README.md` or files under `docs/`).
- If you add scripts or optional features, mention how to run them briefly in the PR description.

Pull requests are reviewed **as time permits**; a short summary of *what* changed and *why* speeds things up.

## Questions and discussion

For “how do I …?” questions, check the docs above first. If something is missing, an issue suggesting a doc improvement is as valuable as code.

## Citation

If you extend this work in academic publications, please cite the IEEE Xplore publication linked in the [README](README.md).
