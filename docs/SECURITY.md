# Security policy

This document describes how to report security concerns for the **crowd-analysis-yolo** repository ([MIT-licensed](README.md) research and educational codebase around BRT crowd analysis with YOLOv8).

## Scope

This project is a **local research and educational** tool: Python scripts, Jupyter notebooks, OpenCV windows, and Ultralytics YOLO inference. Typical use is **offline or user-controlled** execution on your own machine with your own video and weights.

**In scope for this policy:** vulnerabilities that could reasonably affect **users who run this repository’s code** (e.g. unsafe handling of untrusted inputs if introduced in the future, risky defaults in scripts, supply-chain suggestions that belong here rather than upstream libraries).

**Out of scope:** issues in **third-party** projects (Ultralytics, PyTorch, OpenCV, etc.). Report those to their respective maintainers. Generic security of BRT footage or physical infrastructure is outside this software policy.

## How to report

**Preferred:** Use [GitHub private vulnerability reporting](https://docs.github.com/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability) for this repository **if it is enabled** for the repo.

**Otherwise:** Email **Hassan Raza** at [hassan2056764@gmail.com](mailto:hassan2056764@gmail.com) with:

- A short description of the issue and why it matters in the context of this project.
- Steps to reproduce, or proof-of-concept, if applicable.
- Affected component (file, feature, or dependency version if relevant).

Please **do not** file public issues for **undisclosed** exploitable vulnerabilities; use the channels above so they can be addressed before wide disclosure.

## What to expect

- The maintainer will try to **acknowledge** serious reports within a few business days (no guaranteed SLA; this is a research-side project).
- You may be asked for clarification. Fixes may be shipped via normal commits or, where appropriate, a GitHub Security Advisory.

## Safe usage reminders

- Only run code and notebooks from **sources you trust**. Review `pip install` and notebook cells before executing on sensitive machines.
- **Do not** share BRT-related footage in ways that violate rights or the usage notes in the [README](README.md).

For general bugs that are not security-sensitive, use [GitHub Issues](https://github.com/hassanrrraza/crowd-analysis-yolo/issues). For how to contribute fixes, see [CONTRIBUTING.md](CONTRIBUTING.md).
