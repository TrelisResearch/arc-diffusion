# ARC Diffusion
**A diffusion approach to solving ARC.**

Quickstart:
To get started on Runpod, you can avail of this one-click [template (affiliate link)](https://console.runpod.io/deploy?template=4yg1dndu1e&ref=jmfkcdio).

Running scripts:
Install with
```bash
uv sync
```
and see how to run scripts by running:
```bash
uv run script.py -h
```
e.g.
```bash
uv run pipeline.py -h
```
You can run on mac or cpu or gpu (cuda).

*Some sample scripts:*
To run training and evaluation (and pushing to hf):
```bash
uv run pipeline.py --config outputs/test_config.json
```
To run evaluation, with majority voting:
```bash
uv run evaluate.py --config outputs/test/config.json --maj --stats --prefer-best
```