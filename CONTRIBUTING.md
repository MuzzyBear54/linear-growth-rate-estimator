# Contributing

Small, focused contributions are welcome.

## Suggested contribution types
- Make the interface more reproducible (replace hardcoded parameters with CLI flags)
- Add batch mode for parameter scans
- Improve robustness checks or diagnostics
- Add unit tests for the rolling-fit / plateau-selection logic

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Style

- Keep it readable and physics-friendly.
- Prefer explicit variable names over cleverness.
- When adding heuristics (thresholds/gates), document the rationale in comments.
