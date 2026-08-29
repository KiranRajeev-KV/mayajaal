# AGENTS.md

- Keep documentation up to date as you work through the repo: when changing code in `backend/`, update the relevant `README.md` (root or `backend/`) to reflect new modules, CLI commands, or behavior.
- For established numerical or ML functionality, use the supported API of the relevant maintained library (for example scikit-learn for calibration and classification metrics) rather than reimplementing it to satisfy static typing. Declare it as a direct dependency when imported; contain any incomplete third-party typing at a small adapter boundary with narrowly scoped type suppressions and tests against the library's behavior.

## Testing guidance for this hackathon

- Add only high-value tests that protect important contracts, regressions, failure boundaries, or provenance/security invariants.
- Do not add a test for every implementation detail or trivial branch.
- Prefer extending an existing focused test over creating repetitive cases.
- Keep the suite fast and maintainable; this is a hackathon project, not exhaustive production certification.
