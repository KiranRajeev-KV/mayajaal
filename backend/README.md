## Schemas

Canonical storage-independent Pydantic schemas live in `mayajaal.schemas`.

- `ids.py` contains nominal UUID ID types for graph entities.
- `entities.py` contains production-shaped account, identity, commerce, promotion,
  and refund models.
- `events.py` contains the canonical event envelope and its optional synthetic-only
  ground-truth labels.

Run the schema tests from this directory:

```bash
uv run python -m unittest discover -s tests -v
```

## Development commands

`just` is the shared local quality interface. Run the same recipes from either
the repository root or `backend/`:

```bash
just format       # apply Ruff formatting
just lint         # run Ruff lint checks
just typecheck    # run BasedPyright in strict mode
just deps         # check declared dependencies with deptry
just dead-code    # run Vulture at 100% confidence
just test         # run unit tests
just check        # run every non-mutating quality gate
```

`faker`, `numpy`, and `polars` are intentionally retained for the planned
synthetic-data generator. CUDA-enabled `torch`, `torchaudio`, and `torchvision`
are intentionally retained for planned ML work. These are the only targeted
deptry unused-dependency exceptions. Production tooling lives in the `dev`
dependency group.

## Stage 1: synthetic fraud world

`mayajaal.synthetic` generates deterministic, validated temporal histories for
normal customers, legitimate shared households, and promo, refund, and mixed
abuse rings. Fraud ground truth appears only on `Event.synthetic_labels`.

```python
from pathlib import Path

from mayajaal.synthetic import GenerationProfile, export_parquet, generate_world

world = generate_world(GenerationProfile(seed=20260824))
paths = export_parquet(world, Path("artifacts/synthetic-world"))
```

`world` contains typed Pydantic entity/event tuples; `to_tables(world)` returns
the corresponding separate Polars tables. The master seed drives a local NumPy
generator and a local seeded Faker instance, with no global random-state use.

## Stage 2: deterministic resolution

`mayajaal.resolution.resolve_all` resolves entity identifiers deterministically.
Email normalization uses `email-validator`'s `normalized` value without
lowercasing the local part. Phone normalization uses libphonenumber's
`is_possible_number` length-oriented check before formatting to E.164; this is
intentional so formatting resolution can retain plausible historical values
whose regional prefix is not currently valid.

For manual generation, edit the non-secret [config.toml](config.toml) profile
and run either command below:

```bash
just generate
just generate another-profile.toml
```

The script writes Parquet tables to the profile's `[output].directory`; relative
paths are resolved relative to the config file. You can also run
`uv run python -m scripts.generate_dataset --help` for an output-directory
override.
