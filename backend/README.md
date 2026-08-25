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

## Synthetic fraud world

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

## Deterministic resolution

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

## Temporal heterogeneous identity graph

`mayajaal.graph.build_graph_projection(world, resolution)` converts the
generated world and resolution results into a storage-independent graph payload. The
`Neo4jGraphRepository` loads that payload with canonical-ID uniqueness
constraints and event-keyed relationship merges, so reloading an unchanged
world is idempotent. Synthetic fraud labels are intentionally never projected.

Every graph relationship comes from an immutable event and stores only its
`event_id` and `event_time`. To obtain the graph as known at timestamp `T`, use
`relationships_known_at(T)`; it filters `event_time <= T` and therefore does
not expose later activity through aggregate edge properties.

For a local database with the Graph Data Science plugin enabled:

```bash
just docker-up              # start every Compose service
# or: just neo4j-up         # start Neo4j only
just load-neo4j
```

The root Justfile also exposes `docker-down`, `docker-start`, `docker-stop`,
and `docker-logs` (optionally `service=neo4j`) for normal Compose lifecycle
management.

Neo4j is available at `http://localhost:7474` and Bolt at `bolt://localhost:7687`.
The local Compose credentials are `neo4j` / `mayajaal`; override the loader with
`MAYAJAAL_NEO4J_URI`, `MAYAJAAL_NEO4J_USERNAME`, and
`MAYAJAAL_NEO4J_PASSWORD` when needed.
