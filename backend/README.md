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

`faker`, `numpy`, and `polars` are intentionally retained for the synthetic
world and feature-export paths. CUDA-enabled `torch`, `torchaudio`, and
`torchvision` are retained for future representation-learning work. These are
the only targeted deptry unused-dependency exceptions. CatBoost and SHAP are
direct baseline dependencies. Production tooling lives in the `dev` dependency
group.

## Synthetic fraud world

`mayajaal.synthetic` generates deterministic, validated temporal histories from
hidden customer personas and contexts. Ordinary shoppers have different order
cadences, payment preferences, promotion/refund propensities, device mixes, and
identity lifecycle changes. Legitimate households share home addresses and IPs,
while office/campus-like contexts share a network without necessarily sharing a
device, address, or payment identity. Fraud ground truth appears only on the
abuse-relevant `Event.synthetic_labels`; warm-up and ordinary campaign activity
remain unlabelled.

```python
from pathlib import Path

from mayajaal.synthetic import GenerationProfile, export_parquet, generate_world

world = generate_world(GenerationProfile(seed=20260824))
paths = export_parquet(world, Path("artifacts/synthetic-world"))
```

`world` contains typed Pydantic entity/event tuples; `to_tables(world)` returns
the corresponding separate Polars tables. The master seed drives local,
scope-derived NumPy generators and a local seeded Faker instance, with no global
random-state use. The profile retains the original population/ring count fields
and adds nested `population`, `identity_lifecycle`, `commerce`, `calendar`,
`abuse`, `prevalence`, `difficulty_profiles`, `diagnostics`, and `validation`
settings. The checked-in TOML makes every active behavioural, benchmark, and
diagnostic setting explicit and Pydantic validates it at load time. The calendar
creates ordinary seasonal concentration; the `drift` difficulty preset moves
activity later in the configured window.

`difficulty` and `prevalence` are intentionally orthogonal. Difficulty (`easy`,
`standard`, `hard`, or `drift`) selects a fully configured bundle controlling
persona concentration, benign sharing, identity lifecycle churn, campaign
sharing, warm-up, burstiness, and seasonality. Prevalence selects the target
labelled-account rarity independently. `development` may use a target such as
3%; `rare_abuse` defaults to 0.75% when no explicit target is supplied. These
are Mayajaal benchmark configurations, not claims about a merchant's fraud
rate. The Python profile API may set `target_labelled_account_rate=None` for
backwards-compatible count-driven scenarios; TOML uses the explicit target shown
in [config.toml](config.toml).

When a target prevalence is selected, campaigns are independently seeded draws
from the configured small/medium `prevalence.ring_sizes` distribution rather
than a few oversized rings. `strategy_weights` are sampling weights for each
campaign and `timeline_weights` allocate plans across early, middle, and late
chronological windows; `minimum_campaigns_per_timeline_bucket` protects class
support at every configured cutoff. Campaigns retain partial identity sharing,
warm-up orders, and either narrow burst windows or low-and-slow activity. They
are hidden generation plans: only their abuse-relevant event labels become
synthetic evaluation truth.

Benign households and office/campus contexts scale from the ordinary population
through `population.households_per_thousand_ordinary_accounts` and
`population.benign_network_groups_per_thousand_ordinary_accounts`. The legacy
`shared_household_count` and `population.benign_network_group_count` fields are
explicit overrides—set either to a number (including `0`) for compact fixtures
or omit them for population-scaled contexts.

Each `generate_dataset` run also writes `diagnostics.json`. It is an internal
plausibility report—not a claim of calibration to a private merchant dataset—
covering entity/order distributions, temporal gaps and burstiness, typed identity
reuse degrees, peer-set Jaccard overlap, typed multi-identity pairs, K2,2
bipartite four-cycles (including cross-type identity pairs), and graph topology.
It reports both the full account projection, where isolated accounts count as
zero-degree components, and the identity-sharing subgraph, which intentionally
excludes them. The configured component guardrails check both population-scale
component size and the fraction of labelled accounts concentrated in one
component. No SDMetrics dependency is required because no real reference data
is available.

`just synthetic-validate` writes a multi-seed report with early, middle, and
late cutoff-aware feature-health snapshots. It inspects variance, zero rates,
categorical dominance, near-redundant numeric pairs, class histogram overlap,
and one-feature separation for the existing feature schema. Only expected-active
numeric features must vary at the late cutoff; the configured velocity features
are documented as intentionally sparse/context-specific. Synthetic labels are
used only in explicitly evaluation-only overlap/separation fields. Run the full
benchmark pipeline, including CatBoost and SHAP artifacts, with:

```bash
just synthetic-validate --full
```

SHAP's top-feature share produces a review warning only. It is never an input to
generation or a target for tuning the generator. The full validation trains on
all configured accounts but uses the deterministic configured
`validation.shap_sample_count` sample for the offline SHAP report and PNG, so a
10k run remains practical. Each cutoff also reports a review warning if it has
fewer than the configured positive or negative samples; null class metrics are
never silently treated as evidence.

## Deterministic resolution

`mayajaal.resolution.resolve_all` resolves entity identifiers deterministically.
Email normalization uses `email-validator`'s `normalized` value without
lowercasing the local part. Phone normalization uses libphonenumber's
`is_possible_number` length-oriented check before formatting to E.164; this is
intentional so formatting resolution can retain plausible historical values
whose regional prefix is not currently valid.

Address fuzzy matching remains locality-bounded and now chooses the smallest
deterministic candidate group indexed by discriminative street/unit anchors,
capped at 64 candidates before RapidFuzz scoring. It therefore avoids a broad
city/postcode bucket turning into a global all-pairs comparison at benchmark
scale.

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
`event_id`, `event_type`, and `event_time`. Mutable lifecycle values such as an
account or order status and a refund completion state are deliberately excluded
from graph nodes. To obtain the graph as known at timestamp `T`, use
`relationships_known_at(T)`; it filters `event_time <= T` and therefore does
not expose later activity through aggregate edge properties.

For a local database with the Graph Data Science plugin enabled:

```bash
just docker-up              # start every Compose service
# or: just neo4j-up         # start Neo4j only
just neo4j-load
```

The root Justfile also exposes `docker-down`, `docker-start`, `docker-stop`,
and `docker-logs` (optionally `service=neo4j`) for normal Compose lifecycle
management. Before loading a changed seed, profile, or resolution policy, run
`just neo4j-reset` to destructively clear the dedicated derived database and
avoid combining experimental datasets.

Neo4j is available at `http://localhost:7474` and Bolt at `bolt://localhost:7687`.
The local Compose credentials are `neo4j` / `mayajaal`; override the loader with
`MAYAJAAL_NEO4J_URI`, `MAYAJAAL_NEO4J_USERNAME`, and
`MAYAJAAL_NEO4J_PASSWORD` when needed.

## Leakage-safe graph features

`mayajaal.features` is a model-independent feature layer over a
storage-independent `GraphProjection`. `FeatureService.extract(account_id, T)`
first forms a temporal snapshot containing only immutable event facts with
`event_time <= T`; every aggregate is then calculated from that snapshot. The
service does not accept the synthetic world or labels, so labels cannot enter
feature computation. `extract_many(..., T)` builds that cutoff snapshot and its
graph indexes once for the batch, then reuses them for each account vector.

The stable feature schema is composed from small extractors:

- account age: newly created accounts can be riskier when combined with other
  signals, while the value itself is known from the creation fact;
- identity reuse and connected-component size: unusually dense device, IP,
  payment, and shipping-address sharing can reveal coordination, but legitimate
  households remain ordinary shared identities rather than labels;
- commerce history: order value, promotion reuse, and refund request/resolution
  counts describe observed transaction behaviour only;
- 24-hour identity/account velocity: bursts of new accounts or identity links
  can expose rapid coordination;
- latest observed device, payment, promotion, and shipping-country context:
  categorical values are selected only from facts present by the cutoff.

Future relationship facts, including a refund resolution after a request, are
excluded before any family is calculated. Nodes are consulted only through
event-backed links that exist in the cutoff snapshot, except an account's own
creation time. This prevents future orders, identity links, promotions, refunds,
and lifecycle completion from leaking into a vector.

```bash
just features-extract                       # writes account_features.parquet
just features-extract another-profile.toml
```

The command defaults to the profile's `synthetic_world.end_at`; its Python
entry point accepts a timezone-aware `--cutoff` for an earlier reconstruction
and `--output` for the Parquet path.

## CatBoost baseline and SHAP explanations

`mayajaal.baseline` consumes `FeatureVector` values but has no feature logic.
It trains a single-threaded, seeded CatBoost classifier with `bootstrap_type=No`
and `auto_class_weights="Balanced"`. Synthetic labels are used only after
extraction to make offline targets. The training output includes the `.cbm`
model, ordered feature metadata, and a SHAP mean-absolute-contribution bar plot.
TreeSHAP contributions and their base value explain CatBoost's pre-sigmoid
`RawFormulaVal`, not probability; the reusable explanation returns that raw
score alongside the separately computed fraud probability. Their sum is tested
against CatBoost's raw prediction for additivity.

```bash
just baseline-train                         # writes CatBoost + metadata + SHAP plot
just baseline-train another-profile.toml
```

The command defaults to the profile's `synthetic_world.end_at`; its Python
entry point accepts a timezone-aware `--cutoff` for an earlier reconstruction
and `--output-dir` for the artifact directory.
