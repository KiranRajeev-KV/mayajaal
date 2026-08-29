# Root entrypoints delegate to backend/justfile, retaining identical recipe names.

backend_justfile := justfile_directory() + "/backend/justfile"

default: check

fix:
    just --justfile {{backend_justfile}} fix

format:
    just --justfile {{backend_justfile}} format

format-check:
    just --justfile {{backend_justfile}} format-check

lint:
    just --justfile {{backend_justfile}} lint

typecheck:
    just --justfile {{backend_justfile}} typecheck

deps:
    just --justfile {{backend_justfile}} deps

dead-code:
    just --justfile {{backend_justfile}} dead-code

test:
    just --justfile {{backend_justfile}} test

generate config="config.toml":
    just --justfile {{backend_justfile}} generate {{config}}

# Docker Compose lifecycle commands. Pass `service=<name>` to target one service
# where Compose supports it, for example: `just docker-logs service=neo4j`.
docker-up:
    docker compose up -d

docker-down:
    docker compose down

docker-start:
    docker compose start

docker-stop:
    docker compose stop

docker-logs service="":
    docker compose logs --tail=100 {{service}}

neo4j-up:
    docker compose up -d neo4j

# Start or stop only the local operational PostgreSQL service.
db-up:
    docker compose up -d postgres

db-down:
    docker compose stop postgres

db-migrate:
    just --justfile {{backend_justfile}} db-migrate

db-current:
    just --justfile {{backend_justfile}} db-current

db-ping:
    just --justfile {{backend_justfile}} db-ping

api-run:
    just --justfile {{backend_justfile}} api-run

neo4j-load config="config.toml":
    just --justfile {{backend_justfile}} neo4j-load {{config}}

# Destructively clear the derived Neo4j graph before switching datasets.
neo4j-reset:
    just --justfile {{backend_justfile}} neo4j-reset

features-extract config="config.toml":
    just --justfile {{backend_justfile}} features-extract {{config}}

baseline-train config="config.toml":
    just --justfile {{backend_justfile}} baseline-train {{config}}

# Forward evaluation flags, for example: `just held-out-evaluate --full`.
held-out-evaluate *args:
    just --justfile {{backend_justfile}} held-out-evaluate {{args}}

# Forward calibration flags, for example: `just calibration-evaluate --full`.
calibration-evaluate *args:
    just --justfile {{backend_justfile}} calibration-evaluate {{args}}

# Forward one verified frozen-evaluation sample and decision context to the policy CLI.
policy-decide *args:
    just --justfile {{backend_justfile}} policy-decide {{args}}

# Run the frozen 18 live investigation-model comparison runs sequentially.
investigation-model-comparison *args:
    just --justfile {{backend_justfile}} investigation-model-comparison {{args}}

# Forward arbitrary validation flags after the recipe name, for example:
# `just synthetic-validate --full --output-dir artifacts/validation-10k`.
synthetic-validate *args:
    just --justfile {{backend_justfile}} synthetic-validate {{args}}

# Use a non-default profile while retaining arbitrary validation flags.
synthetic-validate-config config *args:
    just --justfile {{backend_justfile}} synthetic-validate-config {{config}} {{args}}

check:
    just --justfile {{backend_justfile}} check

help:
    just --justfile {{backend_justfile}} --list
