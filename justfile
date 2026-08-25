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

load-neo4j config="config.toml":
    just --justfile {{backend_justfile}} load-neo4j {{config}}

check:
    just --justfile {{backend_justfile}} check

help:
    just --justfile {{backend_justfile}} --list
