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

check:
    just --justfile {{backend_justfile}} check

help:
    just --justfile {{backend_justfile}} --list
