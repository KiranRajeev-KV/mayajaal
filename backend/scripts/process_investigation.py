"""Explicit recovery entry point for one durable Stage 12E investigation job."""

import argparse

from mayajaal.api.runtime import create_realtime_application_runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    runtime = create_realtime_application_runtime()
    try:
        result = runtime.investigations.process(args.run_id)
        print(f"{result.run_id} {result.status.value} reused={result.reused}")
    finally:
        runtime.dispose()


if __name__ == "__main__":
    main()
