"""Recover bounded Stage 12B/12C webhook-to-risk pipeline work."""

import argparse

from mayajaal.api.env import load_environment
from mayajaal.api.runtime import create_realtime_application_runtime


def main() -> int:
    load_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--event-id")
    group.add_argument("--limit", type=int)
    arguments = parser.parse_args()
    if arguments.limit is not None and not 1 <= arguments.limit <= 100:
        parser.error("--limit must be between 1 and 100")
    runtime = create_realtime_application_runtime()
    try:
        results = (
            (runtime.pipeline.process(arguments.event_id),)
            if arguments.event_id is not None
            else runtime.pipeline.process_next(limit=arguments.limit)
        )
        for result in results:
            print(
                f"{result.provider_event_id}: {result.processing_status.value} "
                f"type={result.canonical_event_type} state={result.state.value} "
                f"decision={result.decision_id} case={result.case_id}"
            )
    finally:
        runtime.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
