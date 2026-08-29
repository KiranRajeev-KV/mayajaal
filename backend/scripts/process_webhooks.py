"""Process bounded durable webhook inbox rows into the derived Neo4j graph."""

import argparse

from mayajaal.api.db import DatabaseConfig, create_database_runtime
from mayajaal.api.env import load_environment
from mayajaal.api.event_processing import (
    Neo4jRuntimeConfig,
    WebhookEventProcessor,
)
from mayajaal.graph import Neo4jGraphRepository

load_environment()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--event-id")
    group.add_argument("--limit", type=int)
    arguments = parser.parse_args()
    if arguments.limit is not None and not 1 <= arguments.limit <= 100:
        parser.error("--limit must be between 1 and 100")
    database = create_database_runtime(DatabaseConfig.from_environment())
    graph_config = Neo4jRuntimeConfig.from_environment()
    graph = Neo4jGraphRepository(
        graph_config.uri, (graph_config.username, graph_config.password)
    )
    try:
        processor = WebhookEventProcessor(database.sessions, graph)
        results = (
            (processor.process(arguments.event_id),)
            if arguments.event_id is not None
            else processor.process_next(limit=arguments.limit)
        )
        for result in results:
            print(
                f"{result.provider_event_id}: {result.status.value} "
                f"event={result.canonical_event_id} type={result.canonical_event_type} "
                f"nodes={result.graph_nodes_written} relationships={result.graph_relationships_written}"
            )
    finally:
        graph.close()
        database.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
