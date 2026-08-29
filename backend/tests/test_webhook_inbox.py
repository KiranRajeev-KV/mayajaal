"""High-value contracts for the Stage 12A durable webhook inbox."""

import asyncio
import hashlib
import hmac
import json
import unittest
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, cast

from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRoute
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from starlette.requests import Request

from mayajaal.api.app import create_app
from mayajaal.api.db import (
    Base,
    WebhookEventRepository,
    WebhookPayloadConflict,
    session_scope,
)
from mayajaal.api.webhooks import (
    RazorpayWebhookEnvelope,
    WebhookConfig,
    WebhookInboxService,
    WebhookIngestResult,
    verify_razorpay_signature,
)

SECRET = "test-webhook-secret"
WebhookEndpoint = Callable[[Request, Session], Any]


class WebhookInboxTests(unittest.TestCase):
    """Protect raw-body verification and atomic provider-ID durability."""

    def test_raw_body_signature_verification(self) -> None:
        body = _body("payment.captured", 1_780_000_000)
        signature = hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
        self.assertTrue(
            verify_razorpay_signature(raw_body=body, signature=signature, secret=SECRET)
        )
        self.assertFalse(
            verify_razorpay_signature(
                raw_body=body + b" ", signature=signature, secret=SECRET
            )
        )

    def test_duplicate_is_idempotent_but_conflicting_payload_is_rejected(self) -> None:
        engine, sessions = _database()
        body = _body("payment.captured", 1_780_000_000)
        envelope = RazorpayWebhookEnvelope.model_validate_json(body)
        try:
            with session_scope(sessions) as session:
                first = _accept(session, "evt_same", envelope, body)
                second = _accept(session, "evt_same", envelope, body)
                self.assertTrue(first.accepted_new)
                self.assertFalse(second.accepted_new)
                self.assertEqual(
                    len(WebhookEventRepository(session).list_recent(limit=10)), 1
                )

            changed = _body("payment.failed", 1_780_000_001)
            with (
                session_scope(sessions) as session,
                self.assertRaises(WebhookPayloadConflict),
            ):
                _accept(
                    session,
                    "evt_same",
                    RazorpayWebhookEnvelope.model_validate_json(changed),
                    changed,
                )
        finally:
            engine.dispose()

    def test_endpoint_rejects_invalid_signature_and_accepts_duplicate_without_rows(
        self,
    ) -> None:
        engine, sessions = _database()
        app = create_app()
        app.state.database_runtime = type("Runtime", (), {"sessions": sessions})()
        app.state.webhook_config = WebhookConfig.model_validate(
            {"razorpay_webhook_secret": SECRET}
        )
        endpoint = _route(app, "/webhooks/razorpay")
        body = _body("payment.captured", 1_780_000_000)
        try:
            with session_scope(sessions) as session:
                with self.assertRaises(HTTPException) as failure:
                    asyncio.run(
                        _call_endpoint(
                            endpoint, app, session, body, "evt_invalid", "bad"
                        )
                    )
                self.assertEqual(failure.exception.status_code, 401)
                self.assertEqual(
                    len(WebhookEventRepository(session).list_recent(limit=10)), 0
                )

            signature = hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
            with session_scope(sessions) as session:
                first = asyncio.run(
                    _call_endpoint(
                        endpoint, app, session, body, "evt_duplicate", signature
                    )
                )
                second = asyncio.run(
                    _call_endpoint(
                        endpoint, app, session, body, "evt_duplicate", signature
                    )
                )
                self.assertTrue(first.accepted_new)
                self.assertFalse(second.accepted_new)
                self.assertEqual(
                    len(WebhookEventRepository(session).list_recent(limit=10)), 1
                )
        finally:
            engine.dispose()

    def test_out_of_order_provider_timestamps_are_retained(self) -> None:
        engine, sessions = _database()
        newer = _body("payment.captured", 1_780_000_010)
        older = _body("payment.authorized", 1_780_000_005)
        try:
            with session_scope(sessions) as session:
                _accept(
                    session,
                    "evt_new",
                    RazorpayWebhookEnvelope.model_validate_json(newer),
                    newer,
                )
                _accept(
                    session,
                    "evt_old",
                    RazorpayWebhookEnvelope.model_validate_json(older),
                    older,
                )
                old = WebhookEventRepository(session).get("evt_old")
                new = WebhookEventRepository(session).get("evt_new")
                assert old is not None and new is not None
                self.assertLess(old.provider_created_at, new.provider_created_at)
        finally:
            engine.dispose()


def _database() -> tuple[Engine, sessionmaker[Session]]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return engine, sessionmaker(bind=engine, expire_on_commit=False)


def _body(event: str, created_at: int) -> bytes:
    return json.dumps(
        {
            "entity": "event",
            "event": event,
            "contains": ["payment"],
            "payload": {"payment": {"id": "pay_test"}},
            "created_at": created_at,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _accept(
    session: Session, event_id: str, envelope: RazorpayWebhookEnvelope, body: bytes
) -> WebhookIngestResult:
    return WebhookInboxService(session).accept(
        provider_event_id=event_id,
        envelope=envelope,
        raw_body=body,
        received_at=datetime(2026, 8, 29, tzinfo=UTC),
    )


def _route(app: FastAPI, path: str) -> WebhookEndpoint:
    route = next(
        route
        for route in cast(list[object], app.routes)
        if isinstance(route, APIRoute) and route.path == path
    )
    assert isinstance(route, APIRoute)
    return cast(WebhookEndpoint, route.endpoint)


async def _call_endpoint(
    endpoint: WebhookEndpoint,
    app: FastAPI,
    session: Session,
    body: bytes,
    event_id: str,
    signature: str,
) -> WebhookIngestResult:
    consumed = False

    async def receive():
        nonlocal consumed
        if consumed:
            return {"type": "http.request", "body": b"", "more_body": False}
        consumed = True
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/webhooks/razorpay",
            "headers": [
                (b"x-razorpay-signature", signature.encode()),
                (b"x-razorpay-event-id", event_id.encode()),
            ],
            "app": app,
        },
        receive,
    )
    return cast(WebhookIngestResult, await endpoint(request, session))
