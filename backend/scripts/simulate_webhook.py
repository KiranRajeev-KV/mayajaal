"""Deliver deterministic Razorpay-shaped fixtures to a local Mayajaal API."""

import argparse
import hashlib
import hmac
import json
import os
import sys
from collections.abc import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from mayajaal.api.env import load_environment
from mayajaal.api.webhooks import RAZORPAY_WEBHOOK_SECRET_ENVIRONMENT_VARIABLE

load_environment()

DEFAULT_ENDPOINT = "http://127.0.0.1:8000"


def _payload(
    *,
    event: str,
    created_at: int,
    payment_id: str,
    mayajaal: dict[str, object] | None = None,
) -> bytes:
    payload: dict[str, object] = {"payment": {"entity": "payment", "id": payment_id}}
    if mayajaal is not None:
        payload["mayajaal"] = mayajaal
    return json.dumps(
        {
            "entity": "event",
            "event": event,
            "contains": ["payment"],
            "payload": payload,
            "created_at": created_at,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _send(
    *, endpoint: str, secret: str, provider_event_id: str, body: bytes, valid: bool
) -> tuple[int, str]:
    signature = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    if not valid:
        signature = "0" * len(signature)
    request = Request(
        f"{endpoint.rstrip('/')}/webhooks/razorpay",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Razorpay-Signature": signature,
            "x-razorpay-event-id": provider_event_id,
        },
    )
    try:
        with urlopen(request, timeout=10) as response:
            return response.status, response.read().decode("utf-8")
    except HTTPError as error:
        return error.code, error.read().decode("utf-8")
    except URLError as error:
        raise RuntimeError(
            f"unable to reach local webhook endpoint: {error.reason}"
        ) from error


def _deliveries(mode: str) -> Iterable[tuple[str, bytes, bool]]:
    if mode == "normal":
        yield (
            "evt_mayajaal_demo_normal_001",
            _payload(
                event="payment.captured",
                created_at=1_780_000_000,
                payment_id="pay_demo_001",
            ),
            True,
        )
    elif mode == "duplicate":
        body = _payload(
            event="payment.captured",
            created_at=1_780_000_001,
            payment_id="pay_demo_002",
        )
        yield "evt_mayajaal_demo_duplicate_001", body, True
        yield "evt_mayajaal_demo_duplicate_001", body, True
    elif mode == "out-of-order":
        # The newer provider event is intentionally delivered first.
        yield (
            "evt_mayajaal_demo_order_new_001",
            _payload(
                event="payment.captured",
                created_at=1_780_000_010,
                payment_id="pay_demo_003",
            ),
            True,
        )
        yield (
            "evt_mayajaal_demo_order_old_001",
            _payload(
                event="payment.authorized",
                created_at=1_780_000_005,
                payment_id="pay_demo_003",
            ),
            True,
        )
    elif mode == "invalid-signature":
        yield (
            "evt_mayajaal_demo_invalid_001",
            _payload(
                event="payment.captured",
                created_at=1_780_000_020,
                payment_id="pay_demo_004",
            ),
            False,
        )
    elif mode == "graph-demo":
        # Namespaced fixture fields are Mayajaal demo metadata, not Razorpay claims.
        shared_payment = "00000000-0000-0000-0000-0000000000aa"
        yield (
            "evt_mayajaal_graph_device_001",
            _payload(
                event="mayajaal.device.seen",
                created_at=1_780_000_030,
                payment_id="pay_demo_graph_001",
                mayajaal={
                    "account_id": "00000000-0000-0000-0000-000000000001",
                    "device_id": "00000000-0000-0000-0000-0000000000d1",
                    "exposure_paise": 250_000,
                    "context_id": "graph-demo-001",
                },
            ),
            True,
        )
        yield (
            "evt_mayajaal_graph_payment_001",
            _payload(
                event="mayajaal.payment.attached",
                created_at=1_780_000_031,
                payment_id="pay_demo_graph_002",
                mayajaal={
                    "account_id": "00000000-0000-0000-0000-000000000001",
                    "payment_identity_id": shared_payment,
                    "exposure_paise": 250_000,
                    "context_id": "graph-demo-002",
                },
            ),
            True,
        )
        yield (
            "evt_mayajaal_graph_payment_002",
            _payload(
                event="mayajaal.payment.attached",
                created_at=1_780_000_029,
                payment_id="pay_demo_graph_003",
                mayajaal={
                    "account_id": "00000000-0000-0000-0000-000000000002",
                    "payment_identity_id": shared_payment,
                    "exposure_paise": 250_000,
                    "context_id": "graph-demo-003",
                },
            ),
            True,
        )
    else:
        raise ValueError(f"unsupported mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "normal",
            "duplicate",
            "out-of-order",
            "invalid-signature",
            "graph-demo",
        ),
        default="normal",
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    arguments = parser.parse_args()
    secret = os.environ.get(RAZORPAY_WEBHOOK_SECRET_ENVIRONMENT_VARIABLE)
    if not secret:
        raise ValueError(f"{RAZORPAY_WEBHOOK_SECRET_ENVIRONMENT_VARIABLE} must be set")

    for event_id, body, valid in _deliveries(arguments.mode):
        response_status, response_body = _send(
            endpoint=arguments.endpoint,
            secret=secret,
            provider_event_id=event_id,
            body=body,
            valid=valid,
        )
        print(f"{event_id}: HTTP {response_status} {response_body}")
        if valid and not 200 <= response_status < 300:
            return 1
        if not valid and response_status != 401:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
