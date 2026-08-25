"""Behavioural tests for the deterministic resolution policy."""

import unittest
from datetime import UTC, datetime
from inspect import signature
from ipaddress import IPv6Address
from uuid import UUID

from mayajaal.resolution import (
    ResolutionEntityType,
    ResolutionMethod,
    resolve_all,
)
from mayajaal.resolution.normalizers import normalize_email, normalize_phone
from mayajaal.schemas import (
    Account,
    AccountId,
    Address,
    Device,
    DeviceId,
    DevicePlatform,
    DeviceType,
    IPAddress,
    IPAddressId,
    PaymentIdentity,
    PaymentIdentityId,
    PaymentMethod,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def identifier(number: int) -> UUID:
    """Provide readable, fixed UUIDs so canonical selection is testable."""
    return UUID(int=number)


def address(number: int, **changes: str | None) -> Address:
    """Create a valid address with intentionally overridable raw formatting."""
    values: dict[str, str | UUID | None] = {
        "id": identifier(number),
        "recipient_name": "Asha Kumar",
        "line1": "Flat 4B, 12 M.G. Road",
        "line2": None,
        "city": "Bangalore",
        "region": "Karnataka",
        "postal_code": "560001",
        "country_code": "IN",
    }
    values.update(changes)
    return Address.model_validate(values)


class ResolutionTests(unittest.TestCase):
    def test_email_and_phone_libraries_normalize_common_formatting(self) -> None:
        self.assertEqual(normalize_email(" User@Example.Test "), "User@Example.Test")
        self.assertEqual(normalize_phone("+91 98765-43210"), "+919876543210")

    def test_email_local_part_case_is_not_collapsed(self) -> None:
        self.assertNotEqual(
            normalize_email("User@example.test"),
            normalize_email("user@example.test"),
        )

    def test_known_address_formatting_variations_resolve_together(self) -> None:
        results = resolve_all(
            addresses=(
                address(2),
                address(
                    1,
                    recipient_name="Different household member",
                    line1=" apartment 4b / 12 mg road ",
                    city="Bengaluru",
                    postal_code="560 001",
                ),
            )
        ).results
        self.assertEqual(
            {result.canonical_entity_id for result in results}, {identifier(1)}
        )
        self.assertIn(
            ResolutionMethod.NORMALIZED, {result.method for result in results}
        )

    def test_fuzzy_address_matching_is_bounded_by_locality_candidates(self) -> None:
        results = resolve_all(
            addresses=(
                address(1, line1="Flat 4B, 12 Mahatma Gandhi Road"),
                address(2, line1="Flat 4B, 12 Mahatma Gandi Rd"),
                address(
                    3,
                    line1="Flat 4B, 12 Mahatma Gandi Rd",
                    postal_code="110001",
                ),
            )
        ).results
        by_id = {result.raw_entity_id: result for result in results}
        self.assertEqual(by_id[identifier(2)].canonical_entity_id, identifier(1))
        self.assertEqual(by_id[identifier(2)].method, ResolutionMethod.FUZZY)
        self.assertEqual(by_id[identifier(3)].canonical_entity_id, identifier(3))

    def test_unrelated_addresses_do_not_merge(self) -> None:
        results = resolve_all(
            addresses=(
                address(1),
                address(
                    2, line1="99 Marine Drive", city="Mumbai", postal_code="400001"
                ),
            )
        ).results
        self.assertEqual(
            {result.canonical_entity_id for result in results},
            {identifier(1), identifier(2)},
        )

    def test_exact_identifier_resolution_is_deterministic(self) -> None:
        accounts = (
            Account(
                id=AccountId(identifier(2)),
                created_at=NOW,
                email=" User@Example.Test ",
                phone_e164="+919876543210",
            ),
            Account(
                id=AccountId(identifier(1)),
                created_at=NOW,
                email="user@example.test",
                phone_e164="+919876543210",
            ),
        )
        devices = (
            Device(
                id=DeviceId(identifier(4)),
                fingerprint=" DEVICE-ABC ",
                device_type=DeviceType.MOBILE,
                platform=DevicePlatform.ANDROID,
                first_seen_at=NOW,
                last_seen_at=NOW,
            ),
            Device(
                id=DeviceId(identifier(3)),
                fingerprint="device-abc",
                device_type=DeviceType.MOBILE,
                platform=DevicePlatform.ANDROID,
                first_seen_at=NOW,
                last_seen_at=NOW,
            ),
        )
        payments = (
            PaymentIdentity(
                id=PaymentIdentityId(identifier(6)),
                method=PaymentMethod.UPI,
                fingerprint=" PAYMENT-ABC ",
                first_seen_at=NOW,
                last_seen_at=NOW,
            ),
            PaymentIdentity(
                id=PaymentIdentityId(identifier(5)),
                method=PaymentMethod.UPI,
                fingerprint="payment-abc",
                first_seen_at=NOW,
                last_seen_at=NOW,
            ),
        )
        first = resolve_all(
            accounts=accounts, devices=devices, payment_identities=payments
        )
        second = resolve_all(
            accounts=tuple(reversed(accounts)),
            devices=tuple(reversed(devices)),
            payment_identities=tuple(reversed(payments)),
        )
        self.assertEqual(first, second)
        self.assertTrue(all(result.score == 100 for result in first.results))
        self.assertIn(
            ResolutionMethod.NORMALIZED, {result.method for result in first.results}
        )

    def test_shared_household_address_is_shared_without_a_fraud_interpretation(
        self,
    ) -> None:
        results = resolve_all(
            addresses=(
                address(1, recipient_name="Asha Kumar"),
                address(
                    2,
                    recipient_name="Ravi Kumar",
                    line1="Flat 4B 12 MG Road",
                    city="Bengaluru",
                ),
            )
        ).results
        self.assertEqual(
            {result.canonical_entity_id for result in results}, {identifier(1)}
        )
        self.assertTrue(all("fraud" not in result.evidence for result in results))

    def test_fraud_labels_cannot_influence_matching(self) -> None:
        accounts = (
            Account(
                id=AccountId(identifier(1)),
                created_at=NOW,
                email="same@example.test",
            ),
            Account(
                id=AccountId(identifier(2)),
                created_at=NOW,
                email="same@example.test",
            ),
        )
        result = resolve_all(accounts=accounts)
        self.assertEqual(len(result.results), 2)
        self.assertTrue(
            all(
                item.entity_type is ResolutionEntityType.EMAIL
                for item in result.results
            )
        )
        self.assertNotIn("synthetic_labels", signature(resolve_all).parameters)
        self.assertTrue(all("label" not in item.evidence for item in result.results))

    def test_ip_representations_normalize_to_the_same_identifier(self) -> None:
        ips = (
            IPAddress(
                id=IPAddressId(identifier(1)),
                address=IPv6Address("2001:db8::1"),
                first_seen_at=NOW,
                last_seen_at=NOW,
            ),
            IPAddress(
                id=IPAddressId(identifier(2)),
                address=IPv6Address("2001:0db8:0:0:0:0:0:1"),
                first_seen_at=NOW,
                last_seen_at=NOW,
            ),
        )
        results = resolve_all(ip_addresses=ips).results
        self.assertEqual(
            {result.canonical_entity_id for result in results}, {identifier(1)}
        )


if __name__ == "__main__":
    unittest.main()
