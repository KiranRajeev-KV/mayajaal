"""Unit tests for Mayajaal's canonical schema contracts."""

import unittest
from datetime import UTC, datetime, timedelta
from ipaddress import IPv4Address
from uuid import uuid4

from pydantic import ValidationError

from mayajaal.schemas import (
    AbuseType,
    Account,
    AccountId,
    AccountStatus,
    Address,
    AddressId,
    Device,
    DeviceId,
    DevicePlatform,
    DeviceType,
    Event,
    EventId,
    EventType,
    IPAddress,
    IPAddressId,
    Order,
    OrderId,
    PaymentIdentity,
    PaymentIdentityId,
    PaymentMethod,
    Promotion,
    PromotionDiscountType,
    PromotionId,
    Refund,
    RefundId,
    RefundState,
    SyntheticEventLabels,
)

NOW = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)


class SchemaConstructionTests(unittest.TestCase):
    def test_valid_entities_and_event_construct(self) -> None:
        account_id = AccountId(uuid4())
        address_id = AddressId(uuid4())
        order_id = OrderId(uuid4())
        promotion_id = PromotionId(uuid4())

        account = Account(id=account_id, created_at=NOW, email="person@example.test")
        device = Device(
            id=DeviceId(uuid4()),
            fingerprint="device-fingerprint-123",
            device_type=DeviceType.MOBILE,
            platform=DevicePlatform.ANDROID,
            first_seen_at=NOW,
            last_seen_at=NOW,
        )
        ip_address = IPAddress(
            id=IPAddressId(uuid4()),
            address=IPv4Address("203.0.113.7"),
            first_seen_at=NOW,
            last_seen_at=NOW,
        )
        payment = PaymentIdentity(
            id=PaymentIdentityId(uuid4()),
            method=PaymentMethod.UPI,
            fingerprint="payment-fingerprint-123",
            first_seen_at=NOW,
            last_seen_at=NOW,
        )
        address = Address(
            id=address_id,
            recipient_name="Asha Rao",
            line1="12 Market Road",
            city="Bengaluru",
            postal_code="560001",
            country_code="IN",
        )
        promotion = Promotion(
            id=promotion_id,
            code="WELCOME20",
            campaign_name="Welcome",
            discount_type=PromotionDiscountType.PERCENTAGE_BPS,
            discount_value=2_000,
            valid_from=NOW,
            valid_until=NOW + timedelta(days=30),
        )
        order = Order(
            id=order_id,
            account_id=account_id,
            shipping_address_id=address_id,
            placed_at=NOW,
            subtotal_paise=10_000,
            discount_paise=2_000,
            total_paise=8_000,
            item_count=1,
            promotion_id=promotion_id,
        )
        refund = Refund(
            id=RefundId(uuid4()),
            order_id=order_id,
            amount_paise=8_000,
            requested_at=NOW,
            state=RefundState.REQUESTED,
            reason_code="item_not_received",
        )
        event = Event(
            id=EventId(uuid4()),
            event_type=EventType.PROMOTION_REDEEMED,
            occurred_at=NOW,
            ingested_at=NOW,
            account_id=account_id,
            device_id=device.id,
            ip_address_id=ip_address.id,
            payment_identity_id=payment.id,
            order_id=order.id,
            address_id=address.id,
            promotion_id=promotion.id,
            synthetic_labels=SyntheticEventLabels(
                is_coordinated_abuse=True,
                abuse_types=(AbuseType.PROMOTION_ABUSE,),
                coordination_cluster_id="ring-017",
            ),
        )

        self.assertEqual(account.status, AccountStatus.ACTIVE)
        self.assertEqual(order.total_paise, 8_000)
        assert event.synthetic_labels is not None
        self.assertEqual(event.synthetic_labels.coordination_cluster_id, "ring-017")
        self.assertEqual(refund.state, RefundState.REQUESTED)
        self.assertEqual(address.country_code, "IN")


class SchemaValidationTests(unittest.TestCase):
    def test_naive_timestamp_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            _ = Account(
                id=AccountId(uuid4()),
                created_at=datetime(2026, 8, 24, 12, 0),  # noqa: DTZ001 - intentional naive input.
            )

    def test_order_must_balance_in_paise(self) -> None:
        with self.assertRaisesRegex(ValidationError, "total_paise must equal"):
            _ = Order(
                id=OrderId(uuid4()),
                account_id=AccountId(uuid4()),
                shipping_address_id=AddressId(uuid4()),
                placed_at=NOW,
                subtotal_paise=1_000,
                discount_paise=100,
                total_paise=950,
                item_count=1,
            )

    def test_terminal_refund_requires_resolution_time(self) -> None:
        with self.assertRaisesRegex(ValidationError, "terminal refund states"):
            _ = Refund(
                id=RefundId(uuid4()),
                order_id=OrderId(uuid4()),
                amount_paise=100,
                requested_at=NOW,
                state=RefundState.COMPLETED,
                reason_code="duplicate",
            )

    def test_event_requires_its_graph_relationships(self) -> None:
        with self.assertRaisesRegex(ValidationError, "requires order_id, promotion_id"):
            _ = Event(
                id=EventId(uuid4()),
                event_type=EventType.PROMOTION_REDEEMED,
                occurred_at=NOW,
                ingested_at=NOW,
                account_id=AccountId(uuid4()),
            )

    def test_synthetic_labels_are_internally_consistent(self) -> None:
        with self.assertRaisesRegex(
            ValidationError, "must include at least one abuse type"
        ):
            _ = SyntheticEventLabels(is_coordinated_abuse=True)
        with self.assertRaisesRegex(ValidationError, "non-abusive events"):
            _ = SyntheticEventLabels(
                is_coordinated_abuse=False,
                abuse_types=(AbuseType.REFUND_ABUSE,),
            )


if __name__ == "__main__":
    _ = unittest.main()
