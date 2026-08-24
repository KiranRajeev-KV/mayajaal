"""Scenario-driven temporal world generation, independent of storage and ML."""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from ipaddress import IPv4Address
from typing import NewType
from uuid import NAMESPACE_URL, UUID, uuid5

import numpy as np
from numpy.random import Generator
from pydantic import BaseModel

from mayajaal.schemas import (
    AbuseType,
    Account,
    AccountId,
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
    OrderStatus,
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

from .cosmetics import CosmeticFactory, CosmeticIdentity
from .profile import GenerationProfile

RingKind = NewType("RingKind", str)


@dataclass(frozen=True)
class IdentityRefs:
    """Stable graph references assigned to one account's generated activity."""

    device_id: DeviceId
    ip_address_id: IPAddressId
    address_id: AddressId
    payment_identity_id: PaymentIdentityId


@dataclass(frozen=True)
class SyntheticWorld:
    """Validated entity/event records produced by one deterministic scenario run."""

    accounts: tuple[Account, ...]
    devices: tuple[Device, ...]
    ip_addresses: tuple[IPAddress, ...]
    addresses: tuple[Address, ...]
    payment_identities: tuple[PaymentIdentity, ...]
    orders: tuple[Order, ...]
    promotions: tuple[Promotion, ...]
    refunds: tuple[Refund, ...]
    events: tuple[Event, ...]

    def all_models(self) -> tuple[BaseModel, ...]:
        """Return every generated Pydantic record for validation-oriented callers."""
        return (
            *self.accounts,
            *self.devices,
            *self.ip_addresses,
            *self.addresses,
            *self.payment_identities,
            *self.orders,
            *self.promotions,
            *self.refunds,
            *self.events,
        )


class WorldGenerator:
    """Builds normal and fraudulent histories from a known scenario topology."""

    def __init__(self, profile: GenerationProfile) -> None:
        self.profile = profile
        self.rng: Generator = np.random.default_rng(profile.seed)
        self.cosmetics = CosmeticFactory(profile.seed)
        self._serials: defaultdict[str, int] = defaultdict(int)
        self._seen: defaultdict[UUID, list[datetime]] = defaultdict(list)
        self.accounts: list[Account] = []
        self.addresses: list[Address] = []
        self.orders: list[Order] = []
        self.refunds: list[Refund] = []
        self.events: list[Event] = []
        self._identities: dict[UUID, tuple[str, CosmeticIdentity | None]] = {}

    def generate(self) -> SyntheticWorld:
        """Generate each population and finalize identity time windows."""
        self._create_promotions()
        for _ in range(self.profile.normal_account_count):
            self._create_independent_customer()
        for household_number in range(self.profile.shared_household_count):
            self._create_shared_household(household_number)
        self._create_rings(RingKind("promo"), self.profile.promo_ring_count)
        self._create_rings(RingKind("refund"), self.profile.refund_ring_count)
        self._create_rings(RingKind("mixed"), self.profile.mixed_ring_count)
        self.events.sort(key=lambda event: (event.occurred_at, str(event.id)))
        return SyntheticWorld(
            accounts=tuple(self.accounts),
            devices=tuple(self._devices()),
            ip_addresses=tuple(self._ip_addresses()),
            addresses=tuple(self.addresses),
            payment_identities=tuple(self._payment_identities()),
            orders=tuple(self.orders),
            promotions=tuple(self._promotions),
            refunds=tuple(self.refunds),
            events=tuple(self.events),
        )

    def _uuid(self, kind: str) -> UUID:
        self._serials[kind] += 1
        return uuid5(
            NAMESPACE_URL, f"mayajaal:{self.profile.seed}:{kind}:{self._serials[kind]}"
        )

    @staticmethod
    def _as_uuid(value: object) -> UUID:
        """Expose the UUID runtime representation of a nominal schema ID."""
        if not isinstance(value, UUID):
            raise TypeError("schema ID must be UUID-backed")
        return value

    def _timestamp(self) -> datetime:
        available_days = (self.profile.end_at - self.profile.start_at).days - 8
        return self.profile.start_at + timedelta(
            days=int(self.rng.integers(0, available_days)),
            minutes=int(self.rng.integers(0, 24 * 60)),
        )

    def _phone(self) -> str:
        digits = "".join(str(value) for value in self.rng.integers(0, 10, size=10))
        return f"+91{digits}"

    def _create_account(self, created_at: datetime) -> Account:
        cosmetic = self.cosmetics.identity()
        account = Account(
            id=AccountId(self._uuid("account")),
            created_at=created_at,
            email=cosmetic.email,
            phone_e164=self._phone(),
        )
        self.accounts.append(account)
        return account

    def _new_refs(
        self, scope: str, cosmetic: CosmeticIdentity | None = None
    ) -> IdentityRefs:
        device_id = DeviceId(self._uuid("device"))
        ip_address_id = IPAddressId(self._uuid("ip_address"))
        address_id = AddressId(self._uuid("address"))
        payment_identity_id = PaymentIdentityId(self._uuid("payment_identity"))
        identity = cosmetic or self.cosmetics.identity()
        self._identities[self._as_uuid(device_id)] = (
            f"device-{scope}-{device_id}",
            None,
        )
        self._identities[self._as_uuid(ip_address_id)] = (f"ip-{scope}", None)
        self._identities[self._as_uuid(address_id)] = (f"address-{scope}", identity)
        self._identities[self._as_uuid(payment_identity_id)] = (
            f"payment-{scope}-{payment_identity_id}",
            None,
        )
        self.addresses.append(
            Address(
                id=address_id,
                recipient_name=identity.name,
                line1=identity.line1,
                city=identity.city,
                region=identity.region,
                postal_code=identity.postal_code,
                country_code="IN",
            )
        )
        return IdentityRefs(device_id, ip_address_id, address_id, payment_identity_id)

    def _create_independent_customer(self) -> None:
        created_at = self._timestamp()
        account = self._create_account(created_at)
        refs = self._new_refs(f"normal-{account.id}")
        self._emit_history(account, refs, created_at, None, None)

    def _create_shared_household(self, household_number: int) -> None:
        household_cosmetic = self.cosmetics.identity()
        shared_refs = self._new_refs(
            f"household-{household_number}", household_cosmetic
        )
        for member_number in range(self.profile.accounts_per_shared_household):
            created_at = self._timestamp()
            account = self._create_account(created_at)
            payment_identity_id = (
                shared_refs.payment_identity_id
                if member_number == 0
                else PaymentIdentityId(self._uuid("payment_identity"))
            )
            member_refs = IdentityRefs(
                device_id=shared_refs.device_id,
                ip_address_id=shared_refs.ip_address_id,
                address_id=shared_refs.address_id,
                payment_identity_id=payment_identity_id,
            )
            if member_number > 0:
                self._identities[self._as_uuid(member_refs.payment_identity_id)] = (
                    f"payment-household-{household_number}-{member_number}",
                    None,
                )
            self._emit_history(account, member_refs, created_at, None, None)

    def _create_rings(self, kind: RingKind, count: int) -> None:
        for ring_number in range(count):
            shared_refs = self._new_refs(f"{kind}-{ring_number}")
            cluster_id = f"{kind}-ring-{ring_number:03d}"
            for _ in range(self.profile.accounts_per_ring):
                created_at = self._timestamp()
                account = self._create_account(created_at)
                self._emit_history(account, shared_refs, created_at, kind, cluster_id)

    def _create_promotions(self) -> None:
        self._promotions = [
            Promotion(
                id=PromotionId(self._uuid("promotion")),
                code="WELCOME10",
                campaign_name="Welcome discount",
                discount_type=PromotionDiscountType.PERCENTAGE_BPS,
                discount_value=1_000,
                valid_from=self.profile.start_at,
                valid_until=self.profile.end_at,
                max_redemptions_per_account=1,
            ),
            Promotion(
                id=PromotionId(self._uuid("promotion")),
                code="FLASH50",
                campaign_name="Flash acquisition campaign",
                discount_type=PromotionDiscountType.PERCENTAGE_BPS,
                discount_value=5_000,
                valid_from=self.profile.start_at,
                valid_until=self.profile.end_at,
                max_redemptions_per_account=1,
            ),
        ]

    def _labels(
        self, kind: RingKind | None, cluster_id: str | None
    ) -> SyntheticEventLabels | None:
        if kind is None:
            return None
        abuse_types: dict[RingKind, tuple[AbuseType, ...]] = {
            RingKind("promo"): (AbuseType.PROMOTION_ABUSE, AbuseType.ACCOUNT_FARMING),
            RingKind("refund"): (AbuseType.REFUND_ABUSE, AbuseType.ADDRESS_REUSE),
            RingKind("mixed"): (AbuseType.PROMOTION_ABUSE, AbuseType.REFUND_ABUSE),
        }
        return SyntheticEventLabels(
            is_coordinated_abuse=True,
            abuse_types=abuse_types[kind],
            coordination_cluster_id=cluster_id,
        )

    def _add_event(
        self,
        event_type: EventType,
        occurred_at: datetime,
        account_id: AccountId,
        refs: IdentityRefs,
        kind: RingKind | None,
        cluster_id: str | None,
        *,
        order_id: OrderId | None = None,
        promotion_id: PromotionId | None = None,
        refund_id: RefundId | None = None,
    ) -> None:
        self.events.append(
            Event(
                id=EventId(self._uuid("event")),
                event_type=event_type,
                occurred_at=occurred_at,
                ingested_at=occurred_at
                + timedelta(seconds=int(self.rng.integers(1, 60))),
                account_id=account_id,
                device_id=refs.device_id,
                ip_address_id=refs.ip_address_id,
                payment_identity_id=refs.payment_identity_id,
                order_id=order_id,
                address_id=refs.address_id,
                promotion_id=promotion_id,
                refund_id=refund_id,
                synthetic_labels=self._labels(kind, cluster_id),
            )
        )
        for identity_id in (
            self._as_uuid(refs.device_id),
            self._as_uuid(refs.ip_address_id),
            self._as_uuid(refs.payment_identity_id),
        ):
            self._seen[identity_id].append(occurred_at)

    def _emit_history(
        self,
        account: Account,
        refs: IdentityRefs,
        created_at: datetime,
        kind: RingKind | None,
        cluster_id: str | None,
    ) -> None:
        labels = (kind, cluster_id)
        device_at = created_at + timedelta(minutes=5)
        ip_at = device_at + timedelta(minutes=2)
        payment_at = ip_at + timedelta(minutes=3)
        order_at = payment_at + timedelta(hours=int(self.rng.integers(1, 36)))
        self._add_event(
            EventType.ACCOUNT_CREATED, created_at, account.id, refs, *labels
        )
        self._add_event(EventType.DEVICE_SEEN, device_at, account.id, refs, *labels)
        self._add_event(EventType.IP_SEEN, ip_at, account.id, refs, *labels)
        self._add_event(
            EventType.PAYMENT_ATTACHED, payment_at, account.id, refs, *labels
        )

        uses_promo = kind in {RingKind("promo"), RingKind("mixed")} or bool(
            self.rng.integers(0, 2)
        )
        promotion_id = (
            self._promotions[1 if kind is not None else 0].id if uses_promo else None
        )
        subtotal = int(self.rng.integers(2_000, 15_000))
        discount = subtotal // 2 if promotion_id is not None else 0
        order = Order(
            id=OrderId(self._uuid("order")),
            account_id=account.id,
            shipping_address_id=refs.address_id,
            placed_at=order_at,
            subtotal_paise=subtotal,
            discount_paise=discount,
            total_paise=subtotal - discount,
            item_count=int(self.rng.integers(1, 5)),
            status=OrderStatus.DELIVERED,
            promotion_id=promotion_id,
        )
        self.orders.append(order)
        self._add_event(
            EventType.ORDER_PLACED,
            order_at,
            account.id,
            refs,
            *labels,
            order_id=order.id,
        )
        if promotion_id is not None:
            self._add_event(
                EventType.PROMOTION_REDEEMED,
                order_at + timedelta(minutes=1),
                account.id,
                refs,
                *labels,
                order_id=order.id,
                promotion_id=promotion_id,
            )
        if kind in {RingKind("refund"), RingKind("mixed")}:
            assert kind is not None
            assert cluster_id is not None
            self._emit_refund(account, refs, order, kind, cluster_id)

    def _emit_refund(
        self,
        account: Account,
        refs: IdentityRefs,
        order: Order,
        kind: RingKind,
        cluster_id: str,
    ) -> None:
        requested_at = order.placed_at + timedelta(days=2)
        resolved_at = requested_at + timedelta(days=1)
        refund = Refund(
            id=RefundId(self._uuid("refund")),
            order_id=order.id,
            amount_paise=order.total_paise,
            requested_at=requested_at,
            state=RefundState.COMPLETED,
            resolved_at=resolved_at,
            reason_code="item_not_received",
        )
        self.refunds.append(refund)
        self._add_event(
            EventType.REFUND_REQUESTED,
            requested_at,
            account.id,
            refs,
            kind,
            cluster_id,
            order_id=order.id,
            refund_id=refund.id,
        )
        self._add_event(
            EventType.REFUND_RESOLVED,
            resolved_at,
            account.id,
            refs,
            kind,
            cluster_id,
            order_id=order.id,
            refund_id=refund.id,
        )

    def _devices(self) -> list[Device]:
        return [
            Device(
                id=DeviceId(identity_id),
                fingerprint=value,
                device_type=DeviceType.MOBILE,
                platform=DevicePlatform.ANDROID,
                first_seen_at=min(self._seen[identity_id]),
                last_seen_at=max(self._seen[identity_id]),
            )
            for identity_id, (value, _) in self._identities.items()
            if value.startswith("device-")
        ]

    def _ip_addresses(self) -> list[IPAddress]:
        ip_number = 0
        addresses: list[IPAddress] = []
        for identity_id, (value, _) in self._identities.items():
            if not value.startswith("ip-"):
                continue
            ip_number += 1
            octet_3, octet_4 = divmod(ip_number, 254)
            addresses.append(
                IPAddress(
                    id=IPAddressId(identity_id),
                    address=IPv4Address(f"198.51.{octet_3}.{octet_4 + 1}"),
                    first_seen_at=min(self._seen[identity_id]),
                    last_seen_at=max(self._seen[identity_id]),
                )
            )
        return addresses

    def _payment_identities(self) -> list[PaymentIdentity]:
        return [
            PaymentIdentity(
                id=PaymentIdentityId(identity_id),
                method=PaymentMethod.UPI,
                fingerprint=value,
                first_seen_at=min(self._seen[identity_id]),
                last_seen_at=max(self._seen[identity_id]),
                issuer_country_code="IN",
            )
            for identity_id, (value, _) in self._identities.items()
            if value.startswith("payment-")
        ]


def generate_world(profile: GenerationProfile) -> SyntheticWorld:
    """Generate a validated, reproducible synthetic world for one profile."""
    return WorldGenerator(profile).generate()
