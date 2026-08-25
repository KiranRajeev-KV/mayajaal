"""Persona-driven temporal world generation, independent of storage and ML."""

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from ipaddress import IPv4Address
from uuid import NAMESPACE_URL, UUID, uuid5

import numpy as np
from numpy.random import Generator
from pydantic import BaseModel

from mayajaal.schemas import (
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

from .abuse import AbuseStrategy, CampaignPlan, plan_campaign
from .contexts import ContextSharingPlan, household_plan, office_network_plan, shares
from .cosmetics import CosmeticFactory, CosmeticIdentity
from .personas import PersonaSpec, choose_persona
from .profile import GenerationProfile
from .randomness import generator_for


@dataclass(frozen=True)
class IdentityRefs:
    """Stable graph references active for one account interaction."""

    device_id: DeviceId
    ip_address_id: IPAddressId
    address_id: AddressId
    payment_identity_id: PaymentIdentityId


@dataclass(frozen=True)
class IdentityMetadata:
    """Private generation details used to materialize canonical identity records."""

    kind: str
    fingerprint: str | None = None
    device_type: DeviceType | None = None
    platform: DevicePlatform | None = None
    payment_method: PaymentMethod | None = None
    is_proxy_or_vpn: bool | None = None


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
    """Build an ordinary commerce ecosystem and hidden abuse campaigns.

    Campaign plans only influence which immutable events receive synthetic
    evaluation labels. They are not exported and downstream graph/features do
    not receive them.
    """

    def __init__(self, profile: GenerationProfile) -> None:
        self.profile = profile
        self.cosmetics = CosmeticFactory(profile.seed)
        self._serials: defaultdict[str, int] = defaultdict(int)
        self._seen: defaultdict[UUID, list[datetime]] = defaultdict(list)
        self._used_addresses: set[UUID] = set()
        self.accounts: list[Account] = []
        self.addresses: list[Address] = []
        self.orders: list[Order] = []
        self.refunds: list[Refund] = []
        self.events: list[Event] = []
        self._identities: dict[UUID, IdentityMetadata] = {}
        self._promotions: list[Promotion] = []

    def generate(self) -> SyntheticWorld:
        """Generate all populations, then materialize only observed identities."""
        self._create_promotions()
        for number in range(self.profile.normal_account_count):
            self._create_independent_customer(number)
        for number in range(self.profile.shared_household_count):
            self._create_benign_context(
                f"household-{number}",
                household_plan(),
                self.profile.accounts_per_shared_household,
            )
        for number in range(self.profile.population.benign_network_group_count):
            self._create_benign_context(
                f"office-{number}",
                office_network_plan(),
                self.profile.population.accounts_per_benign_network_group,
            )
        self._create_rings(AbuseStrategy.PROMO_FARM, self.profile.promo_ring_count)
        self._create_rings(AbuseStrategy.REFUND_ABUSE, self.profile.refund_ring_count)
        self._create_rings(AbuseStrategy.MIXED, self.profile.mixed_ring_count)
        self.events.sort(key=lambda event: (event.occurred_at, str(event.id)))
        return SyntheticWorld(
            accounts=tuple(self.accounts),
            devices=tuple(self._devices()),
            ip_addresses=tuple(self._ip_addresses()),
            addresses=tuple(
                address
                for address in self.addresses
                if self._as_uuid(address.id) in self._used_addresses
            ),
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
        if not isinstance(value, UUID):
            raise TypeError("schema ID must be UUID-backed")
        return value

    def _timestamp(self, rng: Generator) -> datetime:
        """Choose a creation time with enough room for subsequent activity."""
        available_days = max((self.profile.end_at - self.profile.start_at).days - 7, 1)
        return self.profile.start_at + timedelta(
            days=int(rng.integers(0, available_days)),
            minutes=int(rng.integers(0, 24 * 60)),
        )

    def _campaign_timestamp(self, campaign: CampaignPlan, rng: Generator) -> datetime:
        """Create a campaign cohort near its activity window, with warm-up room."""
        available_days = max((self.profile.end_at - self.profile.start_at).days - 7, 1)
        center = campaign.activity_center_fraction
        if campaign.low_and_slow:
            center = max(center - 0.22, 0.0)
        jitter = int(rng.integers(-1_000, 1_001)) / 1_000.0
        day = int(
            np.clip(
                round(
                    available_days
                    * (center + jitter * campaign.activity_spread_fraction)
                ),
                0,
                available_days,
            )
        )
        return self.profile.start_at + timedelta(
            days=day,
            minutes=int(rng.integers(0, 24 * 60)),
        )

    @staticmethod
    def _phone(rng: Generator) -> str:
        digits = "".join(str(value) for value in rng.integers(0, 10, size=10))
        return f"+91{digits}"

    def _create_account(self, scope: str, created_at: datetime) -> Account:
        cosmetic = self.cosmetics.identity()
        account = Account(
            id=AccountId(self._uuid("account")),
            created_at=created_at,
            email=cosmetic.email,
            phone_e164=self._phone(generator_for(self.profile.seed, f"phone:{scope}")),
        )
        self.accounts.append(account)
        return account

    def _create_device(self, scope: str, persona: PersonaSpec) -> DeviceId:
        rng = generator_for(self.profile.seed, f"device:{scope}")
        device_id = DeviceId(self._uuid("device"))
        mobile = bool(rng.random() < persona.mobile_probability)
        if mobile:
            device_type = DeviceType.MOBILE
            platform = (
                DevicePlatform.ANDROID if rng.random() < 0.72 else DevicePlatform.IOS
            )
        else:
            device_type = DeviceType.DESKTOP
            platform = (
                DevicePlatform.WINDOWS if rng.random() < 0.64 else DevicePlatform.MACOS
            )
        self._identities[self._as_uuid(device_id)] = IdentityMetadata(
            kind="device",
            fingerprint=f"device-{scope}-{device_id}",
            device_type=device_type,
            platform=platform,
        )
        return device_id

    def _create_ip(self, scope: str, *, proxy: bool = False) -> IPAddressId:
        ip_address_id = IPAddressId(self._uuid("ip_address"))
        self._identities[self._as_uuid(ip_address_id)] = IdentityMetadata(
            kind="ip_address",
            fingerprint=f"ip-{scope}",
            is_proxy_or_vpn=proxy,
        )
        return ip_address_id

    def _create_address(
        self, scope: str, cosmetic: CosmeticIdentity | None = None
    ) -> AddressId:
        address_id = AddressId(self._uuid("address"))
        identity = cosmetic or self.cosmetics.identity()
        self._identities[self._as_uuid(address_id)] = IdentityMetadata(kind="address")
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
        return address_id

    def _create_payment(self, scope: str, persona: PersonaSpec) -> PaymentIdentityId:
        rng = generator_for(self.profile.seed, f"payment:{scope}")
        payment_identity_id = PaymentIdentityId(self._uuid("payment_identity"))
        methods = (
            PaymentMethod.UPI,
            PaymentMethod.CARD,
            PaymentMethod.WALLET,
            PaymentMethod.BANK_TRANSFER,
            PaymentMethod.CASH_ON_DELIVERY,
        )
        weights = (
            (0.54, 0.31, 0.10, 0.03, 0.02)
            if persona.mobile_probability >= 0.7
            else (0.37, 0.45, 0.08, 0.06, 0.04)
        )
        method = methods[int(rng.choice(len(methods), p=weights))]
        self._identities[self._as_uuid(payment_identity_id)] = IdentityMetadata(
            kind="payment_identity",
            fingerprint=f"payment-{scope}-{payment_identity_id}",
            payment_method=method,
        )
        return payment_identity_id

    def _new_refs(
        self,
        scope: str,
        persona: PersonaSpec,
        *,
        shared_refs: IdentityRefs | None = None,
        share_device: bool = False,
        share_ip: bool = False,
        share_address: bool = False,
        share_payment: bool = False,
    ) -> IdentityRefs:
        """Allocate an account's starting identity mix without unused local IDs."""
        return IdentityRefs(
            device_id=(
                shared_refs.device_id
                if shared_refs is not None and share_device
                else self._create_device(scope, persona)
            ),
            ip_address_id=(
                shared_refs.ip_address_id
                if shared_refs is not None and share_ip
                else self._create_ip(scope)
            ),
            address_id=(
                shared_refs.address_id
                if shared_refs is not None and share_address
                else self._create_address(scope)
            ),
            payment_identity_id=(
                shared_refs.payment_identity_id
                if shared_refs is not None and share_payment
                else self._create_payment(scope, persona)
            ),
        )

    def _create_independent_customer(self, number: int) -> None:
        scope = f"normal-{number}"
        rng = generator_for(self.profile.seed, scope)
        persona = choose_persona(self.profile.population.persona_weights, rng)
        created_at = self._timestamp(rng)
        account = self._create_account(scope, created_at)
        self._emit_history(
            account, self._new_refs(scope, persona), persona, scope, None
        )

    def _create_benign_context(
        self, scope: str, plan: ContextSharingPlan, member_count: int
    ) -> None:
        """Create households and office/campus-like groups without labels."""
        context_rng = generator_for(self.profile.seed, f"context:{scope}")
        seed_persona = choose_persona(
            self.profile.population.persona_weights, context_rng
        )
        shared_refs = self._new_refs(f"shared-{scope}", seed_persona)
        for member_number in range(member_count):
            member_scope = f"{scope}-member-{member_number}"
            rng = generator_for(self.profile.seed, member_scope)
            persona = choose_persona(self.profile.population.persona_weights, rng)
            account = self._create_account(member_scope, self._timestamp(rng))
            refs = self._new_refs(
                member_scope,
                persona,
                shared_refs=shared_refs,
                share_device=shares(plan.share_device_probability, rng),
                share_ip=plan.share_ip,
                share_address=plan.share_address,
                share_payment=shares(plan.share_payment_probability, rng),
            )
            self._emit_history(account, refs, persona, member_scope, None)

    def _create_rings(self, strategy: AbuseStrategy, count: int) -> None:
        for ring_number in range(count):
            scope = f"{strategy.value}-ring-{ring_number:03d}"
            rng = generator_for(self.profile.seed, scope)
            campaign = plan_campaign(
                strategy,
                scope,
                self.profile.abuse,
                self.profile.difficulty,
                rng,
            )
            seed_persona = choose_persona(self.profile.population.persona_weights, rng)
            shared_refs = self._new_refs(f"shared-{scope}", seed_persona)
            variation = self.profile.abuse.ring_size_variation
            member_count = max(
                2,
                self.profile.accounts_per_ring
                + int(rng.integers(-variation, variation + 1)),
            )
            for member_number in range(member_count):
                member_scope = f"{scope}-member-{member_number}"
                member_rng = generator_for(self.profile.seed, member_scope)
                persona = choose_persona(
                    self.profile.population.persona_weights, member_rng
                )
                account = self._create_account(
                    member_scope, self._campaign_timestamp(campaign, member_rng)
                )
                early_member = member_number < 2
                refs = self._new_refs(
                    member_scope,
                    persona,
                    shared_refs=shared_refs,
                    share_device=campaign.shared_device
                    and (early_member or bool(member_rng.random() < 0.62)),
                    share_ip=campaign.shared_ip
                    and (early_member or bool(member_rng.random() < 0.62)),
                    share_address=campaign.shared_address
                    and (early_member or bool(member_rng.random() < 0.62)),
                    share_payment=campaign.shared_payment
                    and (early_member or bool(member_rng.random() < 0.62)),
                )
                self._emit_history(account, refs, persona, member_scope, campaign)

    def _create_promotions(self) -> None:
        promotion_rows = (
            (
                "WELCOME10",
                "Welcome discount",
                PromotionDiscountType.PERCENTAGE_BPS,
                1_000,
            ),
            (
                "FLASH50",
                "Flash acquisition campaign",
                PromotionDiscountType.PERCENTAGE_BPS,
                5_000,
            ),
            (
                "WEEKEND15",
                "Weekend basket campaign",
                PromotionDiscountType.PERCENTAGE_BPS,
                1_500,
            ),
            (
                "FREESHIP50",
                "Shipping incentive",
                PromotionDiscountType.FIXED_PAISE,
                5_000,
            ),
        )
        self._promotions = [
            Promotion(
                id=PromotionId(self._uuid("promotion")),
                code=code,
                campaign_name=campaign_name,
                discount_type=discount_type,
                discount_value=discount_value,
                valid_from=self.profile.start_at,
                valid_until=self.profile.end_at,
                max_redemptions_per_account=1,
            )
            for code, campaign_name, discount_type, discount_value in promotion_rows
        ]

    @staticmethod
    def _labels(campaign: CampaignPlan | None) -> SyntheticEventLabels | None:
        if campaign is None:
            return None
        return SyntheticEventLabels(
            is_coordinated_abuse=True,
            abuse_types=campaign.abuse_types,
            coordination_cluster_id=campaign.cluster_id,
        )

    def _add_event(
        self,
        event_type: EventType,
        occurred_at: datetime,
        account_id: AccountId,
        *,
        refs: IdentityRefs | None = None,
        order_id: OrderId | None = None,
        promotion_id: PromotionId | None = None,
        refund_id: RefundId | None = None,
        labels: SyntheticEventLabels | None = None,
    ) -> None:
        event = Event(
            id=EventId(self._uuid("event")),
            event_type=event_type,
            occurred_at=occurred_at,
            ingested_at=occurred_at + timedelta(seconds=1),
            account_id=account_id,
            device_id=refs.device_id if refs is not None else None,
            ip_address_id=refs.ip_address_id if refs is not None else None,
            payment_identity_id=refs.payment_identity_id if refs is not None else None,
            order_id=order_id,
            address_id=refs.address_id if refs is not None else None,
            promotion_id=promotion_id,
            refund_id=refund_id,
            synthetic_labels=labels,
        )
        self.events.append(event)
        if event_type is EventType.DEVICE_SEEN and event.device_id is not None:
            self._seen[self._as_uuid(event.device_id)].append(occurred_at)
        elif event_type is EventType.IP_SEEN and event.ip_address_id is not None:
            self._seen[self._as_uuid(event.ip_address_id)].append(occurred_at)
        elif (
            event_type is EventType.PAYMENT_ATTACHED
            and event.payment_identity_id is not None
        ):
            self._seen[self._as_uuid(event.payment_identity_id)].append(occurred_at)
        elif event_type is EventType.ORDER_PLACED and event.address_id is not None:
            self._used_addresses.add(self._as_uuid(event.address_id))

    def _order_times(
        self,
        created_at: datetime,
        count: int,
        persona: PersonaSpec,
        rng: Generator,
        campaign: CampaignPlan | None,
    ) -> tuple[datetime, ...]:
        """Schedule persona-shaped orders without facts beyond the window."""
        if count == 0:
            return ()
        earliest = created_at + timedelta(hours=1)
        latest = self.profile.end_at - timedelta(days=4)
        span_seconds = max((latest - earliest).total_seconds(), 1.0)
        if campaign is not None and campaign.low_and_slow:
            fractions = np.linspace(0.18, 0.92, count) + rng.normal(0.0, 0.045, count)
        elif campaign is not None:
            fractions = campaign.activity_center_fraction + rng.normal(
                0.0, campaign.activity_spread_fraction, count
            )
        else:
            fractions = rng.random(count)
            calendar = self.profile.calendar
            seasonal = rng.random(count) < calendar.seasonal_activity_share
            seasonal_fraction = calendar.seasonal_window_center_fraction + rng.normal(
                0.0, calendar.seasonal_window_width_fraction / 3.0, count
            )
            fractions = np.where(seasonal, seasonal_fraction, fractions)
            if self.profile.difficulty.value == "drift":
                fractions = np.clip(
                    fractions**calendar.drift_late_activity_power, 0.0, 1.0
                )
        times: list[datetime] = []
        for fraction in fractions:
            provisional = earliest + timedelta(
                seconds=float(np.clip(fraction, 0.0, 1.0) * span_seconds)
            )
            hour = int(np.clip(round(rng.normal(persona.preferred_hour, 3.0)), 0, 23))
            candidate = provisional.astimezone(UTC).replace(
                hour=hour,
                minute=int(rng.integers(0, 60)),
                second=0,
                microsecond=0,
            )
            times.append(min(max(candidate, earliest), latest))
        return tuple(sorted(times))

    def _replace_refs(
        self,
        current: IdentityRefs,
        scope: str,
        persona: PersonaSpec,
        rng: Generator,
    ) -> IdentityRefs:
        """Apply ordinary device/payment/address/IP lifecycle changes."""
        lifecycle = self.profile.identity_lifecycle
        return IdentityRefs(
            device_id=(
                self._create_device(f"{scope}-device", persona)
                if rng.random() < lifecycle.additional_device_probability
                else current.device_id
            ),
            ip_address_id=(
                self._create_ip(f"{scope}-travel-ip", proxy=bool(rng.random() < 0.06))
                if rng.random()
                < lifecycle.travel_ip_probability * persona.travel_multiplier
                else current.ip_address_id
            ),
            address_id=(
                self._create_address(f"{scope}-address")
                if rng.random() < lifecycle.additional_address_probability
                else current.address_id
            ),
            payment_identity_id=(
                self._create_payment(f"{scope}-payment", persona)
                if rng.random() < lifecycle.additional_payment_probability
                else current.payment_identity_id
            ),
        )

    def _promotion_for(
        self,
        persona: PersonaSpec,
        rng: Generator,
        campaign: CampaignPlan | None,
        abusive: bool,
    ) -> Promotion | None:
        if (
            abusive
            and campaign is not None
            and campaign.strategy
            in {
                AbuseStrategy.PROMO_FARM,
                AbuseStrategy.MIXED,
            }
        ):
            return self._promotions[1]
        if rng.random() >= persona.promotion_probability:
            return None
        # Flash campaigns are also redeemed by ordinary customers; campaign
        # membership must not become a label-exclusive promotion category.
        return self._promotions[
            int(rng.choice((0, 1, 2, 3), p=(0.35, 0.12, 0.33, 0.20)))
        ]

    @staticmethod
    def _order_amounts(
        persona: PersonaSpec, promotion: Promotion | None, rng: Generator
    ) -> tuple[int, int, int]:
        # A mixture gives an ordinary long-tailed basket distribution without
        # relying on a global random state or a label-conditioned amount.
        percentile = int(rng.integers(0, 100))
        base_value = (
            int(rng.integers(800, 8_000))
            if percentile < 78
            else int(rng.integers(8_000, 30_000))
            if percentile < 96
            else int(rng.integers(30_000, 90_000))
        )
        subtotal = min(max(round(base_value * persona.value_multiplier), 500), 150_000)
        discount = 0
        if promotion is not None:
            if promotion.discount_type is PromotionDiscountType.PERCENTAGE_BPS:
                discount = subtotal * promotion.discount_value // 10_000
            else:
                discount = min(subtotal, promotion.discount_value)
        return subtotal, discount, subtotal - discount

    def _emit_history(
        self,
        account: Account,
        initial_refs: IdentityRefs,
        persona: PersonaSpec,
        scope: str,
        campaign: CampaignPlan | None,
    ) -> None:
        """Emit ordinary history plus an optional hidden campaign action."""
        rng = generator_for(self.profile.seed, f"history:{scope}")
        self._add_event(EventType.ACCOUNT_CREATED, account.created_at, account.id)
        current_refs = initial_refs
        first_seen_at = account.created_at + timedelta(minutes=5)
        self._add_event(
            EventType.DEVICE_SEEN, first_seen_at, account.id, refs=current_refs
        )
        self._add_event(
            EventType.IP_SEEN,
            first_seen_at + timedelta(minutes=2),
            account.id,
            refs=current_refs,
        )
        self._add_event(
            EventType.PAYMENT_ATTACHED,
            first_seen_at + timedelta(minutes=4),
            account.id,
            refs=current_refs,
        )
        ordinary_count = int(rng.poisson(persona.order_rate))
        if campaign is not None:
            ordinary_count = max(ordinary_count, campaign.warmup_orders + 1)
        order_times = self._order_times(
            account.created_at,
            ordinary_count,
            persona,
            rng,
            campaign,
        )
        for number, order_at in enumerate(order_times):
            if number > 0:
                current_refs = self._replace_refs(
                    current_refs, f"{scope}-{number}", persona, rng
                )
            self._add_event(
                EventType.DEVICE_SEEN,
                order_at - timedelta(minutes=5),
                account.id,
                refs=current_refs,
            )
            self._add_event(
                EventType.IP_SEEN,
                order_at - timedelta(minutes=3),
                account.id,
                refs=current_refs,
            )
            self._add_event(
                EventType.PAYMENT_ATTACHED,
                order_at - timedelta(minutes=1),
                account.id,
                refs=current_refs,
            )
            abusive = campaign is not None and number == len(order_times) - 1
            promotion = self._promotion_for(persona, rng, campaign, abusive)
            subtotal, discount, total = self._order_amounts(persona, promotion, rng)
            order = Order(
                id=OrderId(self._uuid("order")),
                account_id=account.id,
                shipping_address_id=current_refs.address_id,
                placed_at=order_at,
                subtotal_paise=subtotal,
                discount_paise=discount,
                total_paise=total,
                item_count=max(1, int(rng.poisson(2.1))),
                status=OrderStatus.DELIVERED,
                promotion_id=promotion.id if promotion is not None else None,
            )
            self.orders.append(order)
            self._add_event(
                EventType.ORDER_PLACED,
                order_at,
                account.id,
                refs=current_refs,
                order_id=order.id,
            )
            if promotion is not None:
                self._add_event(
                    EventType.PROMOTION_REDEEMED,
                    order_at + timedelta(minutes=1),
                    account.id,
                    refs=current_refs,
                    order_id=order.id,
                    promotion_id=promotion.id,
                    labels=self._labels(campaign) if abusive else None,
                )
            fraudulent_refund = (
                abusive
                and campaign is not None
                and campaign.strategy
                in {
                    AbuseStrategy.REFUND_ABUSE,
                    AbuseStrategy.MIXED,
                }
            )
            ordinary_refund = bool(
                rng.random()
                < self.profile.commerce.normal_refund_probability
                * persona.refund_multiplier
            )
            if fraudulent_refund or ordinary_refund:
                self._emit_refund(
                    account,
                    current_refs,
                    order,
                    campaign if fraudulent_refund else None,
                    rng,
                )

    def _emit_refund(
        self,
        account: Account,
        refs: IdentityRefs,
        order: Order,
        campaign: CampaignPlan | None,
        rng: Generator,
    ) -> None:
        requested_at = min(
            order.placed_at + timedelta(days=int(rng.integers(1, 3))),
            self.profile.end_at - timedelta(days=1),
        )
        resolved_at = min(
            requested_at + timedelta(days=int(rng.integers(1, 3))),
            self.profile.end_at,
        )
        amount = max(1, int(order.total_paise * float(rng.uniform(0.45, 1.0))))
        refund = Refund(
            id=RefundId(self._uuid("refund")),
            order_id=order.id,
            amount_paise=amount,
            requested_at=requested_at,
            state=RefundState.COMPLETED,
            resolved_at=resolved_at,
            reason_code=str(
                rng.choice(
                    ("item_not_received", "damaged", "wrong_item", "changed_mind")
                )
            ),
        )
        self.refunds.append(refund)
        labels = self._labels(campaign)
        self._add_event(
            EventType.REFUND_REQUESTED,
            requested_at,
            account.id,
            refs=refs,
            order_id=order.id,
            refund_id=refund.id,
            labels=labels,
        )
        self._add_event(
            EventType.REFUND_RESOLVED,
            resolved_at,
            account.id,
            refs=refs,
            order_id=order.id,
            refund_id=refund.id,
            labels=labels,
        )

    def _devices(self) -> list[Device]:
        return [
            Device(
                id=DeviceId(identity_id),
                fingerprint=metadata.fingerprint or f"device-{identity_id}",
                device_type=metadata.device_type or DeviceType.OTHER,
                platform=metadata.platform or DevicePlatform.OTHER,
                first_seen_at=min(self._seen[identity_id]),
                last_seen_at=max(self._seen[identity_id]),
            )
            for identity_id, metadata in sorted(self._identities.items())
            if metadata.kind == "device" and self._seen[identity_id]
        ]

    def _ip_addresses(self) -> list[IPAddress]:
        addresses: list[IPAddress] = []
        identity_ids = [
            identity_id
            for identity_id, metadata in sorted(self._identities.items())
            if metadata.kind == "ip_address" and self._seen[identity_id]
        ]
        for ip_number, identity_id in enumerate(identity_ids, start=1):
            octet_3, octet_4 = divmod(ip_number - 1, 254)
            metadata = self._identities[identity_id]
            addresses.append(
                IPAddress(
                    id=IPAddressId(identity_id),
                    address=IPv4Address(f"198.51.{octet_3}.{octet_4 + 1}"),
                    first_seen_at=min(self._seen[identity_id]),
                    last_seen_at=max(self._seen[identity_id]),
                    is_proxy_or_vpn=metadata.is_proxy_or_vpn,
                )
            )
        return addresses

    def _payment_identities(self) -> list[PaymentIdentity]:
        return [
            PaymentIdentity(
                id=PaymentIdentityId(identity_id),
                method=metadata.payment_method or PaymentMethod.UPI,
                fingerprint=metadata.fingerprint or f"payment-{identity_id}",
                first_seen_at=min(self._seen[identity_id]),
                last_seen_at=max(self._seen[identity_id]),
                issuer_country_code="IN",
            )
            for identity_id, metadata in sorted(self._identities.items())
            if metadata.kind == "payment_identity" and self._seen[identity_id]
        ]


def generate_world(profile: GenerationProfile) -> SyntheticWorld:
    """Generate a validated, reproducible synthetic world for one profile."""
    return WorldGenerator(profile).generate()
