"""Canonical production-shaped entities for the temporal identity graph."""

from enum import StrEnum

from pydantic import Field, IPvAnyAddress, model_validator

from .common import AwareDatetime, SchemaModel
from .ids import (
    AccountId,
    AddressId,
    DeviceId,
    IPAddressId,
    OrderId,
    PaymentIdentityId,
    PromotionId,
    RefundId,
)


class AccountStatus(StrEnum):
    """Lifecycle state of a customer account."""

    ACTIVE = "active"
    SUSPENDED = "suspended"
    CLOSED = "closed"


class DeviceType(StrEnum):
    """Broad device category inferred at collection time."""

    MOBILE = "mobile"
    TABLET = "tablet"
    DESKTOP = "desktop"
    OTHER = "other"


class DevicePlatform(StrEnum):
    """Operating-system family of a device."""

    ANDROID = "android"
    IOS = "ios"
    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"
    OTHER = "other"


class PaymentMethod(StrEnum):
    """Payment rail used by a tokenized payment identity."""

    CARD = "card"
    UPI = "upi"
    WALLET = "wallet"
    BANK_TRANSFER = "bank_transfer"
    CASH_ON_DELIVERY = "cash_on_delivery"


class OrderStatus(StrEnum):
    """Commercial lifecycle state of an order."""

    PLACED = "placed"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class PromotionDiscountType(StrEnum):
    """How a promotion's discount value is interpreted."""

    FIXED_PAISE = "fixed_paise"
    PERCENTAGE_BPS = "percentage_bps"


class RefundState(StrEnum):
    """Lifecycle state of a refund request."""

    REQUESTED = "requested"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class Account(SchemaModel):
    """A customer account; it contains no abuse or synthetic-label fields."""

    id: AccountId
    created_at: AwareDatetime
    status: AccountStatus = AccountStatus.ACTIVE
    email: str | None = Field(default=None, min_length=3, max_length=320)
    phone_e164: str | None = Field(default=None, pattern=r"^\+[1-9]\d{7,14}$")


class Device(SchemaModel):
    """A stable device identity represented by a privacy-safe fingerprint."""

    id: DeviceId
    fingerprint: str = Field(min_length=8, max_length=256)
    device_type: DeviceType
    platform: DevicePlatform
    first_seen_at: AwareDatetime
    last_seen_at: AwareDatetime
    is_emulator: bool = False

    @model_validator(mode="after")
    def validate_seen_window(self) -> "Device":
        if self.last_seen_at < self.first_seen_at:
            raise ValueError("last_seen_at cannot be earlier than first_seen_at")
        return self


class IPAddress(SchemaModel):
    """A public network address observed during an account interaction."""

    id: IPAddressId
    address: IPvAnyAddress
    first_seen_at: AwareDatetime
    last_seen_at: AwareDatetime
    is_proxy_or_vpn: bool | None = None

    @model_validator(mode="after")
    def validate_seen_window(self) -> "IPAddress":
        if self.last_seen_at < self.first_seen_at:
            raise ValueError("last_seen_at cannot be earlier than first_seen_at")
        return self


class Address(SchemaModel):
    """A normalized shipping address, independent of a customer account."""

    id: AddressId
    recipient_name: str = Field(min_length=1, max_length=200)
    line1: str = Field(min_length=1, max_length=200)
    line2: str | None = Field(default=None, max_length=200)
    city: str = Field(min_length=1, max_length=100)
    region: str | None = Field(default=None, max_length=100)
    postal_code: str = Field(min_length=1, max_length=32)
    country_code: str = Field(pattern=r"^[A-Z]{2}$")


class PaymentIdentity(SchemaModel):
    """A tokenized payment identity; raw account numbers must never be stored here."""

    id: PaymentIdentityId
    method: PaymentMethod
    fingerprint: str = Field(min_length=8, max_length=256)
    first_seen_at: AwareDatetime
    last_seen_at: AwareDatetime
    issuer_country_code: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")

    @model_validator(mode="after")
    def validate_seen_window(self) -> "PaymentIdentity":
        if self.last_seen_at < self.first_seen_at:
            raise ValueError("last_seen_at cannot be earlier than first_seen_at")
        return self


class Promotion(SchemaModel):
    """A promotion definition that can be referenced by redemption events/orders."""

    id: PromotionId
    code: str = Field(min_length=1, max_length=64, pattern=r"^[A-Z0-9_-]+$")
    campaign_name: str = Field(min_length=1, max_length=200)
    discount_type: PromotionDiscountType
    discount_value: int = Field(gt=0)
    valid_from: AwareDatetime
    valid_until: AwareDatetime
    max_redemptions_per_account: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_promotion(self) -> "Promotion":
        if self.valid_until <= self.valid_from:
            raise ValueError("valid_until must be after valid_from")
        if (
            self.discount_type is PromotionDiscountType.PERCENTAGE_BPS
            and self.discount_value > 10_000
        ):
            raise ValueError(
                "percentage discount_value cannot exceed 10,000 basis points"
            )
        return self


class Order(SchemaModel):
    """A purchase and its required account/shipping relationships."""

    id: OrderId
    account_id: AccountId
    shipping_address_id: AddressId
    placed_at: AwareDatetime
    subtotal_paise: int = Field(ge=0)
    discount_paise: int = Field(ge=0)
    total_paise: int = Field(ge=0)
    item_count: int = Field(gt=0)
    status: OrderStatus = OrderStatus.PLACED
    promotion_id: PromotionId | None = None

    @model_validator(mode="after")
    def validate_amounts(self) -> "Order":
        if self.discount_paise > self.subtotal_paise:
            raise ValueError("discount_paise cannot exceed subtotal_paise")
        if self.total_paise != self.subtotal_paise - self.discount_paise:
            raise ValueError(
                "total_paise must equal subtotal_paise minus discount_paise"
            )
        return self


class Refund(SchemaModel):
    """A refund request attached to one order."""

    id: RefundId
    order_id: OrderId
    amount_paise: int = Field(gt=0)
    requested_at: AwareDatetime
    state: RefundState = RefundState.REQUESTED
    resolved_at: AwareDatetime | None = None
    reason_code: str = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def validate_resolution(self) -> "Refund":
        terminal = {RefundState.REJECTED, RefundState.COMPLETED, RefundState.CANCELLED}
        if self.resolved_at is not None and self.resolved_at < self.requested_at:
            raise ValueError("resolved_at cannot be earlier than requested_at")
        if self.state in terminal and self.resolved_at is None:
            raise ValueError("terminal refund states require resolved_at")
        if (
            self.state in {RefundState.REQUESTED, RefundState.APPROVED}
            and self.resolved_at
        ):
            raise ValueError("unresolved refund states cannot have resolved_at")
        return self
