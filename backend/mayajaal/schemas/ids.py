"""Nominal UUID identifiers used by Mayajaal's graph entities."""

from typing import NewType
from uuid import UUID

# NewType keeps IDs distinct for type checkers while Pydantic validates their UUID
# representation at the application boundary.
AccountId = NewType("AccountId", UUID)
DeviceId = NewType("DeviceId", UUID)
IPAddressId = NewType("IPAddressId", UUID)
AddressId = NewType("AddressId", UUID)
PaymentIdentityId = NewType("PaymentIdentityId", UUID)
OrderId = NewType("OrderId", UUID)
PromotionId = NewType("PromotionId", UUID)
RefundId = NewType("RefundId", UUID)
EventId = NewType("EventId", UUID)
