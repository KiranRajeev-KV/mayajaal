"""Pure, deterministic normalizers for raw identity attributes."""

import ipaddress
import re
import unicodedata

import phonenumbers
from email_validator import EmailNotValidError, validate_email

_WHITESPACE = re.compile(r"\s+")
_PUNCTUATION = re.compile(r"[^\w\s]")
_ADDRESS_TOKEN_REPLACEMENTS = {
    "apartment": "apt",
    "appartment": "apt",
    "flat": "apt",
    "floor": "fl",
    "road": "rd",
    "street": "st",
    "st": "st",
    "nagar": "nagar",
    "no": "number",
    "nr": "near",
}
_CITY_ALIASES = {
    "bangalore": "bengaluru",
    "bengaluru": "bengaluru",
    "bombay": "mumbai",
    "calcutta": "kolkata",
    "madras": "chennai",
}


def normalize_text(value: str) -> str:
    """Apply Unicode compatibility normalization and collapse whitespace."""
    compatible = unicodedata.normalize("NFKC", value).casefold()
    return _WHITESPACE.sub(" ", compatible).strip()


def normalize_email(value: str) -> str:
    """Normalize safe email presentation variance without provider-specific rewrites."""
    compact = _WHITESPACE.sub("", unicodedata.normalize("NFKC", value)).strip()
    try:
        # Deliverability is intentionally disabled: resolution is not a network check.
        return validate_email(compact, check_deliverability=False).normalized
    except EmailNotValidError:
        # Preserve deterministic resolution for imperfect historical raw data.
        return compact


def normalize_phone(value: str) -> str:
    """Return a digits-only international phone representation with a leading plus."""
    normalized = unicodedata.normalize("NFKC", value).strip()
    try:
        parsed = phonenumbers.parse(normalized, None)
        # `possible` is deliberate: Stage 2 formats dirty historical values and
        # does not reject numbers solely because a regional prefix is unassigned.
        if phonenumbers.is_possible_number(parsed):
            return phonenumbers.format_number(
                parsed, phonenumbers.PhoneNumberFormat.E164
            )
    except phonenumbers.NumberParseException:
        pass
    digits = "".join(character for character in normalized if character.isdecimal())
    return f"+{digits}" if digits else ""


def normalize_ip_address(value: object) -> str:
    """Render IPv4/IPv6 values in the standard compressed representation."""
    return str(ipaddress.ip_address(str(value).strip()))


def normalize_stable_identifier(value: str) -> str:
    """Normalize presentation variance in device and payment fingerprints."""
    return normalize_text(value).replace(" ", "")


def normalize_city(value: str) -> str:
    """Normalize city spelling aliases before address candidate partitioning."""
    normalized = normalize_text(value)
    return _CITY_ALIASES.get(normalized, normalized)


def normalize_address_component(value: str | None) -> str:
    """Normalize a single address line while retaining meaningful unit numbers."""
    if value is None:
        return ""
    text = normalize_text(value)
    # Periods commonly separate initials in road names (for example, "M.G. Road").
    text = text.replace(".", "")
    text = _PUNCTUATION.sub(" ", text)
    tokens = [_ADDRESS_TOKEN_REPLACEMENTS.get(token, token) for token in text.split()]
    return " ".join(tokens)


def normalize_postal_code(value: str) -> str:
    """Normalize postal-code punctuation and spacing without changing its value."""
    return "".join(
        character for character in normalize_text(value) if character.isalnum()
    )
