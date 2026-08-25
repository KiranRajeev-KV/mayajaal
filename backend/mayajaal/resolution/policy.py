"""Deterministic matching policy, intentionally independent of fraud labels."""

from collections import defaultdict
from collections.abc import Iterable, Sequence
from uuid import UUID

from rapidfuzz import fuzz, process

from mayajaal.schemas import Account, Address, Device, IPAddress, PaymentIdentity

from .candidates import NormalizedAddress, address_candidate_buckets, normalize_address
from .models import (
    ResolutionBundle,
    ResolutionEntityType,
    ResolutionMethod,
    ResolutionResult,
)
from .normalizers import (
    normalize_email,
    normalize_ip_address,
    normalize_phone,
    normalize_stable_identifier,
)

ADDRESS_FUZZY_THRESHOLD = 92.0
MAX_ADDRESS_FUZZY_CANDIDATES = 64
_GENERIC_ADDRESS_ANCHORS = frozenset({"apt", "fl", "near", "number", "rd", "st"})


def _address_anchors(address: NormalizedAddress) -> tuple[str, ...]:
    """Return deterministic discriminative tokens for bounded fuzzy lookup.

    Unit and street-number tokens retain recall for common presentation changes,
    while long street/locality words avoid comparing every address in a broad
    city/postal partition.
    """
    return tuple(
        sorted(
            {
                token
                for token in address.comparison_text.split()
                if token.isdecimal()
                or (len(token) >= 4 and token not in _GENERIC_ADDRESS_ANCHORS)
            }
        )
    )


def _exact_resolution(
    entity_type: ResolutionEntityType,
    values: Iterable[tuple[UUID, str, str]],
) -> list[ResolutionResult]:
    """Resolve an identifier domain by normalized equality with stable canonical IDs."""
    grouped: defaultdict[str, list[tuple[UUID, str]]] = defaultdict(list)
    for entity_id, raw_value, normalized_value in values:
        if normalized_value:
            grouped[normalized_value].append((entity_id, raw_value))

    results: list[ResolutionResult] = []
    for normalized_value in sorted(grouped):
        members = sorted(grouped[normalized_value], key=lambda member: str(member[0]))
        canonical_id, canonical_raw = members[0]
        for entity_id, raw_value in members:
            method = (
                ResolutionMethod.EXACT
                if raw_value == normalized_value
                else ResolutionMethod.NORMALIZED
            )
            evidence = (
                f"exact raw identifier={normalized_value}"
                if method is ResolutionMethod.EXACT
                else f"normalized identifier={normalized_value}"
            )
            if entity_id == canonical_id:
                evidence = f"canonical; {evidence}"
            elif raw_value == canonical_raw:
                evidence = f"same raw identifier as canonical; {evidence}"
            results.append(
                ResolutionResult(
                    entity_type=entity_type,
                    raw_entity_id=entity_id,
                    canonical_entity_id=canonical_id,
                    method=method,
                    score=100.0,
                    evidence=evidence,
                )
            )
    return results


def resolve_addresses(addresses: Sequence[Address]) -> list[ResolutionResult]:
    """Resolve addresses by normalized equality, then bounded RapidFuzz matching."""
    normalized = sorted(
        (normalize_address(address) for address in addresses),
        key=lambda a: str(a.entity_id),
    )
    buckets = address_candidate_buckets(normalized)
    results: list[ResolutionResult] = []

    for bucket_key in sorted(buckets):
        roots: list[NormalizedAddress] = []
        roots_by_id: dict[str, NormalizedAddress] = {}
        roots_by_exact: dict[
            tuple[str, str, str, str, str, str], NormalizedAddress
        ] = {}
        anchor_roots: defaultdict[str, list[NormalizedAddress]] = defaultdict(list)
        for address in buckets[bucket_key]:
            exact_root = roots_by_exact.get(address.exact_key)
            if exact_root is not None:
                results.append(
                    ResolutionResult(
                        entity_type=ResolutionEntityType.ADDRESS,
                        raw_entity_id=address.entity_id,
                        canonical_entity_id=exact_root.entity_id,
                        method=ResolutionMethod.NORMALIZED,
                        score=100.0,
                        evidence=(
                            "normalized address fields match canonical within "
                            f"country/city/postal bucket={bucket_key}"
                        ),
                    )
                )
                continue

            anchored_candidates = [
                (anchor, anchor_roots[anchor])
                for anchor in _address_anchors(address)
                if anchor_roots[anchor]
            ]
            candidate_anchor, candidate_roots = (
                min(
                    anchored_candidates,
                    key=lambda item: (len(item[1]), item[0]),
                )
                if anchored_candidates
                else (None, [])
            )
            choices = {
                str(root.entity_id): root.comparison_text
                for root in sorted(
                    candidate_roots, key=lambda root: str(root.entity_id)
                )[:MAX_ADDRESS_FUZZY_CANDIDATES]
            }
            matches = process.extract(
                address.comparison_text,
                choices,
                scorer=fuzz.ratio,
                score_cutoff=ADDRESS_FUZZY_THRESHOLD,
                limit=1,
            )
            if matches:
                _, score, root_id = matches[0]
                canonical = roots_by_id[root_id]
                results.append(
                    ResolutionResult(
                        entity_type=ResolutionEntityType.ADDRESS,
                        raw_entity_id=address.entity_id,
                        canonical_entity_id=canonical.entity_id,
                        method=ResolutionMethod.FUZZY,
                        score=float(score),
                        evidence=(
                            f"RapidFuzz ratio={score:.1f} >= {ADDRESS_FUZZY_THRESHOLD:.1f}; "
                            f"candidate bucket={bucket_key}; "
                            f"anchor={candidate_anchor}; candidates={len(choices)}"
                        ),
                    )
                )
                continue

            roots.append(address)
            roots_by_id[str(address.entity_id)] = address
            roots_by_exact[address.exact_key] = address
            for anchor in _address_anchors(address):
                anchor_roots[anchor].append(address)
            results.append(
                ResolutionResult(
                    entity_type=ResolutionEntityType.ADDRESS,
                    raw_entity_id=address.entity_id,
                    canonical_entity_id=address.entity_id,
                    method=ResolutionMethod.EXACT,
                    score=100.0,
                    evidence=f"canonical normalized address in bucket={bucket_key}",
                )
            )
    return results


def resolve_all(
    *,
    accounts: Sequence[Account] = (),
    addresses: Sequence[Address] = (),
    ip_addresses: Sequence[IPAddress] = (),
    payment_identities: Sequence[PaymentIdentity] = (),
    devices: Sequence[Device] = (),
) -> ResolutionBundle:
    """Resolve supplied raw entities without inspecting events or synthetic labels."""
    results = [*resolve_addresses(addresses)]
    results.extend(
        _exact_resolution(
            ResolutionEntityType.EMAIL,
            (
                (account.id, account.email, normalize_email(account.email))
                for account in accounts
                if account.email
            ),
        )
    )
    results.extend(
        _exact_resolution(
            ResolutionEntityType.PHONE,
            (
                (account.id, account.phone_e164, normalize_phone(account.phone_e164))
                for account in accounts
                if account.phone_e164
            ),
        )
    )
    results.extend(
        _exact_resolution(
            ResolutionEntityType.IP_ADDRESS,
            (
                (ip.id, str(ip.address), normalize_ip_address(ip.address))
                for ip in ip_addresses
            ),
        )
    )
    results.extend(
        _exact_resolution(
            ResolutionEntityType.PAYMENT_IDENTITY,
            (
                (
                    payment.id,
                    payment.fingerprint,
                    normalize_stable_identifier(payment.fingerprint),
                )
                for payment in payment_identities
            ),
        )
    )
    results.extend(
        _exact_resolution(
            ResolutionEntityType.DEVICE,
            (
                (
                    device.id,
                    device.fingerprint,
                    normalize_stable_identifier(device.fingerprint),
                )
                for device in devices
            ),
        )
    )
    return ResolutionBundle(
        results=tuple(
            sorted(
                results,
                key=lambda result: (result.entity_type, str(result.raw_entity_id)),
            )
        )
    )
