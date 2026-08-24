"""Pricing registry (doc §7.5 / §9.4): loads the vendored snapshot plus optional
user overrides, and resolves a (provider, model) pair to its rate card.

v0.2 schema additions (every one optional and backward-compatible):

- ``cache_read_per_token`` / ``cache_write_per_token``: absolute $/token rates
  for cached input, for providers that publish cached-input as its own price
  rather than a multiplier (Google, xAI, and OpenAI's per-model cached rates
  that are not a uniform 0.1x). When present they win over the multiplier.
- ``cache_write_mult_1h``: Anthropic bills 1-hour-TTL cache writes at 2x base
  input (vs 1.25x for the default 5-minute TTL). Used only when the trace
  itself carries a TTL breakdown (Claude Code session logs do; the OTel
  gen_ai.* attributes don't) — never assumed.
- ``tiers``: context-length-tiered rates (Gemini and Grok reprice the whole
  request once the prompt crosses a token threshold). Each tier is
  ``{above_input_tokens: N, ...rate fields...}``; the engine selects the last
  tier whose threshold the call's prompt-side tokens reach, else the base
  fields. A tier may set ``unpriced: true`` when the provider's rate for that
  range isn't published — those calls are left unpriced, never guessed.
- ``rates``: dated rate windows (``{from: "YYYY-MM-DD", until: "YYYY-MM-DD",
  ...rate fields...}``) for prices with a published effective period (e.g.
  Gemini 3.7 Flash promo pricing through 2026-12-31). The engine picks the
  window containing the call's own timestamp; a call outside every window is
  left unpriced with a clear reason.

Model-id resolution: lookup is exact-match first, then a conservative
normalization pass — a trailing date suffix (``-20250929`` / ``-2024-08-06``)
or ``-latest`` tag resolves to its base entry *within the same provider*.
Cross-provider aliasing (Bedrock/Vertex ids at first-party rates) is
deliberately NOT done: partner pricing genuinely differs, and guessing it
would fabricate a number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelPrice:
    input: float
    output: float
    cache_read_mult: float
    cache_write_mult: float
    min_cacheable_prefix_tokens: int
    cache_read_per_token: float | None = None
    cache_write_per_token: float | None = None
    cache_write_mult_1h: float | None = None
    # True for a published-but-rateless range (e.g. OpenAI >272K context):
    # the call is left unpriced rather than priced at the wrong tier.
    unpriced: bool = False

    def cache_read_rate(self) -> float:
        """Effective $/token for cached-input reads."""
        if self.cache_read_per_token is not None:
            return self.cache_read_per_token
        return self.input * self.cache_read_mult

    def cache_write_rate(self) -> float:
        """Effective $/token for cache writes (default/5-minute TTL)."""
        if self.cache_write_per_token is not None:
            return self.cache_write_per_token
        return self.input * self.cache_write_mult


class UnknownModelError(LookupError):
    """No registry entry for (provider, model); the caller must warn, not guess."""


class PricingConfigError(Exception):
    """A pricing.yaml (vendored or --pricing override) entry exists but is
    malformed — wrong type, missing required fields, or an invalid shape.
    Maps to exit code 2 (config/usage error), not the ingestion-error path --
    pricing data isn't trace content."""


_RATE_FIELDS = {
    "input",
    "output",
    "cache_read_mult",
    "cache_write_mult",
    "cache_write_mult_1h",
    "cache_read_per_token",
    "cache_write_per_token",
    "min_cacheable_prefix_tokens",
}
_DATE_SUFFIX_COMPACT = re.compile(r"-\d{8}$")
_DATE_SUFFIX_DASHED = re.compile(r"-\d{4}-\d{2}-\d{2}$")

_UNPRICED_CARD = ModelPrice(
    input=0.0,
    output=0.0,
    cache_read_mult=1.0,
    cache_write_mult=1.0,
    min_cacheable_prefix_tokens=0,
    unpriced=True,
)


def _require_number(value: Any, path: str) -> float:
    # bool is an int subclass; a YAML `true` here is a config mistake, not a rate.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PricingConfigError(
            f"pricing field {path} must be a number, got {type(value).__name__} ({value!r}) — "
            'a quoted value like "3.0e-6" parses as a string; remove the quotes'
        )
    return float(value)


def _validate_rate_fields(entry: dict[str, Any], path: str) -> None:
    for field in _RATE_FIELDS & entry.keys():
        if entry[field] is None:
            # An explicit `field: null` would pass silently here and then
            # crash the engine with float(None)/int(None) at price time —
            # a raw TypeError at the wrong exit code. There is no "unset a
            # merged field" semantic; omit the field instead.
            raise PricingConfigError(
                f"pricing field {path}.{field} is null — remove the field entirely "
                "instead of setting it to null (overrides merge field-level; an "
                "omitted field keeps the vendored value)"
            )
        if field == "min_cacheable_prefix_tokens":
            if isinstance(entry[field], bool) or not isinstance(entry[field], int):
                raise PricingConfigError(
                    f"pricing field {path}.{field} must be an integer, got {entry[field]!r}"
                )
        else:
            _require_number(entry[field], f"{path}.{field}")


def _parse_date(value: Any, path: str) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise PricingConfigError(f"{path} must be a YYYY-MM-DD date, got {value!r}") from exc
    raise PricingConfigError(f"{path} must be a YYYY-MM-DD date, got {value!r}")


def _validate_model_entry(entry: Any, path: str) -> None:
    if not isinstance(entry, dict):
        raise PricingConfigError(
            f"pricing entry {path} must be a mapping of rate fields, got "
            f"{type(entry).__name__} ({entry!r})"
        )
    _validate_rate_fields(entry, path)
    tiers = entry.get("tiers")
    if tiers is not None:
        if not isinstance(tiers, list):
            raise PricingConfigError(f"{path}.tiers must be a list")
        for i, tier in enumerate(tiers):
            if not isinstance(tier, dict):
                raise PricingConfigError(f"{path}.tiers[{i}] must be a mapping")
            if "above_input_tokens" not in tier:
                raise PricingConfigError(f"{path}.tiers[{i}] is missing above_input_tokens")
            if isinstance(tier["above_input_tokens"], bool) or not isinstance(
                tier["above_input_tokens"], int
            ):
                raise PricingConfigError(f"{path}.tiers[{i}].above_input_tokens must be an integer")
            _validate_rate_fields(tier, f"{path}.tiers[{i}]")
    rates = entry.get("rates")
    if rates is not None:
        if not isinstance(rates, list) or not rates:
            raise PricingConfigError(f"{path}.rates must be a non-empty list")
        for i, window in enumerate(rates):
            if not isinstance(window, dict):
                raise PricingConfigError(f"{path}.rates[{i}] must be a mapping")
            for bound in ("from", "until"):
                if bound in window:
                    _parse_date(window[bound], f"{path}.rates[{i}].{bound}")
            _validate_rate_fields(window, f"{path}.rates[{i}]")
    # Required base rates: on the entry itself, or inside every dated window.
    holders = rates if rates is not None else [entry]
    for i, holder in enumerate(holders):
        where = f"{path}.rates[{i}]" if rates is not None else path
        missing = [key for key in ("input", "output") if key not in holder]
        if missing and not holder.get("unpriced"):
            raise PricingConfigError(
                f"pricing entry {where} is missing required field(s): {', '.join(missing)}"
            )


class PricingRegistry:
    def __init__(self, providers: dict[str, dict[str, Any]], version: str) -> None:
        self._providers = providers
        self.version = version

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        overrides_path: str | Path | None = None,
    ) -> PricingRegistry:
        raw = cls._load_yaml(path) if path is not None else cls._load_vendored()
        providers = cls._validated_providers(raw, source="pricing snapshot")
        version: str = raw.get("version", "unknown")

        if overrides_path is not None:
            override_raw = cls._load_yaml(overrides_path)
            override_providers = cls._validated_providers(
                override_raw, source=str(overrides_path), partial=True
            )
            for provider, models in override_providers.items():
                merged_models = providers.setdefault(provider, {})
                for model_id, override_entry in models.items():
                    # Field-level merge: overriding just input/output must not
                    # silently reset the vendored cache multipliers to 1.0 (a
                    # 10x cache-read overcharge). Note tiers/rates lists
                    # replace wholesale — set them fully or not at all.
                    base_entry = merged_models.get(model_id, {})
                    merged = {**base_entry, **override_entry}
                    if (
                        "rates" in merged
                        and "rates" not in override_entry
                        and _RATE_FIELDS & override_entry.keys()
                    ):
                        # The vendored entry prices through dated rate
                        # windows, and window fields always win over
                        # entry-level scalars — a top-level scalar override
                        # here would load cleanly and then do nothing, which
                        # is worse than an error.
                        raise PricingConfigError(
                            f"pricing override for {provider}/{model_id} sets rate fields "
                            "on a model priced through dated `rates` windows — the "
                            "windows' own values always win. Override the full `rates` "
                            "list instead."
                        )
                    _validate_model_entry(merged, f"{provider}/{model_id}")
                    merged_models[model_id] = merged
            version = override_raw.get("version", version)

        return cls(providers, version)

    @staticmethod
    def _validated_providers(
        raw: dict[str, Any], source: str, partial: bool = False
    ) -> dict[str, dict[str, Any]]:
        providers = raw.get("providers", {})
        if not isinstance(providers, dict):
            raise PricingConfigError(f"{source}: 'providers' must be a mapping")
        for provider, models in providers.items():
            if not isinstance(models, dict):
                raise PricingConfigError(
                    f"{source}: provider {provider!r} must map model ids to rate entries, "
                    f"got {type(models).__name__} ({models!r})"
                )
            for model_id, entry in models.items():
                if partial:
                    # Override entries may be partial (they merge over the
                    # vendored entry); only validate the field types here.
                    if not isinstance(entry, dict):
                        raise PricingConfigError(
                            f"{source}: pricing entry {provider}/{model_id} must be a mapping"
                        )
                    _validate_rate_fields(entry, f"{provider}/{model_id}")
                else:
                    _validate_model_entry(entry, f"{provider}/{model_id}")
        return providers

    @staticmethod
    def _load_vendored() -> dict[str, Any]:
        vendored = resources.files("wattage.pricing.data").joinpath("pricing.yaml")
        loaded = yaml.safe_load(vendored.read_text(encoding="utf-8"))
        return dict(loaded) if loaded else {}

    @staticmethod
    def _load_yaml(path: str | Path) -> dict[str, Any]:
        with open(path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        return dict(loaded) if loaded else {}

    def _entry_for(self, provider: str, model: str) -> tuple[dict[str, Any], str]:
        provider_models = self._providers.get(provider) or self._providers.get(provider.lower())
        if provider_models is None:
            raise UnknownModelError(f"No pricing entry for {provider}/{model}")
        resolved = _resolve_model_id(provider_models, model)
        if resolved is None:
            raise UnknownModelError(f"No pricing entry for {provider}/{model}")
        return provider_models[resolved], resolved

    def get(
        self,
        provider: str,
        model: str,
        prompt_tokens: int | None = None,
        at: datetime | None = None,
    ) -> ModelPrice:
        """Resolve the rate card for one call.

        With no prompt_tokens/at, returns the base (lowest-tier, current-date)
        card — what detectors use for waste math. The engine passes both so
        tiered and dated entries price each call exactly.
        """
        entry, _ = self._entry_for(provider, model)
        return self._card_from_entry(entry, provider, model, prompt_tokens, at)

    def resolve(
        self,
        provider: str,
        model: str,
        prompt_tokens: int | None = None,
        at: datetime | None = None,
    ) -> tuple[ModelPrice, str]:
        """Like get(), but also returns the registry model id that priced the
        call (differs from `model` when a date-suffixed/-latest id resolved
        to its base entry)."""
        entry, resolved_id = self._entry_for(provider, model)
        return self._card_from_entry(entry, provider, model, prompt_tokens, at), resolved_id

    def _card_from_entry(
        self,
        entry: dict[str, Any],
        provider: str,
        model: str,
        prompt_tokens: int | None,
        at: datetime | None,
    ) -> ModelPrice:
        fields = self._select_rate_window(entry, provider, model, at)
        # Tiers may live on the entry or inside the selected window.
        tiers = fields.get("tiers", entry.get("tiers"))
        if prompt_tokens is not None and isinstance(tiers, list):
            selected: dict[str, Any] | None = None
            for tier in sorted(tiers, key=lambda t: t["above_input_tokens"]):
                if prompt_tokens >= tier["above_input_tokens"]:
                    selected = tier
            if selected is not None:
                fields = {**fields, **{k: v for k, v in selected.items() if k in _RATE_FIELDS}}
                if selected.get("unpriced"):
                    return _UNPRICED_CARD
        if fields.get("unpriced"):
            return _UNPRICED_CARD
        return ModelPrice(
            input=_require_number(fields["input"], f"{provider}/{model}.input"),
            output=_require_number(fields["output"], f"{provider}/{model}.output"),
            cache_read_mult=float(fields.get("cache_read_mult", 1.0)),
            cache_write_mult=float(fields.get("cache_write_mult", 1.0)),
            min_cacheable_prefix_tokens=int(fields.get("min_cacheable_prefix_tokens", 0)),
            cache_read_per_token=(
                float(fields["cache_read_per_token"])
                if fields.get("cache_read_per_token") is not None
                else None
            ),
            cache_write_per_token=(
                float(fields["cache_write_per_token"])
                if fields.get("cache_write_per_token") is not None
                else None
            ),
            cache_write_mult_1h=(
                float(fields["cache_write_mult_1h"])
                if fields.get("cache_write_mult_1h") is not None
                else None
            ),
        )

    @staticmethod
    def _select_rate_window(
        entry: dict[str, Any], provider: str, model: str, at: datetime | None
    ) -> dict[str, Any]:
        rates = entry.get("rates")
        if not isinstance(rates, list):
            return entry
        # A call with no usable timestamp prices at the window covering
        # "now" — the trace being analyzed was necessarily produced by then.
        when = (at or datetime.now(timezone.utc)).date()
        for window in rates:
            start = _parse_date(window["from"], "rates.from") if "from" in window else date.min
            end = _parse_date(window["until"], "rates.until") if "until" in window else date.max
            if start <= when <= end:
                merged = {k: v for k, v in entry.items() if k not in ("rates",)}
                merged.update({k: v for k, v in window.items() if k not in ("from", "until")})
                return merged
        raise UnknownModelError(
            f"No pricing entry for {provider}/{model} effective on {when.isoformat()} "
            "(the entry's dated rate windows don't cover this call's timestamp)"
        )


def _resolve_model_id(models: dict[str, Any], model: str) -> str | None:
    if model in models:
        return model
    # Compact -YYYYMMDD suffixes (Anthropic's convention) resolve to the
    # base entry: Anthropic documents the dateless id as an alias of the
    # dated snapshot, so both bill the same rate card. Dashed -YYYY-MM-DD
    # suffixes (OpenAI's convention) deliberately do NOT resolve: OpenAI's
    # dated SKUs are genuinely different price points (gpt-4o-2024-05-13 is
    # $5/$15 with no caching, while gpt-4o is $2.50/$10) — resolving them to
    # the base entry would silently bill the wrong rate, so they stay
    # unpriced until the user supplies an override.
    stripped = _DATE_SUFFIX_COMPACT.sub("", model)
    if stripped != model and stripped in models:
        return stripped
    if model.endswith("-latest"):
        base = model[: -len("-latest")]
        if base in models:
            return base
    return None
