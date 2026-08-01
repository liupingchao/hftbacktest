"""Symbol profiles for Binance-lead / Hyperliquid-lag public research.

The registry is intentionally public-data scoped. Live order sizing, tick/lot
precision, and risk envelopes still require separate task acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolProfile:
    profile_id: str
    binance_symbol: str
    hyperliquid_coin: str
    target_symbol: str
    aliases: tuple[str, ...]

    @property
    def basis_contract_caveat(self) -> str:
        return (
            "diagnostic_only_binance_usdm_futures_"
            f"{self.binance_symbol}_vs_hyperliquid_{self.hyperliquid_coin}_contract_basis"
        )


_PROFILES: dict[str, SymbolProfile] = {
    "btc": SymbolProfile(
        profile_id="btc",
        binance_symbol="BTCUSDT",
        hyperliquid_coin="BTC",
        target_symbol="BTC",
        aliases=("BTC", "BTCUSDT", "BTCUSD", "BTC-USD", "BTC/USDT", "BTC/USDC"),
    ),
    "eth": SymbolProfile(
        profile_id="eth",
        binance_symbol="ETHUSDT",
        hyperliquid_coin="ETH",
        target_symbol="ETH",
        aliases=("ETH", "ETHUSDT", "ETHUSD", "ETH-USD", "ETH/USDT", "ETH/USDC"),
    ),
    "mu": SymbolProfile(
        profile_id="mu",
        binance_symbol="MUUSDT",
        hyperliquid_coin="xyz:MU",
        target_symbol="XYZ:MU",
        aliases=(
            "MU",
            "xyz:MU",
            "XYZ:MU",
            "XYZMU",
            "MUUSDT",
            "MUUSD",
            "MUUSDC",
            "MU-USDC",
            "MU-USD",
            "MU/USDT",
            "MU/USDC",
        ),
    ),
    "skhynix": SymbolProfile(
        profile_id="skhynix",
        binance_symbol="SKHYNIXUSDT",
        hyperliquid_coin="xyz:SKHX",
        target_symbol="XYZ:SKHX",
        aliases=(
            "SKHYNIX",
            "xyz:SKHYNIX",
            "SKHX",
            "xyz:SKHX",
            "XYZ:SKHX",
            "XYZSKHX",
            "SKHYNIXUSDT",
            "SKHYNIXUSD",
            "SKHYNIXUSDC",
            "SKHYNIX-USDC",
            "SKHYNIX-USD",
            "SKHYNIX/USDT",
            "SKHYNIX/USDC",
        ),
    ),
}


def available_profile_ids() -> tuple[str, ...]:
    return tuple(sorted(_PROFILES))


def get_symbol_profile(profile_id: str | None = None) -> SymbolProfile:
    resolved = (profile_id or "btc").strip().lower()
    try:
        return _PROFILES[resolved]
    except KeyError as exc:
        supported = ", ".join(available_profile_ids())
        raise ValueError(f"unsupported symbol profile {profile_id!r}; supported profiles: {supported}") from exc


def _canonical_key(value: object) -> str:
    return str(value or "").upper().replace("-", "").replace("/", "").replace("_", "")


def normalize_signal_symbol(value: object) -> str:
    symbol = str(value or "").upper()
    key = _canonical_key(symbol)
    if not key:
        return ""
    for profile in _PROFILES.values():
        alias_keys = {_canonical_key(alias) for alias in profile.aliases}
        if key in alias_keys:
            return profile.target_symbol
    return symbol


def basis_contract_caveat(*, binance_symbol: str, hyperliquid_coin: str) -> str:
    return (
        "diagnostic_only_binance_usdm_futures_"
        f"{binance_symbol.upper()}_vs_hyperliquid_{hyperliquid_coin}_contract_basis"
    )
