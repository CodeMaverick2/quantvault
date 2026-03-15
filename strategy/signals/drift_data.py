"""Drift Data API client for funding rates, candles, and market data."""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import aiohttp
import pandas as pd

logger = logging.getLogger(__name__)

DRIFT_DATA_API = "https://data.api.drift.trade"
MAX_RECORDS_PER_REQUEST = 750


@dataclass
class FundingRateRecord:
    ts: int
    funding_rate: float          # per-period rate (8h-equivalent normalized)
    funding_rate_long: float
    funding_rate_short: float
    cumulative_long: float
    cumulative_short: float
    mark_twap: float
    oracle_twap: float
    period_revenue: float

    @property
    def hourly_rate(self) -> float:
        """Convert 8h funding rate to hourly equivalent."""
        return self.funding_rate / 8.0

    @property
    def apr(self) -> float:
        """Annualized funding rate as APR."""
        return self.hourly_rate * 24 * 365.25

    @property
    def basis_pct(self) -> float:
        """Basis: (mark - oracle) / oracle."""
        if self.oracle_twap == 0:
            return 0.0
        return (self.mark_twap - self.oracle_twap) / self.oracle_twap


@dataclass
class CandleRecord:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    fill_open: float
    fill_close: float


class DriftDataClient:
    """Async client for the Drift Data API."""

    def __init__(self, base_url: str = DRIFT_DATA_API, timeout_secs: int = 30):
        self._base = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout_secs)
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "DriftDataClient":
        self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self

    async def __aexit__(self, *args) -> None:
        if self._session:
            await self._session.close()

    async def _get(self, path: str, params: dict | None = None) -> dict | list:
        if self._session is None:
            raise RuntimeError("Use as async context manager")
        url = f"{self._base}{path}"
        async with self._session.get(url, params=params) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def get_funding_rates(
        self,
        symbol: str,
        limit: int = MAX_RECORDS_PER_REQUEST,
    ) -> list[FundingRateRecord]:
        """Fetch latest funding rate records for a market."""
        data = await self._get(f"/market/{symbol}/fundingRates", {"limit": limit})
        records = data if isinstance(data, list) else data.get("fundingRates", [])
        return [self._parse_funding(r) for r in records]

    async def get_funding_rates_date(
        self,
        symbol: str,
        year: int,
        month: int,
        day: int,
    ) -> list[FundingRateRecord]:
        """Fetch funding rates for a specific date."""
        data = await self._get(f"/market/{symbol}/fundingRates/{year}/{month}/{day}")
        records = data if isinstance(data, list) else data.get("fundingRates", [])
        return [self._parse_funding(r) for r in records]

    async def get_candles(
        self,
        symbol: str,
        resolution: str = "60",  # 1,5,15,60,240,D,W
        limit: int = 500,
    ) -> list[CandleRecord]:
        """Fetch OHLCV candles."""
        data = await self._get(
            f"/market/{symbol}/candles/{resolution}",
            {"limit": limit},
        )
        records = data if isinstance(data, list) else data.get("candles", [])
        return [self._parse_candle(r) for r in records]

    async def get_open_interest(self, symbol: str) -> pd.DataFrame:
        """Fetch open interest history."""
        data = await self._get(f"/amm/openInterest", {"market": symbol})
        records = data if isinstance(data, list) else data.get("openInterest", [])
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
        return df.sort_values("ts")

    async def get_oracle_price(self, symbol: str, limit: int = 200) -> pd.DataFrame:
        """Fetch oracle price snapshots."""
        data = await self._get("/amm/oraclePrice", {"market": symbol, "limit": limit})
        records = data if isinstance(data, list) else data.get("prices", [])
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
        return df.sort_values("ts")

    async def get_multi_market_funding(
        self,
        symbols: list[str],
        limit: int = 200,
    ) -> dict[str, pd.DataFrame]:
        """Fetch funding rates for multiple markets concurrently."""
        tasks = {sym: self.get_funding_rates(sym, limit) for sym in symbols}
        results = {}
        for sym, coro in tasks.items():
            try:
                records = await coro
                df = self._records_to_df(records)
                results[sym] = df
            except Exception as exc:
                logger.warning("Failed to fetch funding for %s: %s", sym, exc)
                results[sym] = pd.DataFrame()
        return results

    # ------------------------------------------------------------------ #
    # Parsing helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_funding(raw: dict) -> FundingRateRecord:
        return FundingRateRecord(
            ts=int(raw.get("ts", 0)),
            funding_rate=float(raw.get("fundingRate", 0)),
            funding_rate_long=float(raw.get("fundingRateLong", 0)),
            funding_rate_short=float(raw.get("fundingRateShort", 0)),
            cumulative_long=float(raw.get("cumulativeFundingRateLong", 0)),
            cumulative_short=float(raw.get("cumulativeFundingRateShort", 0)),
            mark_twap=float(raw.get("markTwap", 0)),
            oracle_twap=float(raw.get("oracleTwap", 0)),
            period_revenue=float(raw.get("periodRevenue", 0)),
        )

    @staticmethod
    def _parse_candle(raw: dict) -> CandleRecord:
        return CandleRecord(
            ts=int(raw.get("ts", raw.get("start", 0))),
            open=float(raw.get("open", 0)),
            high=float(raw.get("high", 0)),
            low=float(raw.get("low", 0)),
            close=float(raw.get("close", 0)),
            volume=float(raw.get("volume", 0)),
            fill_open=float(raw.get("fillOpen", raw.get("open", 0))),
            fill_close=float(raw.get("fillClose", raw.get("close", 0))),
        )

    @staticmethod
    def _records_to_df(records: list[FundingRateRecord]) -> pd.DataFrame:
        if not records:
            return pd.DataFrame()
        rows = [
            {
                "ts": r.ts,
                "datetime": datetime.fromtimestamp(r.ts, tz=timezone.utc),
                "funding_rate": r.funding_rate,
                "funding_rate_long": r.funding_rate_long,
                "funding_rate_short": r.funding_rate_short,
                "mark_twap": r.mark_twap,
                "oracle_twap": r.oracle_twap,
                "basis_pct": r.basis_pct,
                "hourly_rate": r.hourly_rate,
                "apr": r.apr,
                "period_revenue": r.period_revenue,
            }
            for r in records
        ]
        df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
        return df
