"""cTrader Open API gateway — extended from execution/hard pattern.

Adds: close-position, stop-loss/take-profit, limit orders, idempotent tracking.
Uses Spotware OpenApiPy + crochet sync bridge.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.config import CTraderSettings

_PROTO_SETUP = False


def _ensure_crochet() -> None:
    global _PROTO_SETUP
    if not _PROTO_SETUP:
        import crochet
        crochet.setup()
        _PROTO_SETUP = True


@dataclass
class PositionInfo:
    position_id: int
    symbol_id: int
    volume: int
    side: str
    entry_price: float
    swap: float = 0.0
    pnl: float = 0.0


@dataclass
class OrderInfo:
    order_id: int
    symbol_id: int
    side: str
    volume: int
    order_type: str
    status: str


class IdempotencyStore:
    """SQLite-backed idempotency tracking for order submissions."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._seen: dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            data = json.loads(self._path.read_text(encoding="utf-8"))
            now = time.monotonic()
            self._seen = {k: v for k, v in data.items() if now - v < 86400}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._seen), encoding="utf-8")

    def try_consume(self, key: str) -> bool:
        """Return True if key is new (consumed), False if duplicate."""
        if key in self._seen:
            return False
        self._seen[key] = time.monotonic()
        self._save()
        return True


class CTraderGateway:
    """Extended cTrader wrapper: auth, reconcile, market/limit orders, close, SL/TP."""

    def __init__(self, settings: CTraderSettings, *, idempotency_path: Path | None = None) -> None:
        self.settings = settings
        self._client: Any = None
        self._idempotency = IdempotencyStore(idempotency_path) if idempotency_path else None

    def is_configured(self) -> bool:
        return self.settings.can_authenticate()

    def _host_port(self) -> tuple[str, int]:
        from ctrader_open_api.endpoints import EndPoints
        if self.settings.demo:
            return EndPoints.PROTOBUF_DEMO_HOST, EndPoints.PROTOBUF_PORT
        return EndPoints.PROTOBUF_LIVE_HOST, EndPoints.PROTOBUF_PORT

    def connect(self) -> Any:
        """Start SSL client, application auth, account auth."""
        import crochet
        from twisted.internet import defer

        _ensure_crochet()

        if not self.is_configured():
            raise RuntimeError(
                "Set CTRADER_CLIENT_ID, CTRADER_CLIENT_SECRET, CTRADER_ACCESS_TOKEN, "
                "CTRADER_ACCOUNT_ID, CTRADER_SYMBOL_ID",
            )

        from ctrader_open_api.client import Client
        from ctrader_open_api.protobuf import Protobuf
        from ctrader_open_api.tcpProtocol import TcpProtocol

        host, port = self._host_port()
        client = Client(host, port, TcpProtocol)
        client.startService()
        gw = self

        @defer.inlineCallbacks
        def chain() -> Any:
            yield client.whenConnected(failAfterFailures=1)
            app = Protobuf.get(
                "ApplicationAuthReq",
                clientId=self.settings.client_id,
                clientSecret=self.settings.client_secret,
            )
            yield client.send(app)
            acc = Protobuf.get(
                "AccountAuthReq",
                ctidTraderAccountId=self.settings.account_id,
                accessToken=self.settings.access_token,
            )
            yield client.send(acc)
            gw._client = client
            return client

        @crochet.wait_for(timeout=60)
        def _connect() -> Any:
            return chain()

        return _connect()

    def disconnect(self) -> None:
        if self._client is not None:
            try:
                self._client.stopService()
            except Exception:
                pass
            self._client = None

    def reconcile(self) -> list[PositionInfo]:
        """Reconcile all open positions."""
        import crochet
        from ctrader_open_api.protobuf import Protobuf

        @crochet.wait_for(timeout=30)
        def _reconcile() -> Any:
            if self._client is None:
                raise RuntimeError("connect() first")
            req = Protobuf.get("ReconcileReq", ctidTraderAccountId=self.settings.account_id)
            return self._client.send(req)

        raw = _reconcile()
        return self._extract_positions(raw)

    def submit_market_order(
        self,
        *,
        side_buy: bool,
        volume: int,
        label: str = "agent",
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> Any:
        """Submit a market order with optional SL/TP."""
        import crochet
        from ctrader_open_api.messages.OpenApiModelMessages_pb2 import MARKET, BUY, SELL
        from ctrader_open_api.protobuf import Protobuf

        if self.settings.symbol_id is None:
            raise RuntimeError("CTRADER_SYMBOL_ID required")

        trade_side = BUY if side_buy else SELL
        kwargs: dict[str, Any] = {
            "ctidTraderAccountId": self.settings.account_id,
            "symbolId": self.settings.symbol_id,
            "orderType": MARKET,
            "tradeSide": trade_side,
            "volume": volume,
            "label": label,
        }

        if stop_loss is not None:
            kwargs["stopPrice"] = stop_loss
        if take_profit is not None:
            kwargs["takePrice"] = take_profit

        req = Protobuf.get("NewOrderReq", **kwargs)

        @crochet.wait_for(timeout=30)
        def _submit() -> Any:
            if self._client is None:
                raise RuntimeError("connect() first")
            return self._client.send(req)

        return _submit()

    def submit_limit_order(
        self,
        *,
        side_buy: bool,
        volume: int,
        price: float,
        label: str = "agent_limit",
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> Any:
        """Submit a limit order for grid levels."""
        import crochet
        from ctrader_open_api.messages.OpenApiModelMessages_pb2 import LIMIT, BUY, SELL
        from ctrader_open_api.protobuf import Protobuf

        if self.settings.symbol_id is None:
            raise RuntimeError("CTRADER_SYMBOL_ID required")

        trade_side = BUY if side_buy else SELL
        kwargs: dict[str, Any] = {
            "ctidTraderAccountId": self.settings.account_id,
            "symbolId": self.settings.symbol_id,
            "orderType": LIMIT,
            "tradeSide": trade_side,
            "volume": volume,
            "price": price,
            "label": label,
        }

        if stop_loss is not None:
            kwargs["stopPrice"] = stop_loss
        if take_profit is not None:
            kwargs["takePrice"] = take_profit

        req = Protobuf.get("NewOrderReq", **kwargs)

        @crochet.wait_for(timeout=30)
        def _submit() -> Any:
            if self._client is None:
                raise RuntimeError("connect() first")
            return self._client.send(req)

        return _submit()

    def close_position(self, position_id: int, *, volume: int | None = None) -> Any:
        """Close an open position (partially or fully)."""
        import crochet
        from ctrader_open_api.protobuf import Protobuf

        kwargs: dict[str, Any] = {
            "ctidTraderAccountId": self.settings.account_id,
            "positionId": position_id,
        }
        if volume is not None:
            kwargs["volume"] = volume

        req = Protobuf.get("ClosePositionReq", **kwargs)

        @crochet.wait_for(timeout=30)
        def _close() -> Any:
            if self._client is None:
                raise RuntimeError("connect() first")
            return self._client.send(req)

        return _close()

    def cancel_order(self, order_id: int) -> Any:
        """Cancel a pending limit order."""
        import crochet
        from ctrader_open_api.protobuf import Protobuf

        req = Protobuf.get(
            "CancelOrderReq",
            ctidTraderAccountId=self.settings.account_id,
            orderId=order_id,
        )

        @crochet.wait_for(timeout=30)
        def _cancel() -> Any:
            if self._client is None:
                raise RuntimeError("connect() first")
            return self._client.send(req)

        return _cancel()

    @staticmethod
    def _extract_positions(reconcile_message: Any) -> list[PositionInfo]:
        """Parse ProtoOAReconcileRes positions."""
        from ctrader_open_api.messages.OpenApiModelMessages_pb2 import BUY

        out: list[PositionInfo] = []
        if reconcile_message is None or not hasattr(reconcile_message, "position"):
            return out

        for p in reconcile_message.position:
            td = getattr(p, "tradeData", None)
            if td is None:
                continue
            out.append(PositionInfo(
                position_id=int(getattr(p, "id", 0)),
                symbol_id=int(getattr(td, "symbolId", 0)),
                volume=int(getattr(td, "volume", 0)),
                side="buy" if int(getattr(td, "tradeSide", 0)) == BUY else "sell",
                entry_price=float(getattr(td, "openPrice", 0)),
                swap=float(getattr(p, "swap", 0)),
                pnl=float(getattr(p, "price", 0)) - float(getattr(td, "openPrice", 0)),
            ))
        return out
