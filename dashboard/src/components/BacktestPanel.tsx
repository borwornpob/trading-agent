import { useEffect, useMemo, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  Play,
  Loader2,
  BarChart3,
  TrendingUp,
  AlertCircle,
  ArrowUpRight,
  ArrowDownRight,
  Activity,
  Target,
  ShieldAlert,
  Percent,
  Hash,
  Scale,
  ChevronUp,
  ChevronDown,
  Layers,
} from "lucide-react";
import {
  CandlestickSeries,
  ColorType,
  CrosshairMode,
  createChart,
  createSeriesMarkers,
  LineSeries,
  LineStyle,
  type IChartApi,
  type Time,
  type UTCTimestamp,
} from "lightweight-charts";
import { cn } from "@/lib/utils";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface Trade {
  timestamp: string;
  entry_timestamp: string;
  side: "buy" | "sell";
  direction: "long" | "short";
  entry_price: number;
  exit_price: number;
  pnl: number;
  pnl_pct: number;
  exit_reason: string;
  bars_held: number;
  sl_price: number;
  tp_price: number;
  grid_levels_filled?: number;
  avg_entry_price?: number;
  total_size?: number;
}

interface OhlcvBar {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
}

interface Signal {
  timestamp: string;
  pred_class: number;
  score: number;
  p_up: number;
  p_down: number;
}

interface BacktestResult {
  id: string;
  params: {
    start_date: string;
    end_date: string;
    bar_frequency: string;
    initial_cash: number;
    commission_rate: number;
    units: number;
    grid_enabled: boolean;
    sl_atr_mult: number;
    tp_atr_mult: number;
    sr_cap_tp: boolean;
    grid_max_levels: number;
    grid_sizing_decay: number;
    grid_atr_stop_mult: number;
  };
  metrics: Record<string, number>;
  n_trades: number;
  trades: Trade[];
  equity_curve: number[];
  equity_points?: EquityPoint[];
  ohlcv: OhlcvBar[];
  signals: Signal[];
}

interface EquityPoint {
  timestamp: string;
  equity: number;
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

function fmtDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString("en-US", { month: "short", day: "2-digit" });
}

function fmtDateShort(iso: string): string {
  const d = new Date(iso);
  return `${(d.getMonth() + 1).toString().padStart(2, "0")}/${d.getDate().toString().padStart(2, "0")}`;
}

function fmtPrice(v: number): string {
  return v.toFixed(2);
}

function fmtPct(v: number): string {
  return `${v >= 0 ? "+" : ""}${v.toFixed(1)}%`;
}

function fmtDollar(v: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(v);
}

function fmtCompact(v: number): string {
  if (v >= 1_000_000) return `$${(v / 1_000_000).toFixed(1)}M`;
  if (v >= 1_000) return `$${(v / 1_000).toFixed(1)}K`;
  return `$${v.toFixed(0)}`;
}

function toChartTime(iso: string): Time {
  return Math.floor(new Date(iso).getTime() / 1000) as UTCTimestamp;
}

function resizeChart(chart: IChartApi, el: HTMLDivElement) {
  chart.applyOptions({
    width: el.clientWidth,
    height: el.clientHeight,
  });
}

const chartBaseOptions = {
  autoSize: true,
  layout: {
    background: { type: ColorType.Solid, color: "transparent" },
    textColor: "#6b7280",
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
    fontSize: 11,
  },
  grid: {
    vertLines: { color: "rgba(120,113,108,0.08)" },
    horzLines: { color: "rgba(120,113,108,0.08)" },
  },
  crosshair: {
    mode: CrosshairMode.Magnet,
    vertLine: { color: "rgba(120,113,108,0.45)", labelBackgroundColor: "#1f1d1c" },
    horzLine: { color: "rgba(120,113,108,0.45)", labelBackgroundColor: "#1f1d1c" },
  },
  rightPriceScale: {
    borderColor: "rgba(120,113,108,0.16)",
    scaleMargins: { top: 0.12, bottom: 0.12 },
  },
  timeScale: {
    borderColor: "rgba(120,113,108,0.16)",
    timeVisible: true,
    secondsVisible: false,
  },
};

/* ------------------------------------------------------------------ */
/*  Tooltip Styles                                                     */
/* ------------------------------------------------------------------ */

const tooltipStyle: React.CSSProperties = {
  backgroundColor: "rgba(15, 15, 20, 0.92)",
  border: "1px solid rgba(212, 175, 55, 0.25)",
  borderRadius: "8px",
  fontSize: "12px",
  padding: "8px 12px",
  backdropFilter: "blur(8px)",
};

/* ------------------------------------------------------------------ */
/*  Sub-components                                                     */
/* ------------------------------------------------------------------ */

/** Single hero metric card */
function MetricCard({
  label,
  value,
  icon: Icon,
  color = "text-foreground",
  sub,
}: {
  label: string;
  value: string;
  icon: React.ElementType;
  color?: string;
  sub?: string;
}) {
  return (
    <div className="rounded-lg border border-border bg-surface-2 px-4 py-3.5 text-center">
      <div className="mb-1.5 flex items-center justify-center gap-1.5">
        <Icon className="size-3.5 text-muted-foreground" />
        <span className="text-[10px] font-bold uppercase tracking-[0.12em] text-muted-foreground">
          {label}
        </span>
      </div>
      <span
        className={cn(
          "block font-mono text-xl font-extrabold tabular-nums",
          color,
        )}
      >
        {value}
      </span>
      {sub && (
        <span className="mt-0.5 block text-[10px] font-semibold text-muted-foreground">
          {sub}
        </span>
      )}
    </div>
  );
}

function EquityCurveFinancialChart({
  points,
  initialCash,
}: {
  points: EquityPoint[];
  initialCash: number;
}) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el || points.length === 0) return;

    const chart = createChart(el, {
      ...chartBaseOptions,
      localization: {
        priceFormatter: (price: number) => fmtDollar(price),
      },
    });
    const equitySeries = chart.addSeries(LineSeries, {
      color: "#d4af37",
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: true,
      crosshairMarkerVisible: true,
    });

    equitySeries.setData(
      points
        .filter((p) => Number.isFinite(p.equity))
        .map((p) => ({
          time: toChartTime(p.timestamp),
          value: p.equity,
        })),
    );
    equitySeries.createPriceLine({
      price: initialCash,
      color: "rgba(212,175,55,0.45)",
      lineStyle: LineStyle.Dashed,
      lineWidth: 1,
      axisLabelVisible: true,
      title: "Start",
    });

    resizeChart(chart, el);
    chart.timeScale().fitContent();

    const observer = new ResizeObserver(() => resizeChart(chart, el));
    observer.observe(el);

    return () => {
      observer.disconnect();
      chart.remove();
    };
  }, [initialCash, points]);

  return <div ref={ref} className="h-full w-full" />;
}

function PriceTradesFinancialChart({
  bars,
  trades,
  lastTrade,
}: {
  bars: OhlcvBar[];
  trades: Trade[];
  lastTrade: Trade | null;
}) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el || bars.length === 0) return;

    const chart = createChart(el, {
      ...chartBaseOptions,
      localization: {
        priceFormatter: (price: number) => price.toFixed(2),
      },
    });
    const candles = chart.addSeries(CandlestickSeries, {
      upColor: "#5fbf75",
      downColor: "#de5c51",
      borderUpColor: "#5fbf75",
      borderDownColor: "#de5c51",
      wickUpColor: "rgba(95,191,117,0.75)",
      wickDownColor: "rgba(222,92,81,0.75)",
      priceLineVisible: false,
    });

    candles.setData(
      bars.map((bar) => ({
        time: toChartTime(bar.timestamp),
        open: bar.open,
        high: bar.high,
        low: bar.low,
        close: bar.close,
      })),
    );

    createSeriesMarkers(
      candles,
      trades.map((trade, i) => {
        const isLong = trade.direction === "long";
        return {
          id: `trade-${i}`,
          time: toChartTime(trade.entry_timestamp),
          position: "atPriceMiddle" as const,
          price: trade.entry_price,
          shape: isLong ? ("arrowUp" as const) : ("arrowDown" as const),
          color: isLong ? "#2faf5c" : "#d9473f",
          text: isLong ? "Long" : "Short",
          size: 1.25,
        };
      }),
      { zOrder: "top" },
    );

    if (lastTrade) {
      candles.createPriceLine({
        price: lastTrade.tp_price,
        color: "rgba(34,197,94,0.6)",
        lineStyle: LineStyle.Dashed,
        lineWidth: 1,
        axisLabelVisible: true,
        title: `TP ${fmtPrice(lastTrade.tp_price)}`,
      });
      candles.createPriceLine({
        price: lastTrade.sl_price,
        color: "rgba(239,68,68,0.6)",
        lineStyle: LineStyle.Dashed,
        lineWidth: 1,
        axisLabelVisible: true,
        title: `SL ${fmtPrice(lastTrade.sl_price)}`,
      });
    }

    resizeChart(chart, el);
    chart.timeScale().fitContent();

    const observer = new ResizeObserver(() => resizeChart(chart, el));
    observer.observe(el);

    return () => {
      observer.disconnect();
      chart.remove();
    };
  }, [bars, lastTrade, trades]);

  return <div ref={ref} className="h-full w-full" />;
}

/** Custom equity tooltip */
function EquityTooltip({
  active,
  payload,
  initialCash,
}: {
  active?: boolean;
  payload?: Array<{
    value: number;
    payload: { index: number; equity: number; date: string };
  }>;
  initialCash: number;
}) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  const pnl = d.equity - initialCash;
  const pnlPct = (pnl / initialCash) * 100;
  return (
    <div style={tooltipStyle}>
      <div className="mb-1 text-[10px] uppercase tracking-wider text-muted-foreground">
        {d.date}
      </div>
      <div className="font-mono text-sm font-bold text-foreground">
        {fmtDollar(d.equity)}
      </div>
      <div
        className={cn(
          "font-mono text-xs",
          pnl >= 0 ? "text-long" : "text-short",
        )}
      >
        {pnl >= 0 ? "+" : ""}
        {fmtDollar(pnl)} ({fmtPct(pnlPct)})
      </div>
    </div>
  );
}

/** Custom price tooltip */
function PriceTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ value: number; payload: Record<string, unknown> }>;
}) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload as Record<string, unknown>;
  return (
    <div style={tooltipStyle}>
      <div className="mb-1 text-[10px] uppercase tracking-wider text-muted-foreground">
        {d.date as string}
      </div>
      <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-xs font-mono">
        <span className="text-muted-foreground">O</span>
        <span className="text-foreground">{fmtPrice(d.open as number)}</span>
        <span className="text-muted-foreground">H</span>
        <span className="text-foreground">{fmtPrice(d.high as number)}</span>
        <span className="text-muted-foreground">L</span>
        <span className="text-foreground">{fmtPrice(d.low as number)}</span>
        <span className="text-muted-foreground">C</span>
        <span className="text-foreground">{fmtPrice(d.close as number)}</span>
      </div>
    </div>
  );
}

/** Custom trade scatter tooltip */
function TradeTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ value: number; payload: Record<string, unknown> }>;
}) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload as Record<string, unknown>;
  return (
    <div style={tooltipStyle}>
      <div className="mb-1 text-[10px] uppercase tracking-wider text-muted-foreground">
        {d.date as string}
      </div>
      <div className="text-xs font-mono">
        <div
          className={cn(
            "font-bold",
            d.side === "buy" ? "text-long" : "text-short",
          )}
        >
          {(d.side as string).toUpperCase()} Entry
        </div>
        <div className="text-foreground">{fmtPrice(d.price as number)}</div>
      </div>
    </div>
  );
}

/** Exit reason badge color helper */
function exitReasonColor(reason: string): string {
  switch (reason) {
    case "take_profit":
      return "long";
    case "stop_loss":
      return "short";
    case "signal_flip":
      return "volatile";
    case "end_of_data":
    default:
      return "secondary";
  }
}

function exitReasonLabel(reason: string): string {
  switch (reason) {
    case "take_profit":
      return "Take Profit";
    case "stop_loss":
      return "Stop Loss";
    case "signal_flip":
      return "Signal Flip";
    case "end_of_data":
      return "End of Data";
    default:
      return reason;
  }
}

/* ------------------------------------------------------------------ */
/*  Main Component                                                     */
/* ------------------------------------------------------------------ */

export default function BacktestPanel() {
  const [startDate, setStartDate] = useState("2024-01-01");
  const [endDate, setEndDate] = useState("2024-12-31");
  const [result, setResult] = useState<
    BacktestResult | { error: string } | null
  >(null);
  const [loading, setLoading] = useState(false);

  const isError = result && "error" in result;
  const data = !isError ? (result as BacktestResult | null) : null;

  /* ── Run backtest ──────────────────────────────────────────────── */
  const runBacktest = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/backtest/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ start_date: startDate, end_date: endDate }),
      });
      const json = await res.json();
      setResult(json);
    } catch (err) {
      setResult({ error: String(err) });
    } finally {
      setLoading(false);
    }
  };

  /* ── Derived data ──────────────────────────────────────────────── */

  // Equity curve chart data
  const equityCurve = data?.equity_curve ?? [];
  const ohlcvBars = data?.ohlcv ?? [];
  const tradeList = data?.trades ?? [];

  const equityPoints = useMemo(() => {
    if (!data || equityCurve.length === 0) return [];
    if (data.equity_points?.length) {
      return data.equity_points.filter((p) => Number.isFinite(p.equity));
    }

    const values =
      equityCurve.length === ohlcvBars.length + 1
        ? equityCurve.slice(1)
        : equityCurve;

    return values
      .map((equity, i) => ({
        timestamp: ohlcvBars[i]?.timestamp,
        equity,
      }))
      .filter((p): p is EquityPoint => Boolean(p.timestamp));
  }, [data, equityCurve, ohlcvBars]);

  // Most recent trade for SL/TP reference lines
  const lastTrade = useMemo(() => {
    if (!data || tradeList.length === 0) return null;
    return tradeList[tradeList.length - 1];
  }, [data, tradeList]);

  // Exit reason breakdown
  const exitBreakdown = useMemo(() => {
    if (!data) return [];
    const trades = data.trades ?? [];
    if (trades.length === 0) return [];
    const counts: Record<string, number> = {};
    for (const t of trades) {
      counts[t.exit_reason] = (counts[t.exit_reason] || 0) + 1;
    }
    return Object.entries(counts)
      .sort((a, b) => b[1] - a[1])
      .map(([reason, count]) => ({
        reason,
        label: exitReasonLabel(reason),
        count,
        pct: trades.length > 0 ? (count / trades.length) * 100 : 0,
        variant: exitReasonColor(reason),
      }));
  }, [data]);

  // Metrics helpers
  const metrics = data?.metrics ?? {};
  const totalReturnPct = metrics.total_return_pct ?? 0;
  const sharpe = metrics.sharpe ?? 0;
  const maxDrawdownPct = metrics.max_drawdown_pct ?? 0;
  const winRate = metrics.win_rate ?? 0;
  const numTrades = metrics.num_trades ?? data?.n_trades ?? 0;
  const profitFactor = metrics.profit_factor ?? 0;
  const initialCash = data?.params?.initial_cash ?? 100000;

  // Trade log totals
  const totals = useMemo(() => {
    if (!data) return { pnl: 0, pnlPct: 0 };
    const trades = data.trades ?? [];
    if (trades.length === 0) return { pnl: 0, pnlPct: 0 };
    return trades.reduce(
      (acc, t) => ({
        pnl: acc.pnl + t.pnl,
        pnlPct: acc.pnlPct + t.pnl_pct,
      }),
      { pnl: 0, pnlPct: 0 },
    );
  }, [data]);

  /* ── Render ────────────────────────────────────────────────────── */

  return (
    <Card className="flex h-full min-h-0 flex-col overflow-hidden">
      <CardHeader className="shrink-0 pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="size-3.5 text-primary" />
            Backtest
          </CardTitle>
          {data && (
            <div className="flex items-center gap-2">
              <Badge
                variant="secondary"
                className="gap-1 font-mono text-[10px]"
              >
                <Hash className="size-3" />
                {data.id}
              </Badge>
              <Badge
                variant={totalReturnPct >= 0 ? "long" : "short"}
                className="gap-1"
              >
                <TrendingUp className="size-3" />
                {fmtPct(totalReturnPct)}
              </Badge>
            </div>
          )}
        </div>
      </CardHeader>

      <CardContent className="flex min-h-0 flex-col gap-5 overflow-y-auto">
        {/* ── Date Range Inputs ──────────────────────────────────── */}
        <div className="grid gap-3 md:grid-cols-[1fr_1fr_auto] md:items-end">
          <div className="flex-1 space-y-1.5">
            <label className="text-[10px] font-bold uppercase tracking-[0.15em] text-muted-foreground">
              Start Date
            </label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className="w-full rounded-lg border border-border bg-card px-3 py-2 font-mono text-sm text-foreground tabular-nums transition-colors focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary/30"
            />
          </div>
          <div className="flex-1 space-y-1.5">
            <label className="text-[10px] font-bold uppercase tracking-[0.15em] text-muted-foreground">
              End Date
            </label>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              className="w-full rounded-lg border border-border bg-card px-3 py-2 font-mono text-sm text-foreground tabular-nums transition-colors focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary/30"
            />
          </div>
          <Button
            onClick={runBacktest}
            disabled={loading}
            variant={loading ? "secondary" : "default"}
            size="sm"
            className="gap-2 shrink-0"
          >
            {loading ? (
              <Loader2 className="size-3.5 animate-spin" />
            ) : (
              <Play className="size-3.5" />
            )}
            {loading ? "Running…" : "Run"}
          </Button>
        </div>

        {/* ── Error ──────────────────────────────────────────────── */}
        {isError && (
          <div className="flex items-center gap-2 rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3">
            <AlertCircle className="size-4 shrink-0 text-destructive" />
            <span className="text-xs font-semibold text-destructive">
              {(result as { error: string }).error}
            </span>
          </div>
        )}

        {/* ── Results ────────────────────────────────────────────── */}
        {data && data.equity_curve && data.ohlcv && data.trades ? (
          <>
            <Separator />

            {/* ── Summary Metrics Grid ──────────────────────────── */}
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
              <MetricCard
                label="Total Return"
                value={fmtPct(totalReturnPct)}
                icon={TrendingUp}
                color={totalReturnPct >= 0 ? "text-long" : "text-short"}
                sub={fmtDollar(
                  (data.equity_curve[data.equity_curve.length - 1] ??
                    initialCash) - initialCash,
                )}
              />
              <MetricCard
                label="Sharpe"
                value={sharpe.toFixed(2)}
                icon={Activity}
                color={
                  sharpe > 1
                    ? "text-long"
                    : sharpe > 0
                      ? "text-volatile"
                      : "text-short"
                }
              />
              <MetricCard
                label="Max Drawdown"
                value={`${maxDrawdownPct.toFixed(1)}%`}
                icon={ShieldAlert}
                color={
                  Math.abs(maxDrawdownPct) < 10
                    ? "text-long"
                    : Math.abs(maxDrawdownPct) < 20
                      ? "text-volatile"
                      : "text-short"
                }
              />
              <MetricCard
                label="Win Rate"
                value={`${(winRate * 100).toFixed(1)}%`}
                icon={Percent}
                color={winRate > 0.5 ? "text-long" : "text-volatile"}
              />
              <MetricCard
                label="Trades"
                value={String(numTrades)}
                icon={Hash}
              />
              <MetricCard
                label="Profit Factor"
                value={profitFactor.toFixed(2)}
                icon={Scale}
                color={
                  profitFactor > 1.5
                    ? "text-long"
                    : profitFactor > 1
                      ? "text-volatile"
                      : "text-short"
                }
              />
            </div>

            {/* ── Extra Metrics Row ─────────────────────────────── */}
            <div className="flex flex-wrap gap-2">
              {metrics.annualized_return !== undefined && (
                <div className="flex items-center gap-1.5 rounded-md bg-surface-2 px-3 py-1.5">
                  <span className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
                    Ann. Return
                  </span>
                  <span
                    className={cn(
                      "font-mono text-xs font-bold tabular-nums",
                      metrics.annualized_return >= 0
                        ? "text-long"
                        : "text-short",
                    )}
                  >
                    {fmtPct(metrics.annualized_return)}
                  </span>
                </div>
              )}
              {metrics.sortino !== undefined && (
                <div className="flex items-center gap-1.5 rounded-md bg-surface-2 px-3 py-1.5">
                  <span className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
                    Sortino
                  </span>
                  <span className="font-mono text-xs font-bold tabular-nums text-foreground">
                    {metrics.sortino.toFixed(2)}
                  </span>
                </div>
              )}
              {metrics.calmar !== undefined && (
                <div className="flex items-center gap-1.5 rounded-md bg-surface-2 px-3 py-1.5">
                  <span className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
                    Calmar
                  </span>
                  <span className="font-mono text-xs font-bold tabular-nums text-foreground">
                    {metrics.calmar.toFixed(2)}
                  </span>
                </div>
              )}
              {metrics.annualized_vol !== undefined && (
                <div className="flex items-center gap-1.5 rounded-md bg-surface-2 px-3 py-1.5">
                  <span className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
                    Ann. Vol
                  </span>
                  <span className="font-mono text-xs font-bold tabular-nums text-foreground">
                    {metrics.annualized_vol.toFixed(1)}%
                  </span>
                </div>
              )}
            </div>

            {/* ── Equity Curve Chart ────────────────────────────── */}
            <div className="space-y-2">
              <div className="flex items-center gap-2 px-1">
                <TrendingUp className="size-3.5 text-primary" />
                <span className="text-[11px] font-bold uppercase tracking-[0.15em] text-muted-foreground">
                  Equity Curve
                </span>
                <span className="ml-auto font-mono text-xs text-muted-foreground">
                  {fmtDollar(initialCash)} →{" "}
                  {fmtDollar(
                    equityPoints[equityPoints.length - 1]?.equity ??
                      data.equity_curve[data.equity_curve.length - 1] ??
                      initialCash,
                  )}
                </span>
              </div>
              <div className="h-64 overflow-hidden rounded-lg border border-border bg-surface-1/50 p-1">
                <EquityCurveFinancialChart
                  points={equityPoints}
                  initialCash={initialCash}
                />
              </div>
            </div>

            {/* ── Price Chart with Trade Markers ────────────────── */}
            <div className="space-y-2">
              <div className="flex items-center gap-2 px-1">
                <BarChart3 className="size-3.5 text-primary" />
                <span className="text-[11px] font-bold uppercase tracking-[0.15em] text-muted-foreground">
                  Price &amp; Trades
                </span>
                {lastTrade && (
                  <span className="ml-auto flex items-center gap-3 text-[10px] font-mono text-muted-foreground">
                    <span className="flex items-center gap-1">
                      <ChevronUp className="size-3 text-long" /> Long
                    </span>
                    <span className="flex items-center gap-1">
                      <ChevronDown className="size-3 text-short" /> Short
                    </span>
                  </span>
                )}
              </div>
              <div className="h-[26rem] overflow-hidden rounded-lg border border-border bg-surface-1/50 p-1">
                <PriceTradesFinancialChart
                  bars={ohlcvBars}
                  trades={tradeList}
                  lastTrade={lastTrade}
                />
              </div>
            </div>

            <Separator />

            {/* ── Trade Log Table ────────────────────────────────── */}
            <div className="space-y-2">
              <div className="flex items-center gap-2 px-1">
                <Activity className="size-3.5 text-primary" />
                <span className="text-[11px] font-bold uppercase tracking-[0.15em] text-muted-foreground">
                  Trade Log
                </span>
                <Badge variant="secondary" className="ml-auto gap-1">
                  {data.trades.length} trades
                </Badge>
              </div>
              <div className="max-h-80 overflow-auto rounded-lg border border-border">
                <table className="w-full text-left">
                  <thead className="sticky top-0 z-10 bg-surface-2">
                    <tr>
                      {[
                        "#",
                        "Entry Date",
                        "Direction",
                        "Entry",
                        "Exit",
                        "PnL",
                        "PnL %",
                        "Exit Reason",
                        "Grid",
                        "Bars",
                      ].map((h) => (
                        <th
                          key={h}
                          className="px-3 py-2 text-[10px] font-bold uppercase tracking-widest text-muted-foreground"
                        >
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border/50">
                    {data.trades.map((t, i) => (
                      <tr
                        key={i}
                        className={cn(
                          "transition-colors hover:bg-surface-2/50",
                          t.pnl >= 0 ? "text-long" : "text-short",
                        )}
                      >
                        <td className="px-3 py-1.5 font-mono text-xs tabular-nums text-muted-foreground">
                          {i + 1}
                        </td>
                        <td className="px-3 py-1.5 font-mono text-xs tabular-nums">
                          {fmtDateShort(t.entry_timestamp)}
                        </td>
                        <td className="px-3 py-1.5">
                          <Badge
                            variant={t.direction === "long" ? "long" : "short"}
                            className="gap-1 text-[9px]"
                          >
                            {t.direction === "long" ? (
                              <ArrowUpRight className="size-2.5" />
                            ) : (
                              <ArrowDownRight className="size-2.5" />
                            )}
                            {t.direction.toUpperCase()}
                          </Badge>
                        </td>
                        <td className="px-3 py-1.5 font-mono text-xs tabular-nums text-foreground">
                          {fmtPrice(t.entry_price)}
                        </td>
                        <td className="px-3 py-1.5 font-mono text-xs tabular-nums text-foreground">
                          {fmtPrice(t.exit_price)}
                        </td>
                        <td
                          className={cn(
                            "px-3 py-1.5 font-mono text-xs font-bold tabular-nums",
                            t.pnl >= 0 ? "text-long" : "text-short",
                          )}
                        >
                          {t.pnl >= 0 ? "+" : ""}
                          {fmtDollar(t.pnl)}
                        </td>
                        <td
                          className={cn(
                            "px-3 py-1.5 font-mono text-xs font-bold tabular-nums",
                            t.pnl_pct >= 0 ? "text-long" : "text-short",
                          )}
                        >
                          {t.pnl_pct >= 0 ? "+" : ""}
                          {(t.pnl_pct * 100).toFixed(2)}%
                        </td>
                        <td className="px-3 py-1.5">
                          <Badge
                            variant={
                              exitReasonColor(t.exit_reason) as
                                | "long"
                                | "short"
                                | "volatile"
                                | "secondary"
                            }
                            className="text-[9px]"
                          >
                            {exitReasonLabel(t.exit_reason)}
                          </Badge>
                        </td>
                        <td className="px-3 py-1.5">
                          {t.grid_levels_filled && t.grid_levels_filled > 1 ? (
                            <Badge variant="gold" className="gap-1 text-[9px]">
                              <Layers className="size-2.5" />
                              {t.grid_levels_filled}L
                            </Badge>
                          ) : (
                            <span className="font-mono text-[10px] text-muted-foreground">
                              1
                            </span>
                          )}
                        </td>
                        <td className="px-3 py-1.5 font-mono text-xs tabular-nums text-muted-foreground">
                          {t.bars_held}
                        </td>
                      </tr>
                    ))}
                    {/* Totals row */}
                    <tr className="border-t-2 border-border bg-surface-2 font-bold">
                      <td
                        colSpan={6}
                        className="px-3 py-2 text-[10px] uppercase tracking-widest text-muted-foreground"
                      >
                        Total
                      </td>
                      <td
                        className={cn(
                          "px-3 py-2 font-mono text-xs tabular-nums",
                          totals.pnl >= 0 ? "text-long" : "text-short",
                        )}
                      >
                        {totals.pnl >= 0 ? "+" : ""}
                        {fmtDollar(totals.pnl)}
                      </td>
                      <td
                        className={cn(
                          "px-3 py-2 font-mono text-xs tabular-nums",
                          totals.pnlPct >= 0 ? "text-long" : "text-short",
                        )}
                      >
                        {totals.pnlPct >= 0 ? "+" : ""}
                        {(totals.pnlPct * 100).toFixed(2)}%
                      </td>
                      <td colSpan={2} />
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            {/* ── Exit Reason Breakdown ─────────────────────────── */}
            <div className="space-y-2">
              <div className="flex items-center gap-2 px-1">
                <Target className="size-3.5 text-primary" />
                <span className="text-[11px] font-bold uppercase tracking-[0.15em] text-muted-foreground">
                  Exit Reasons
                </span>
              </div>
              <div className="flex flex-wrap gap-3">
                {exitBreakdown.map((item) => (
                  <div
                    key={item.reason}
                    className="flex items-center gap-2.5 rounded-lg bg-surface-2 px-4 py-2.5"
                  >
                    <Badge
                      variant={
                        item.variant as
                          | "long"
                          | "short"
                          | "volatile"
                          | "secondary"
                      }
                      className="shrink-0 text-[9px]"
                    >
                      {item.label}
                    </Badge>
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-sm font-bold tabular-nums text-foreground">
                        {item.count}
                      </span>
                      <span className="text-[10px] font-semibold text-muted-foreground">
                        ({item.pct.toFixed(0)}%)
                      </span>
                      {/* Mini bar */}
                      <div className="h-2 w-16 overflow-hidden rounded-full bg-surface-1">
                        <div
                          className={cn(
                            "h-full rounded-full transition-all",
                            item.variant === "long"
                              ? "bg-long/60"
                              : item.variant === "short"
                                ? "bg-short/60"
                                : item.variant === "volatile"
                                  ? "bg-volatile/60"
                                  : "bg-muted-foreground/40",
                          )}
                          style={{ width: `${Math.max(item.pct, 2)}%` }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* ── Backtest Params Summary ───────────────────────── */}
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1 px-1 pt-1">
              <span className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
                Params:
              </span>
              <span className="font-mono text-[10px] text-muted-foreground">
                Freq: {data.params?.bar_frequency ?? "daily"}
              </span>
              <span className="font-mono text-[10px] text-muted-foreground">
                Cash: {fmtDollar(data.params?.initial_cash ?? 100000)}
              </span>
              <span className="font-mono text-[10px] text-muted-foreground">
                Comm:{" "}
                {((data.params?.commission_rate ?? 0.0002) * 100).toFixed(2)}%
              </span>
              <span className="font-mono text-[10px] text-muted-foreground">
                SL: {data.params?.sl_atr_mult ?? 2}× ATR
              </span>
              <span className="font-mono text-[10px] text-muted-foreground">
                TP: {data.params?.tp_atr_mult ?? 1.5}× ATR
              </span>
              {data.params?.sr_cap_tp && (
                <Badge variant="gold" className="text-[9px]">
                  SR Cap TP
                </Badge>
              )}
              {data.params?.grid_enabled ? (
                <>
                  <Badge variant="long" className="gap-1 text-[9px]">
                    <Layers className="size-2.5" />
                    Grid {data.params?.grid_max_levels ?? 3}L
                  </Badge>
                  <span className="font-mono text-[10px] text-muted-foreground">
                    Decay: {data.params?.grid_sizing_decay ?? 0.7}
                  </span>
                </>
              ) : (
                <span className="font-mono text-[10px] text-muted-foreground">
                  Grid: OFF
                </span>
              )}
            </div>
          </>
        ) : data ? (
          <div className="flex items-center gap-2 rounded-lg border border-yellow-500/30 bg-yellow-500/10 px-4 py-3">
            <AlertCircle className="size-4 shrink-0 text-yellow-500" />
            <span className="text-xs font-semibold text-yellow-500">
              Incomplete data received from server.
            </span>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
