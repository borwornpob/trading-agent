import { useState, useEffect, useCallback } from "react";
import { cn } from "@/lib/utils";
import DashboardLayout from "./components/DashboardLayout";
import SignalPanel from "./components/SignalPanel";
import RiskPanel from "./components/RiskPanel";
import RegimePanel from "./components/RegimePanel";
import SentimentPanel from "./components/SentimentPanel";
import AgentLoopPanel from "./components/AgentLoopPanel";
import BacktestPanel from "./components/BacktestPanel";
import ConfigPanel from "./components/ConfigPanel";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { FileText, AlertCircle, Activity } from "lucide-react";

const API_BASE = "/api";

interface SignalData {
  direction: string;
  pred_class: number;
  score: number;
  p_up: number;
  p_down: number;
  regime: string;
  sentiment: number;
  event_risk: boolean;
  conviction: string;
}

interface RiskData {
  shadow_mode: boolean;
  kill_switch: boolean;
  persisted_kill: boolean;
  day_utc: string | null;
  realized_pnl: number;
  consecutive_losses: number;
  recovery_mode: boolean;
  sizing_multiplier: number;
}

interface NewsItem {
  headline: string;
  sentiment: number;
  confidence: number;
  key_drivers: string[];
  impact_horizon: string;
  event_risk: boolean;
  gold_relevant: boolean;
  source?: string;
  url?: string;
}

interface DashboardData {
  config: {
    symbol: string;
    broker_symbol: string;
    bar_frequency: string;
    shadow_mode: boolean;
    kill_switch: boolean;
    grid_enabled: boolean;
    genai?: {
      model: string;
      provider: string;
      base_url: string;
      news_enabled: boolean;
      configured: boolean;
    };
  };
  risk: RiskData;
  account?: Record<string, unknown> | null;
  positions?: Array<Record<string, unknown>>;
  orders?: Array<Record<string, unknown>>;
  live_status?: {
    status: string;
    updated_at_utc: string | null;
    error?: string | null;
  };
  last_cycle: {
    timestamp_utc: string;
    market_data_source?: string;
    signal: SignalData;
    execution_plan: {
      mode: string;
      direction: string;
      stop_loss: number | null;
      take_profit: number | null;
    };
    sr_prediction: Record<string, number>;
    volatility: Record<string, number>;
    orders_submitted: Array<Record<string, unknown>>;
    notes: string[];
  } | null;
  cycles?: Array<{
    timestamp_utc?: string;
    market_data_source?: string;
    direction?: string;
    pred_class?: number | null;
    p_up?: number | null;
    p_down?: number | null;
    score?: number | null;
    conviction?: string | null;
    regime?: string;
    execution_mode?: string;
    orders_count?: number;
    notes?: string[];
  }>;
  news?: {
    headlines?: NewsItem[];
    sentiment?: number;
    event_risk?: boolean;
    status?: string;
    model?: string;
    provider?: string;
    enabled?: boolean;
    configured?: boolean;
  };
}

function App() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(false);
  const [wsStatus, setWsStatus] = useState<
    "connecting" | "connected" | "disconnected"
  >("disconnected");

  const fetchDashboard = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/dashboard`);
      const json = await res.json();
      setData(json);
    } catch (err) {
      console.error("Failed to fetch dashboard:", err);
    }
  }, []);

  const refreshCycle = useCallback(async () => {
    setLoading(true);
    try {
      await fetch(`${API_BASE}/dashboard/refresh`, { method: "POST" });
      await fetchDashboard();
    } catch (err) {
      console.error("Failed to refresh:", err);
    } finally {
      setLoading(false);
    }
  }, [fetchDashboard]);

  useEffect(() => {
    fetchDashboard();
    const poll = window.setInterval(fetchDashboard, 10000);

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    const ws = new WebSocket(wsUrl);

    setWsStatus("connecting");

    ws.onopen = () => setWsStatus("connected");
    ws.onclose = () => setWsStatus("disconnected");
    ws.onerror = () => setWsStatus("disconnected");
    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === "cycle_complete") {
          fetchDashboard();
        }
      } catch {
        // Ignore non-JSON control frames from future websocket protocols.
      }
    };

    return () => {
      window.clearInterval(poll);
      ws.close();
    };
  }, [fetchDashboard]);

  const signal = data?.last_cycle?.signal;
  const plan = data?.last_cycle?.execution_plan;
  const risk = data?.risk;
  const sr = data?.last_cycle?.sr_prediction;
  const vol = data?.last_cycle?.volatility;
  const notes = data?.last_cycle?.notes || [];
  const hasSignal = Boolean(signal);
  const hasRisk = Boolean(risk);
  const hasRegime = Boolean(signal && sr);
  const hasNotes = notes.length > 0;
  const signalSpan = hasSignal && hasRisk ? "lg:col-span-4" : "lg:col-span-6";
  const sentimentSpan =
    hasSignal || hasRisk
      ? hasSignal && hasRisk
        ? "lg:col-span-4"
        : "lg:col-span-6"
      : "lg:col-span-12";
  const riskSpan = hasSignal ? "lg:col-span-4" : "lg:col-span-6";
  const agentSpan = hasRegime ? "lg:col-span-7" : "lg:col-span-12";
  const backtestSpan = hasNotes ? "lg:col-span-5" : "lg:col-span-7";
  const configSpan = hasNotes ? "lg:col-span-3" : "lg:col-span-5";
  const cell = (extra: string) =>
    cn(
      "min-h-0",
      "flex flex-col",
      // Stretch inner cards in bento cells
      "[&>*]:min-h-0 [&>*]:h-full [&>*]:flex-1",
      extra,
    );

  // Filter out informational notes for the "issues" display
  const issueNotes = notes.filter(
    (n) =>
      n.includes("error") ||
      n.includes("blocked") ||
      n.includes("failed") ||
      n.includes("not_available"),
  );
  const infoNotes = notes.filter((n) => !issueNotes.includes(n));
  const account = data?.account;
  const positions = data?.positions || [];
  const orders = data?.orders || [];
  const liveStatus = data?.live_status;

  return (
    <DashboardLayout
      wsStatus={wsStatus}
      onRefresh={refreshCycle}
      loading={loading}
      shadowMode={data?.config.shadow_mode ?? true}
    >
      <div
        className={cn(
          "rounded-3xl border border-border/40 bg-gradient-to-br from-surface-1/85 via-surface-0/40 to-background p-3.5 shadow-[inset_0_1px_0_0_rgba(255,255,255,0.35)] sm:rounded-[1.65rem] sm:p-4 lg:p-5",
        )}
      >
        <div
          className={cn(
            "grid grid-cols-1 items-stretch gap-3.5 sm:gap-4",
            "lg:grid-cols-12 lg:gap-4",
            "lg:grid-flow-dense lg:[grid-template-columns:repeat(12,minmax(0,1fr))] lg:items-stretch",
          )}
        >
        {/* ── Row 1: Signal + Sentiment + Risk ─────────────────── */}
        {signal && (
          <div className={cell(signalSpan)}>
            <SignalPanel
              direction={signal.direction}
              predClass={signal.pred_class}
              score={signal.score}
              pUp={signal.p_up}
              pDown={signal.p_down}
              conviction={signal.conviction}
              sentiment={signal.sentiment}
            />
          </div>
        )}

        <div className={cell(sentimentSpan)}>
          <SentimentPanel
            sentiment={data?.news?.sentiment ?? signal?.sentiment ?? 0}
            eventRisk={data?.news?.event_risk ?? signal?.event_risk ?? false}
            headlines={data?.news?.headlines}
            model={data?.news?.model ?? data?.config.genai?.model}
            provider={data?.news?.provider ?? data?.config.genai?.provider}
            status={data?.news?.status}
            enabled={data?.news?.enabled ?? data?.config.genai?.news_enabled}
            configured={data?.news?.configured ?? data?.config.genai?.configured}
          />
        </div>

        {risk && (
          <div className={cell(riskSpan)}>
            <RiskPanel risk={risk} />
          </div>
        )}

        {/* ── Row 2: Regime + Agent Loop ───────────────────────── */}
        {signal && sr && (
          <div className={cell("lg:col-span-5")}>
            <RegimePanel
              regime={signal.regime}
              srPrediction={sr}
              volatility={vol || {}}
              plan={plan || null}
            />
          </div>
        )}

        <div
          className={cell(
            cn(
              agentSpan,
              "lg:min-h-[22rem] xl:min-h-[26rem]",
            ),
          )}
          data-bento="agent"
        >
          <AgentLoopPanel
            lastCycle={data?.last_cycle ?? null}
            cycles={data?.cycles ?? []}
          />
        </div>

        {/* ── Row 3: Backtest + Config ─────────────────────────── */}
        <div className={cell(backtestSpan)}>
          <BacktestPanel />
        </div>

        <div className={cell(configSpan)}>
          <ConfigPanel
            shadowMode={data?.config.shadow_mode}
            gridEnabled={data?.config.grid_enabled}
            onUpdate={fetchDashboard}
          />
        </div>

        <div className={cell(hasNotes ? "lg:col-span-5" : "lg:col-span-7")}>
          <Card className="h-full">
            <CardHeader className="pb-4">
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2">
                  <Activity className="size-3.5 text-primary" />
                  Live MT5 State
                </CardTitle>
                <Badge variant={liveStatus?.status === "running" ? "gold" : "secondary"}>
                  {liveStatus?.status?.toUpperCase() || "WAITING"}
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="grid grid-cols-2 gap-2">
              <div className="rounded-lg bg-surface-2 px-4 py-3">
                <span className="block text-[11px] font-semibold uppercase tracking-wider text-foreground/65">
                  Equity
                </span>
                <span className="mt-0.5 block font-mono text-xs font-bold tabular-nums">
                  {typeof account?.equity === "number" ? account.equity.toFixed(2) : "—"}
                </span>
              </div>
              <div className="rounded-lg bg-surface-2 px-4 py-3">
                <span className="block text-[11px] font-semibold uppercase tracking-wider text-foreground/65">
                  Free Margin
                </span>
                <span className="mt-0.5 block font-mono text-xs font-bold tabular-nums">
                  {typeof account?.free_margin === "number"
                    ? account.free_margin.toFixed(2)
                    : "—"}
                </span>
              </div>
              <div className="rounded-lg bg-surface-2 px-4 py-3">
                <span className="block text-[11px] font-semibold uppercase tracking-wider text-foreground/65">
                  Open Positions
                </span>
                <span className="mt-0.5 block font-mono text-xs font-bold tabular-nums">
                  {positions.length}
                </span>
              </div>
              <div className="rounded-lg bg-surface-2 px-4 py-3">
                <span className="block text-[11px] font-semibold uppercase tracking-wider text-foreground/65">
                  Orders Seen
                </span>
                <span className="mt-0.5 block font-mono text-xs font-bold tabular-nums">
                  {orders.length}
                </span>
              </div>
              <div className="col-span-2 rounded-lg bg-surface-2 px-4 py-3">
                <span className="block text-[11px] font-semibold uppercase tracking-wider text-foreground/65">
                  Market Data
                </span>
                <span className="mt-0.5 block font-mono text-xs font-bold tabular-nums">
                  {data?.last_cycle?.market_data_source?.toUpperCase() || "—"}
                </span>
              </div>
              <div className="col-span-2 rounded-lg bg-surface-2 px-4 py-3">
                <span className="block text-[11px] font-semibold uppercase tracking-wider text-foreground/65">
                  Last Sync
                </span>
                <span className="mt-0.5 block font-mono text-xs font-bold tabular-nums">
                  {liveStatus?.updated_at_utc
                    ? new Date(liveStatus.updated_at_utc).toLocaleString()
                    : "—"}
                </span>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* ── Row 3 continued: Cycle Notes ─────────────────────── */}
        {hasNotes && (
          <div className={cell("lg:col-span-4")}>
            <Card className="h-full">
              <CardHeader className="pb-4">
                <div className="flex items-center justify-between">
                  <CardTitle className="flex items-center gap-2">
                    <FileText className="size-3.5 text-primary" />
                    Cycle Notes
                  </CardTitle>
                  <Badge variant="secondary">{notes.length}</Badge>
                </div>
              </CardHeader>
              <CardContent className="space-y-2">
                {/* Issues */}
                {issueNotes.length > 0 && (
                  <div className="space-y-1.5">
                    {issueNotes.map((note, i) => (
                      <div
                        key={`issue-${i}`}
                        className="flex items-start gap-2 rounded-lg border border-destructive/20 bg-destructive/10 px-3 py-2"
                      >
                        <AlertCircle className="mt-0.5 size-3.5 shrink-0 text-destructive" />
                        <span className="font-mono text-xs font-semibold text-destructive">
                          {note}
                        </span>
                      </div>
                    ))}
                  </div>
                )}

                {/* Info */}
                {infoNotes.length > 0 && (
                  <div className="space-y-1">
                    {infoNotes.map((note, i) => (
                      <div
                        key={`info-${i}`}
                        className="flex items-center gap-2 rounded-md bg-surface-2 px-3 py-1.5"
                      >
                        <div className="h-1 w-1 shrink-0 rounded-full bg-primary" />
                        <span className="font-mono text-[11px] text-muted-foreground">
                          {note}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        )}
        </div>
      </div>
    </DashboardLayout>
  );
}

export default App;
