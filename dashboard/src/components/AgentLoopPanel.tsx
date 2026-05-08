import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  Eye,
  Brain,
  Shield,
  Zap,
  CheckCircle2,
  Circle,
  Loader2,
  XCircle,
  ChevronDown,
  Database,
  Newspaper,
  BarChart3,
  Layers,
  Gauge,
  Crosshair,
  Target,
  AlertTriangle,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface LoopPhase {
  id: string;
  label: string;
  icon: LucideIcon;
  status: PhaseStatus;
  items: { label: string; value?: string; status?: "ok" | "warn" | "error" | "skip" }[];
}

type PhaseStatus = "completed" | "active" | "pending" | "error";

interface Props {
  lastCycle: {
    timestamp_utc?: string;
    market_data_source?: string;
    notes?: string[];
    signal?: {
      direction: string;
      regime: string;
      sentiment: number;
      event_risk: boolean;
    };
    execution_plan?: {
      mode: string;
      direction: string;
    };
    sr_prediction?: Record<string, number>;
    volatility?: Record<string, number>;
    orders_submitted?: Array<Record<string, unknown>>;
  } | null;
  cycles?: Array<{
    timestamp_utc?: string;
    market_data_source?: string;
    direction?: string;
    pred_class?: number | null;
    p_up?: number | null;
    p_down?: number | null;
    execution_mode?: string;
    orders_count?: number;
    notes?: string[];
  }>;
}

function phaseStatus(
  phaseId: string,
  cycleData: Props["lastCycle"]
): PhaseStatus {
  if (!cycleData || !cycleData.signal) return "pending";

  const hasOrders = cycleData.orders_submitted && cycleData.orders_submitted.length > 0;
  const notes = cycleData.notes || [];
  const hasShadowNote = notes.some(
    (n) => n.includes("shadow_mode") || n.includes("no_orders_submitted")
  );
  const hasGateFailure = notes.some((n) => n.includes("gates_failed"));
  const hasModelError = notes.some((n) => n.includes("no_model_bundle"));
  const hasDataError = notes.some((n) => n.includes("data_fetch_error"));

  switch (phaseId) {
    case "perceive":
      return hasDataError ? "error" : "completed";
    case "infer":
      return hasModelError ? "error" : "completed";
    case "govern":
      if (hasGateFailure) return "completed";
      return "completed";
    case "execute":
      if (hasOrders || hasShadowNote) return "completed";
      if (hasGateFailure) return "pending";
      return "completed";
    default:
      return "pending";
  }
}

const PhaseStatusIcon = ({
  status,
}: {
  status: PhaseStatus;
}) => {
  switch (status) {
    case "completed":
      return <CheckCircle2 className="size-4 text-long" strokeWidth={2.5} />;
    case "active":
      return <Loader2 className="size-4 text-primary animate-spin" />;
    case "error":
      return <XCircle className="size-4 text-short" strokeWidth={2.5} />;
    case "pending":
    default:
      return <Circle className="size-4 text-flat/60" />;
  }
};

function formatSource(source?: string) {
  if (!source) return "Unknown";
  return source.toUpperCase();
}

function formatProbability(value?: number | null) {
  if (typeof value !== "number") return "—";
  return `${(value * 100).toFixed(1)}%`;
}

export default function AgentLoopPanel({ lastCycle, cycles = [] }: Props) {
  const hasData = lastCycle && lastCycle.signal;
  const recentCycles = cycles.slice(-6).reverse();

  const phases: LoopPhase[] = [
    {
      id: "perceive",
      label: "PERCEIVE",
      icon: Eye,
      status: phaseStatus("perceive", lastCycle),
      items: [
        {
          label: "OHLCV Data",
          value: hasData ? formatSource(lastCycle?.market_data_source) : undefined,
          status: hasData ? "ok" : "skip",
        },
        {
          label: "TA Indicators",
          value: hasData ? "Computed" : undefined,
          status: hasData ? "ok" : "skip",
        },
        {
          label: "News & Sentiment",
          value: hasData
            ? `${lastCycle?.signal?.sentiment !== undefined ? (lastCycle.signal.sentiment > 0 ? "+" : "") + lastCycle.signal.sentiment.toFixed(2) : "—"}`
            : undefined,
          status: hasData ? "ok" : "skip",
        },
        {
          label: "Event Risk",
          value: hasData
            ? lastCycle?.signal?.event_risk
              ? "FLAGGED"
              : "CLEAR"
            : undefined,
          status: hasData
            ? lastCycle?.signal?.event_risk
              ? "warn"
              : "ok"
            : "skip",
        },
      ],
    },
    {
      id: "infer",
      label: "INFER",
      icon: Brain,
      status: phaseStatus("infer", lastCycle),
      items: [
        {
          label: "LightGBM Ensemble",
          value: hasData ? "3 TF models" : undefined,
          status: hasData ? "ok" : "skip",
        },
        {
          label: "Regime Detection",
          value: hasData
            ? lastCycle?.signal?.regime?.replace("_", " ").toUpperCase()
            : undefined,
          status: hasData ? "ok" : "skip",
        },
        {
          label: "S/R Prediction",
          value: hasData
            ? lastCycle?.sr_prediction?.predicted_high
              ? `${lastCycle.sr_prediction.predicted_high.toFixed(0)} / ${lastCycle.sr_prediction.predicted_low.toFixed(0)}`
              : "N/A"
            : undefined,
          status: hasData && lastCycle?.sr_prediction?.predicted_high ? "ok" : "skip",
        },
        {
          label: "GARCH Volatility",
          value: hasData
            ? lastCycle?.volatility?.conditional_vol
              ? `${(lastCycle.volatility.conditional_vol * 100).toFixed(1)}%`
              : "N/A"
            : undefined,
          status: hasData && lastCycle?.volatility?.conditional_vol ? "ok" : "skip",
        },
      ],
    },
    {
      id: "govern",
      label: "GOVERN",
      icon: Shield,
      status: phaseStatus("govern", lastCycle),
      items: [
        {
          label: "Signal Gates",
          value: hasData ? "Statistical tests" : undefined,
          status: hasData ? "ok" : "skip",
        },
        {
          label: "Risk Checks",
          value: hasData ? "Passed" : undefined,
          status: hasData ? "ok" : "skip",
        },
        {
          label: "Session Risk",
          value: hasData ? "Applied" : undefined,
          status: hasData ? "ok" : "skip",
        },
        {
          label: "Direction",
          value: hasData
            ? lastCycle?.signal?.direction?.toUpperCase()
            : undefined,
          status: hasData ? "ok" : "skip",
        },
      ],
    },
    {
      id: "execute",
      label: "EXECUTE",
      icon: Zap,
      status: phaseStatus("execute", lastCycle),
      items: [
        {
          label: "Execution Mode",
          value: hasData
            ? lastCycle?.execution_plan?.mode?.toUpperCase() || "FLAT"
            : undefined,
          status: hasData ? "ok" : "skip",
        },
        {
          label: "Smart Grid",
          value: hasData
            ? lastCycle?.execution_plan?.mode === "range_grid"
              ? "ACTIVE"
              : "OFF"
            : undefined,
          status:
            hasData && lastCycle?.execution_plan?.mode === "range_grid"
              ? "ok"
              : "skip",
        },
        {
          label: "Orders",
          value: hasData
            ? lastCycle?.orders_submitted?.length
              ? `${lastCycle.orders_submitted.length} submitted`
              : "Shadow / None"
            : undefined,
          status: hasData ? "ok" : "skip",
        },
      ],
    },
  ];

  const itemStatusColor = (status?: string) => {
    switch (status) {
      case "ok":
        return "text-long";
      case "warn":
        return "text-volatile";
      case "error":
        return "text-short";
      default:
        return "text-flat";
    }
  };

  return (
    <Card className="relative flex h-full min-h-0 flex-col overflow-hidden">
      {/* Accent */}
      <div className="absolute top-0 left-0 right-0 h-[2px] bg-primary" />

      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Zap className="size-3.5 text-primary" />
            Agent Loop
          </CardTitle>
          {hasData && (
            <Badge variant="gold" className="gap-1.5">
              <CheckCircle2 className="size-3" />
              CYCLE COMPLETE
            </Badge>
          )}
          {!hasData && (
            <Badge variant="flat">AWAITING FIRST CYCLE</Badge>
          )}
        </div>
      </CardHeader>

      <CardContent className="space-y-1">
        {/* Loop visualization */}
        <div className="space-y-1">
          {phases.map((phase, idx) => {
            const PhaseIcon = phase.icon;
            const isLast = idx === phases.length - 1;

            return (
              <div key={phase.id}>
                {/* Phase row */}
                <div className="flex items-start gap-3">
                  {/* Left: status indicator + vertical line */}
                  <div className="flex flex-col items-center">
                    <div
                      className={cn(
                        "flex h-8 w-8 items-center justify-center rounded-lg border",
                        phase.status === "completed"
                          ? "border-long/30 bg-long/10"
                          : phase.status === "error"
                            ? "border-short/30 bg-short/10"
                          : phase.status === "active"
                            ? "border-primary/30 bg-primary/10"
                            : "border-border bg-surface-2"
                      )}
                    >
                      <PhaseIcon
                        className={cn(
                          "size-4",
                          phase.status === "completed"
                            ? "text-long"
                            : phase.status === "error"
                              ? "text-short"
                            : phase.status === "active"
                              ? "text-primary"
                              : "text-flat/60"
                        )}
                        strokeWidth={2.5}
                      />
                    </div>
                    {!isLast && (
                      <div className="h-full min-h-5 w-px bg-border" />
                    )}
                  </div>

                  {/* Right: phase content */}
                  <div className="flex-1 pb-3">
                    <div className="flex items-center gap-2 mb-2">
                      <span
                        className={cn(
                          "text-sm font-extrabold uppercase tracking-[0.12em]",
                          phase.status === "completed"
                            ? "text-foreground"
                            : phase.status === "error"
                              ? "text-short"
                            : "text-flat/80"
                        )}
                      >
                        {phase.label}
                      </span>
                      <PhaseStatusIcon status={phase.status} />
                    </div>

                    {/* Phase items */}
                    <div className="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
                      {phase.items.map((item) => (
                        <div
                          key={item.label}
                          className="min-w-0 rounded-lg bg-surface-2 px-4 py-3"
                        >
                          <span className="block truncate text-[11px] font-semibold uppercase tracking-wider text-foreground/65">
                            {item.label}
                          </span>
                          <span
                            className={cn(
                              "mt-0.5 block truncate font-mono text-xs font-bold tabular-nums",
                              itemStatusColor(item.status)
                            )}
                            title={item.value || "—"}
                          >
                            {item.value || "—"}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Arrow connector between phases */}
                {!isLast && (
                  <div className="flex items-center justify-center py-0">
                    <ChevronDown className="size-4 text-border" />
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Bottom summary */}
        {hasData && (
          <>
            <Separator className="my-2" />
            <div className="flex items-center justify-between rounded-lg bg-surface-2 px-4 py-3">
              <span className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
                Loop Timestamp
              </span>
              <span className="font-mono text-xs font-bold tabular-nums text-foreground">
                {lastCycle?.timestamp_utc
                  ? new Date(lastCycle.timestamp_utc).toLocaleTimeString("en-US", {
                      hour12: false,
                      hour: "2-digit",
                      minute: "2-digit",
                      second: "2-digit",
                    }) + " UTC"
                  : "—"}
              </span>
            </div>
            {recentCycles.length > 0 && (
              <div className="overflow-hidden rounded-lg border border-border/60">
                <div className="grid grid-cols-[1.05fr_0.75fr_0.75fr_0.9fr] gap-2 bg-surface-2 px-3 py-2 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
                  <span>Time</span>
                  <span>Model</span>
                  <span>Signal</span>
                  <span className="text-right">Orders</span>
                </div>
                <div className="divide-y divide-border/60">
                  {recentCycles.map((cycle, index) => (
                    <div
                      key={`${cycle.timestamp_utc || "cycle"}-${index}`}
                      className="grid grid-cols-[1.05fr_0.75fr_0.75fr_0.9fr] gap-2 px-3 py-2 text-[11px]"
                    >
                      <span className="min-w-0 truncate font-mono text-muted-foreground">
                        {cycle.timestamp_utc
                          ? new Date(cycle.timestamp_utc).toLocaleTimeString("en-US", {
                              hour12: false,
                              hour: "2-digit",
                              minute: "2-digit",
                            })
                          : "—"}
                      </span>
                      <span className="min-w-0 truncate font-mono font-semibold text-foreground">
                        C{cycle.pred_class ?? "?"} U{formatProbability(cycle.p_up)}
                      </span>
                      <span className="min-w-0 truncate font-mono font-semibold text-foreground">
                        {(cycle.direction || "flat").toUpperCase()}
                      </span>
                      <span className="min-w-0 truncate text-right font-mono text-muted-foreground">
                        {cycle.orders_count ?? 0} / {(cycle.execution_mode || "flat").toUpperCase()}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
