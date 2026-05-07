import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  Layers,
  Target,
  TrendingUp,
  TrendingDown,
  Minus,
  Gauge,
  Crosshair,
  ArrowUpDown,
} from "lucide-react";

interface Props {
  regime: string;
  srPrediction: Record<string, number>;
  volatility: Record<string, number>;
  plan: {
    mode: string;
    direction: string;
    stop_loss: number | null;
    take_profit: number | null;
  } | null;
}

const regimeMeta: Record<
  string,
  {
    variant: "long" | "short" | "ranging" | "volatile" | "flat" | "trending";
    icon: React.ElementType;
    label: string;
    color: string;
  }
> = {
  trending_up: {
    variant: "trending",
    icon: TrendingUp,
    label: "TRENDING UP",
    color: "bg-trending-up",
  },
  trending_down: {
    variant: "short",
    icon: TrendingDown,
    label: "TRENDING DOWN",
    color: "bg-trending-down",
  },
  ranging: {
    variant: "ranging",
    icon: ArrowUpDown,
    label: "RANGING",
    color: "bg-ranging",
  },
  volatile: {
    variant: "volatile",
    icon: Gauge,
    label: "VOLATILE",
    color: "bg-volatile",
  },
  unknown: {
    variant: "flat",
    icon: Minus,
    label: "UNKNOWN",
    color: "bg-flat",
  },
};

const modeIcons: Record<string, React.ElementType> = {
  trend: TrendingUp,
  range_grid: Layers,
  flat: Minus,
};

export default function RegimePanel({
  regime,
  srPrediction,
  volatility,
  plan,
}: Props) {
  const meta = regimeMeta[regime] || regimeMeta.unknown;
  const RegimeIcon = meta.icon;
  const ModeIcon = plan ? modeIcons[plan.mode] || Minus : Minus;
  const hasSR = srPrediction.predicted_high && srPrediction.predicted_low;

  return (
    <Card className="relative flex h-full min-h-0 flex-col overflow-hidden">
      {/* Regime accent line */}
      <div className={`absolute top-0 left-0 right-0 h-[2px] ${meta.color}`} />

      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Layers className="size-3.5 text-primary" />
            Regime & S/R
          </CardTitle>
          <div className="flex items-center gap-2">
            <Badge variant={meta.variant} className="gap-1.5">
              <RegimeIcon className="size-3" strokeWidth={3} />
              {meta.label}
            </Badge>
            {plan && (
              <Badge variant="secondary" className="gap-1.5">
                <ModeIcon className="size-3" />
                {plan.mode}
              </Badge>
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-5">
        {/* S/R Prediction Levels */}
        {hasSR && (
          <div className="space-y-3">
            <span className="text-[11px] font-bold uppercase tracking-[0.15em] text-muted-foreground">
              Support / Resistance
            </span>

            {/* Price level bars */}
            <div className="space-y-2">
              {/* Predicted High */}
              <div className="flex items-center gap-3 rounded-lg bg-surface-2 px-4 py-3">
                <div className="flex h-7 w-7 items-center justify-center rounded-md border border-short/20 bg-short/10">
                  <TrendingUp className="size-3.5 text-short" />
                </div>
                <div className="flex-1">
                  <span className="block text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
                    Predicted High
                  </span>
                  <span className="block font-mono text-base font-extrabold tabular-nums text-foreground">
                    {srPrediction.predicted_high.toFixed(2)}
                  </span>
                </div>
              </div>

              {/* Predicted Low */}
              <div className="flex items-center gap-3 rounded-lg bg-surface-2 px-4 py-3">
                <div className="flex h-7 w-7 items-center justify-center rounded-md border border-long/20 bg-long/10">
                  <TrendingDown className="size-3.5 text-long" />
                </div>
                <div className="flex-1">
                  <span className="block text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
                    Predicted Low
                  </span>
                  <span className="block font-mono text-base font-extrabold tabular-nums text-foreground">
                    {srPrediction.predicted_low.toFixed(2)}
                  </span>
                </div>
              </div>

              {/* Range & Bounce */}
              <div className="grid grid-cols-2 gap-2">
                <div className="rounded-lg bg-surface-2 px-4 py-3 text-center">
                  <span className="block text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
                    Range
                  </span>
                  <span className="block font-mono text-lg font-extrabold tabular-nums text-foreground">
                    {(
                      srPrediction.predicted_high - srPrediction.predicted_low
                    ).toFixed(2)}
                  </span>
                </div>
                <div className="rounded-lg bg-surface-2 px-4 py-3 text-center">
                  <span className="block text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
                    Bounce Prob
                  </span>
                  <span
                    className={`block font-mono text-lg font-extrabold tabular-nums ${
                      srPrediction.bounce_probability > 0.55
                        ? "text-long"
                        : "text-volatile"
                    }`}
                  >
                    {(srPrediction.bounce_probability * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* SL / TP from execution plan */}
        {plan && (plan.stop_loss || plan.take_profit) && (
          <>
            <Separator />
            <div className="space-y-2">
              <span className="text-[11px] font-bold uppercase tracking-[0.15em] text-muted-foreground">
                Execution Levels
              </span>
              <div className="grid grid-cols-2 gap-2">
                <div className="rounded-lg border border-short/20 bg-short/5 px-4 py-3">
                  <div className="mb-1.5 flex items-center gap-1.5">
                    <Crosshair className="size-3 text-short" />
                    <span className="text-[10px] font-bold uppercase tracking-widest text-short">
                      Stop Loss
                    </span>
                  </div>
                  <span className="block font-mono text-lg font-extrabold tabular-nums text-foreground">
                    {plan.stop_loss?.toFixed(2) || "—"}
                  </span>
                </div>
                <div className="rounded-lg border border-long/20 bg-long/5 px-4 py-3">
                  <div className="mb-1.5 flex items-center gap-1.5">
                    <Target className="size-3 text-long" />
                    <span className="text-[10px] font-bold uppercase tracking-widest text-long">
                      Take Profit
                    </span>
                  </div>
                  <span className="block font-mono text-lg font-extrabold tabular-nums text-foreground">
                    {plan.take_profit?.toFixed(2) || "—"}
                  </span>
                </div>
              </div>
            </div>
          </>
        )}

        {/* Volatility */}
        {volatility.conditional_vol && (
          <>
            <Separator />
            <div className="space-y-2">
              <span className="text-[11px] font-bold uppercase tracking-[0.15em] text-muted-foreground">
                Volatility
              </span>
              <div className="flex items-center justify-between rounded-lg bg-surface-2 px-4 py-3">
                <div className="flex items-center gap-2">
                  <Gauge className="size-4 text-muted-foreground" />
                  <span className="font-mono text-sm font-bold text-foreground">
                    {(volatility.conditional_vol * 100).toFixed(1)}%
                  </span>
                </div>
                <Badge
                  variant={
                    String(volatility.vol_regime) === "high"
                      ? "volatile"
                      : "secondary"
                  }
                >
                  {String(volatility.vol_regime).toUpperCase()}
                </Badge>
              </div>
              {volatility.trailing_stop_distance && (
                <div className="flex items-center justify-between px-1">
                  <span className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
                    Trailing Stop Distance
                  </span>
                  <span className="font-mono text-xs font-bold tabular-nums text-muted-foreground">
                    {volatility.trailing_stop_distance.toFixed(2)}
                  </span>
                </div>
              )}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
