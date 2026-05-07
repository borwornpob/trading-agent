import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";

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

interface Props {
  risk: RiskData;
}

function getRiskMeta(risk: RiskData): {
  label: string;
  variant: "long" | "volatile" | "short" | "event";
} {
  if (risk.kill_switch || risk.persisted_kill)
    return { label: "KILL SWITCH", variant: "event" };
  if (risk.recovery_mode) return { label: "RECOVERY", variant: "volatile" };
  if (risk.consecutive_losses >= 2)
    return { label: "CAUTION", variant: "volatile" };
  return { label: "NOMINAL", variant: "long" };
}

function getRiskDotColor(risk: RiskData): string {
  if (risk.kill_switch || risk.persisted_kill)
    return "bg-short";
  if (risk.recovery_mode || risk.consecutive_losses >= 2)
    return "bg-volatile";
  return "bg-long";
}

export default function RiskPanel({ risk }: Props) {
  const { label, variant } = getRiskMeta(risk);
  const dotColor = getRiskDotColor(risk);
  const isKill = risk.kill_switch || risk.persisted_kill;

  return (
    <Card className="relative flex h-full min-h-0 flex-col overflow-hidden">
      {/* State accent */}
      <div
        className={`absolute top-0 left-0 h-[2px] w-full ${
          isKill
            ? "bg-short"
            : risk.recovery_mode
              ? "bg-volatile"
              : "bg-long"
        }`}
      />

      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <CardTitle>Risk State</CardTitle>
          <div className="flex items-center gap-2.5">
            <div
              className={`h-2.5 w-2.5 rounded-full ${dotColor} ${isKill ? "animate-pulse" : ""}`}
            />
            <Badge variant={variant}>{label}</Badge>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-5">
        {/* Hero Metric */}
        <div>
          <span className="text-[11px] font-semibold uppercase tracking-[0.15em] text-muted-foreground">
            Daily P&L
          </span>
          <div
            className={`mt-1 font-mono text-3xl font-extrabold tracking-tight ${
              risk.realized_pnl >= 0 ? "text-long" : "text-short"
            }`}
          >
            {risk.realized_pnl >= 0 ? "+" : ""}
            {risk.realized_pnl.toFixed(0)}
          </div>
        </div>

        <Separator />

        {/* Metrics Grid */}
        <div className="grid grid-cols-3 gap-4">
          <div className="space-y-1">
            <span className="text-[10px] font-bold uppercase tracking-[0.12em] text-muted-foreground">
              Losses
            </span>
            <div className="flex items-baseline gap-1.5">
              <span
                className={`font-mono text-xl font-extrabold ${
                  risk.consecutive_losses >= 2
                    ? "text-volatile"
                    : "text-foreground"
                }`}
              >
                {risk.consecutive_losses}
              </span>
              <span className="text-[10px] text-muted-foreground">streak</span>
            </div>
          </div>

          <div className="space-y-1">
            <span className="text-[10px] font-bold uppercase tracking-[0.12em] text-muted-foreground">
              Size
            </span>
            <div className="font-mono text-xl font-extrabold text-foreground">
              {risk.sizing_multiplier}x
            </div>
          </div>

          <div className="space-y-1">
            <span className="text-[10px] font-bold uppercase tracking-[0.12em] text-muted-foreground">
              Mode
            </span>
            <Badge
              variant={risk.shadow_mode ? "shadow" : "long"}
              className="mt-0.5"
            >
              {risk.shadow_mode ? "SHADOW" : "LIVE"}
            </Badge>
          </div>
        </div>

        {/* Kill switch indicator */}
        {isKill && (
          <>
            <Separator />
            <div className="flex items-center gap-2 rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3">
              <div className="h-1.5 w-1.5 animate-pulse rounded-full bg-destructive" />
              <span className="text-xs font-bold uppercase tracking-wider text-destructive">
                All trading halted
              </span>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
