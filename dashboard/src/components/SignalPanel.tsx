import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  TrendingUp,
  TrendingDown,
  Minus,
  Zap,
  Activity,
} from "lucide-react";

interface Props {
  direction: string;
  predClass: number;
  score: number;
  pUp: number;
  pDown: number;
  conviction: string;
  sentiment: number;
}

const directionMeta: Record<
  string,
  { variant: "long" | "short" | "flat"; icon: React.ElementType; label: string }
> = {
  long: { variant: "long", icon: TrendingUp, label: "LONG" },
  short: { variant: "short", icon: TrendingDown, label: "SHORT" },
  flat: { variant: "flat", icon: Minus, label: "FLAT" },
};

function convictionColor(conviction: string): string {
  switch (conviction) {
    case "high":
      return "text-long";
    case "medium":
      return "text-volatile";
    case "low":
      return "text-flat";
    default:
      return "text-flat";
  }
}

export default function SignalPanel({
  direction,
  predClass,
  score,
  pUp,
  pDown,
  conviction,
  sentiment,
}: Props) {
  const meta = directionMeta[direction] || directionMeta.flat;
  const DirIcon = meta.icon;

  return (
    <Card className="relative flex h-full min-h-0 flex-col overflow-hidden">
      {/* Accent line at top */}
      <div
        className={`absolute top-0 left-0 right-0 h-[2px] ${
          direction === "long"
            ? "bg-long"
            : direction === "short"
              ? "bg-short"
              : "bg-flat"
        }`}
      />

      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Activity className="size-3.5 text-primary" />
            Signal
          </CardTitle>
          <Badge variant={meta.variant} className="gap-1.5">
            <DirIcon className="size-3" strokeWidth={3} />
            {meta.label}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-5">
        {/* Conviction */}
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-semibold uppercase tracking-widest text-muted-foreground">
            Conviction
          </span>
          <span
            className={`text-sm font-extrabold uppercase tracking-wider ${convictionColor(conviction)}`}
          >
            <Zap className="inline size-3.5 mr-1 -mt-0.5" strokeWidth={3} />
            {conviction}
          </span>
        </div>

        {/* Probability bars */}
        <div className="space-y-3">
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-bold uppercase tracking-widest text-muted-foreground">
                P(UP)
              </span>
              <span className="font-mono text-sm font-bold tabular-nums text-long">
                {(pUp * 100).toFixed(1)}%
              </span>
            </div>
            <div className="relative h-2 w-full overflow-hidden rounded-full bg-surface-3">
              <div
                className="absolute inset-y-0 left-0 rounded-full bg-long transition-all duration-700 ease-out"
                style={{ width: `${Math.max(pUp * 100, 1)}%` }}
              />
            </div>
          </div>

          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-bold uppercase tracking-widest text-muted-foreground">
                P(DOWN)
              </span>
              <span className="font-mono text-sm font-bold tabular-nums text-short">
                {(pDown * 100).toFixed(1)}%
              </span>
            </div>
            <div className="relative h-2 w-full overflow-hidden rounded-full bg-surface-3">
              <div
                className="absolute inset-y-0 left-0 rounded-full bg-short transition-all duration-700 ease-out"
                style={{ width: `${Math.max(pDown * 100, 1)}%` }}
              />
            </div>
          </div>
        </div>

        <Separator />

        {/* Stats row */}
        <div className="grid grid-cols-3 gap-4">
          <div className="space-y-1 text-center">
            <span className="block text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
              Score
            </span>
            <span className="block font-mono text-lg font-extrabold tabular-nums">
              {score.toFixed(3)}
            </span>
          </div>
          <div className="space-y-1 text-center">
            <span className="block text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
              Sentiment
            </span>
            <span
              className={`block font-mono text-lg font-extrabold tabular-nums ${
                sentiment > 0
                  ? "text-long"
                  : sentiment < 0
                    ? "text-short"
                    : "text-flat"
              }`}
            >
              {sentiment > 0 ? "+" : ""}
              {sentiment.toFixed(2)}
            </span>
          </div>
          <div className="space-y-1 text-center">
            <span className="block text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
              Class
            </span>
            <span className="block font-mono text-lg font-extrabold tabular-nums">
              {predClass}
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
