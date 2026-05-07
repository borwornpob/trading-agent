import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  Brain,
  AlertTriangle,
  Sparkles,
  Shield,
  Clock,
  Zap,
} from "lucide-react";

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

interface Props {
  sentiment: number;
  eventRisk: boolean;
  headlines?: NewsItem[];
  model?: string;
  provider?: string;
  status?: string;
  enabled?: boolean;
  configured?: boolean;
}

function sentimentColor(score: number): string {
  if (score > 0.3) return "text-long";
  if (score > 0) return "text-long/80";
  if (score === 0) return "text-flat";
  if (score > -0.3) return "text-short/80";
  return "text-short";
}

function sentimentLabel(score: number): string {
  if (score > 0.5) return "VERY BULLISH";
  if (score > 0.2) return "BULLISH";
  if (score > 0) return "SLIGHTLY BULLISH";
  if (score === 0) return "NEUTRAL";
  if (score > -0.2) return "SLIGHTLY BEARISH";
  if (score > -0.5) return "BEARISH";
  return "VERY BEARISH";
}

function horizonIcon(horizon: string): string {
  switch (horizon) {
    case "short":
      return "< 24h";
    case "medium":
      return "1–5d";
    case "long":
      return "> 5d";
    default:
      return horizon;
  }
}

function emptyMessage(
  status?: string,
  enabled?: boolean,
  configured?: boolean,
): string {
  if (enabled === false) return "News scoring disabled";
  if (configured === false || status === "missing_api_key") {
    return "GenAI API key missing";
  }
  if (status === "no_headlines") return "No RSS headlines returned";
  if (status === "error") return "Sentiment refresh failed";
  return "Run a cycle to fetch news";
}

export default function SentimentPanel({
  sentiment,
  eventRisk,
  headlines = [],
  model,
  provider,
  status,
  enabled,
  configured,
}: Props) {
  const hasHeadlines = headlines.length > 0;
  const avgConfidence = hasHeadlines
    ? headlines.reduce((sum, h) => sum + h.confidence, 0) / headlines.length
    : 0;
  const modelLabel = model?.trim() || "Model not set";
  const providerLabel = provider?.trim() || "OpenAI-compatible";

  // Clamp sentiment between -1 and 1
  const clampedSentiment = Math.max(-1, Math.min(1, sentiment));
  // Map -1..1 to 0..100 for gauge position
  const gaugePosition = ((clampedSentiment + 1) / 2) * 100;

  return (
    <Card className="relative flex h-full min-h-0 flex-col overflow-hidden">
      {/* Sentiment accent */}
      <div
        className={`absolute top-0 left-0 right-0 h-[2px] ${
          sentiment > 0
            ? "bg-long"
            : sentiment < 0
              ? "bg-short"
              : "bg-flat"
        }`}
      />

      <CardHeader className="relative pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Brain className="size-3.5 text-primary" />
            GenAI Sentiment
          </CardTitle>
          <div className="flex items-center gap-2">
            {eventRisk && (
              <Badge variant="event" className="gap-1.5">
                <AlertTriangle className="size-3" />
                EVENT RISK
              </Badge>
            )}
            <Badge variant="gold" className="gap-1.5">
              <Sparkles className="size-3" />
              {providerLabel}
            </Badge>
            <Badge
              variant="secondary"
              className="max-w-[9rem] truncate text-[10px]"
              title={modelLabel}
            >
              {modelLabel}
            </Badge>
          </div>
        </div>
      </CardHeader>

      <CardContent className="relative space-y-4">
        {/* Hero Sentiment Score */}
        <div className="flex items-end justify-between">
          <div>
            <span className="block text-[10px] font-bold uppercase tracking-[0.2em] text-muted-foreground">
              Weighted Sentiment
            </span>
            <div className="mt-1 flex items-baseline gap-2">
              <span
                className={`font-mono text-4xl font-extrabold tracking-tighter tabular-nums ${sentimentColor(sentiment)}`}
              >
                {sentiment > 0 ? "+" : ""}
                {sentiment.toFixed(2)}
              </span>
              <span className="text-sm font-bold uppercase tracking-wider text-muted-foreground">
                {sentimentLabel(sentiment)}
              </span>
            </div>
          </div>
          <Badge
            variant={
              sentiment > 0.2 ? "long" : sentiment < -0.2 ? "short" : "flat"
            }
            className="text-[10px]"
          >
            {sentiment > 0.2
              ? "BULLISH"
              : sentiment < -0.2
                ? "BEARISH"
                : "NEUTRAL"}
          </Badge>
        </div>

        {/* Sentiment Gauge */}
        <div className="space-y-2">
          <div className="relative h-3 w-full overflow-hidden rounded-full bg-surface-3">
            <div className="absolute left-1/2 top-0 h-full w-px bg-border" />
            {/* Indicator */}
            <div
              className="absolute top-0 h-full w-1 -translate-x-1/2 rounded-full bg-primary transition-all duration-700 ease-out"
              style={{ left: `${gaugePosition}%` }}
            />
          </div>
          <div className="flex justify-between">
            <span className="text-[10px] font-bold uppercase tracking-widest text-short/70">
              Bearish
            </span>
            <span className="text-[10px] font-bold uppercase tracking-widest text-flat">
              Neutral
            </span>
            <span className="text-[10px] font-bold uppercase tracking-widest text-long/70">
              Bullish
            </span>
          </div>
        </div>

        <Separator />

        {/* Event Risk Banner */}
        {eventRisk && (
          <div className="flex items-center gap-2.5 rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3">
            <Shield className="size-4 shrink-0 text-destructive" />
            <div className="min-w-0">
              <span className="block text-[11px] font-bold uppercase tracking-wider text-destructive">
                High-Impact Event Detected
              </span>
              <span className="block text-[10px] text-destructive/70">
                NFP · FOMC · CPI · Rate Decision - Grid disabled
              </span>
            </div>
          </div>
        )}

        {/* Confidence + Headlines Count */}
        {hasHeadlines && (
          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-lg bg-surface-2 px-4 py-3 text-center">
              <span className="block text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
                Avg Confidence
              </span>
              <span className="block font-mono text-lg font-extrabold tabular-nums text-foreground">
                {(avgConfidence * 100).toFixed(0)}%
              </span>
            </div>
            <div className="rounded-lg bg-surface-2 px-4 py-3 text-center">
              <span className="block text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
                Headlines
              </span>
              <span className="block font-mono text-lg font-extrabold tabular-nums text-foreground">
                {headlines.length}
              </span>
            </div>
          </div>
        )}

        {/* Headline List */}
        {hasHeadlines && (
          <div className="space-y-2">
            <span className="text-[11px] font-bold uppercase tracking-[0.15em] text-muted-foreground">
              Recent Analysis
            </span>
            <div className="space-y-1.5 max-h-[280px] overflow-y-auto pr-1">
              {headlines.slice(0, 8).map((item, i) => (
                <div
                  key={i}
                  className="group rounded-lg border border-border/70 bg-surface-2 px-4 py-3 transition-colors hover:border-primary/30"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0 flex-1">
                      <p className="text-[12px] font-semibold leading-snug text-foreground/90 line-clamp-2">
                        {item.headline}
                      </p>
                      <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
                        {item.source && (
                          <span className="inline-flex items-center rounded-full bg-surface-3 px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
                            {item.source}
                          </span>
                        )}
                        {item.key_drivers?.slice(0, 2).map((driver, j) => (
                          <span
                            key={j}
                            className="inline-flex items-center rounded-full bg-surface-3 px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-muted-foreground"
                          >
                            {driver.replace(/_/g, " ")}
                          </span>
                        ))}
                        {item.event_risk && (
                          <span className="inline-flex items-center gap-0.5 rounded-full bg-destructive/10 px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-destructive">
                            <AlertTriangle className="size-2.5" />
                            EVENT
                          </span>
                        )}
                      </div>
                    </div>
                    <div className="flex shrink-0 flex-col items-end gap-1">
                      <span
                        className={`font-mono text-xs font-extrabold tabular-nums ${sentimentColor(item.sentiment)}`}
                      >
                        {item.sentiment > 0 ? "+" : ""}
                        {item.sentiment.toFixed(2)}
                      </span>
                      <div className="flex items-center gap-1 text-muted-foreground">
                        <Clock className="size-2.5" />
                        <span className="text-[10px] font-semibold uppercase tracking-wider">
                          {horizonIcon(item.impact_horizon)}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Empty state */}
        {!hasHeadlines && (
          <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-border py-5">
            <Brain className="size-7 text-muted-foreground/30" />
            <span className="mt-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground/50">
              {emptyMessage(status, enabled, configured)}
            </span>
          </div>
        )}

        {/* Footer info */}
        {hasHeadlines && (
          <div className="rounded-lg border border-border bg-surface-2 px-4 py-3">
            <div className="flex items-start gap-2">
              <Zap className="size-3.5 shrink-0 mt-0.5 text-primary" />
              <p className="text-[11px] font-medium leading-relaxed text-muted-foreground">
                Sentiment scored via{" "}
                <span className="font-bold text-foreground">{providerLabel}</span>{" "}
                using{" "}
                <span className="font-bold text-foreground">{modelLabel}</span>{" "}
                with structured JSON output. Acts as both{" "}
                <span className="font-bold text-foreground">feature</span> and{" "}
                <span className="font-bold text-foreground">post-hoc filter</span>{" "}
                in the autonomous loop.
              </p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
