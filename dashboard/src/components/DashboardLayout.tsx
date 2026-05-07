import { type ReactNode } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Activity, Radio, RefreshCw, Shield, Zap } from "lucide-react";

interface Props {
  children: ReactNode;
  wsStatus: "connecting" | "connected" | "disconnected";
  onRefresh: () => void;
  loading: boolean;
  shadowMode: boolean;
}

const wsDotColor = {
  connected: "bg-long",
  connecting: "bg-volatile animate-pulse-slow",
  disconnected: "bg-short",
};

const wsLabel = {
  connected: "LIVE",
  connecting: "SYNCING",
  disconnected: "OFFLINE",
};

export default function DashboardLayout({
  children,
  wsStatus,
  onRefresh,
  loading,
  shadowMode,
}: Props) {
  return (
    <div className="box-border flex min-h-dvh min-h-screen w-full flex-col bg-background">
      {/* ── Header ─────────────────────────────────────── */}
      <header className="sticky top-0 z-50 border-b border-border bg-background/92 backdrop-blur-xl">
        <div className="flex h-16 w-full items-center justify-between gap-3 px-1 sm:px-2">
          {/* Left: Brand */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2.5">
              <div className="flex h-9 w-9 items-center justify-center rounded-full border border-primary/20 bg-accent">
                <Zap className="size-4 text-primary" strokeWidth={2.5} />
              </div>
              <div className="flex flex-col">
                <h1 className="font-display text-2xl leading-none text-foreground">
                  Gold Agent
                </h1>
                <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-primary">
                  Autonomous Trading System
                </span>
              </div>
            </div>

            <div className="hidden sm:flex items-center gap-2 ml-2">
              <Badge variant="gold">XAUUSD</Badge>
              {shadowMode && <Badge variant="shadow">SHADOW</Badge>}
            </div>
          </div>

          {/* Right: Status + Actions */}
          <div className="flex items-center gap-4">
            {/* Connection indicator */}
            <div className="flex items-center gap-2 rounded-full border border-border bg-card px-3 py-1.5">
              <div
                className={cn(
                  "h-2 w-2 rounded-full",
                  wsDotColor[wsStatus],
                )}
              />
              <span className="text-[11px] font-bold uppercase tracking-widest text-muted-foreground">
                {wsLabel[wsStatus]}
              </span>
              <Radio className="h-3.5 w-3.5 text-muted-foreground" />
            </div>

            {/* Run Cycle Button */}
            <Button
              onClick={onRefresh}
              disabled={loading}
              variant={loading ? "secondary" : "default"}
              size="sm"
              className="gap-2"
            >
              <RefreshCw
                className={cn("h-3.5 w-3.5", loading && "animate-spin")}
              />
              {loading ? "Running" : "Run Cycle"}
            </Button>
          </div>
        </div>
      </header>

      {/* ── Sub-header bar ──────────────────────────────── */}
      <div className="border-b border-border bg-surface-1">
        <div className="flex h-8 w-full items-center gap-5 overflow-x-auto px-1 sm:px-2">
          <div className="flex items-center gap-1.5 text-muted-foreground">
            <Shield className="h-3.5 w-3.5" />
            <span className="text-[11px] font-semibold uppercase tracking-wider">
              {shadowMode ? "Shadow Mode Active" : "Live Mode"}
            </span>
          </div>
          <div className="h-4 w-px bg-border" />
          <div className="flex items-center gap-1.5 text-muted-foreground">
            <Activity className="h-3.5 w-3.5" />
            <span className="text-[11px] font-semibold uppercase tracking-wider">
              Regime-Adaptive · Multi-TF Ensemble · GenAI Sentiment
            </span>
          </div>
        </div>
      </div>

      {/* ── Main Content ────────────────────────────────── */}
      <main className="min-h-0 flex-1">
        <div className="w-full px-1 py-5 sm:px-2 sm:py-6">
          {children}
        </div>
      </main>

      {/* ── Footer ──────────────────────────────────────── */}
      <footer className="mt-auto border-t border-border bg-background">
        <div className="flex h-9 w-full items-center justify-between gap-2 overflow-x-auto px-1 sm:px-2">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground/80">
            Gold Trading Agent v0.1
          </span>
          <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground/80">
            ML4T · Perceive → Infer → Govern → Execute
          </span>
        </div>
      </footer>
    </div>
  );
}
