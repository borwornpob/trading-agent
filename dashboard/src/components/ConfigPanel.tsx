import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Settings, Shield, Grid3x3 } from "lucide-react";

interface Props {
  shadowMode?: boolean;
  gridEnabled?: boolean;
  onUpdate: () => void;
}

export default function ConfigPanel({
  shadowMode,
  gridEnabled,
  onUpdate,
}: Props) {
  const [gridOn, setGridOn] = useState(gridEnabled ?? true);

  const updateConfig = async (key: string, value: boolean | number) => {
    const body: Record<string, unknown> = {};
    body[key] = value;
    await fetch("/api/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    onUpdate();
  };

  return (
    <Card className="flex h-full min-h-0 flex-col">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Settings className="size-3.5 text-primary" />
            Configuration
          </CardTitle>
          <Badge variant="secondary" className="text-[10px]">
            LIVE TWEAK
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-5">
        {/* Shadow Mode */}
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-3 min-w-0">
            <div
              className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border ${
                shadowMode
                  ? "border-primary/30 bg-accent"
                  : "border-border bg-surface-2"
              }`}
            >
              <Shield
                className={`size-4 ${
                  shadowMode ? "text-primary" : "text-muted-foreground"
                }`}
                strokeWidth={2.5}
              />
            </div>
            <div className="min-w-0">
              <span className="block text-sm font-bold text-foreground">
                Shadow Mode
              </span>
              <span className="block text-[11px] text-muted-foreground truncate">
                {shadowMode
                  ? "Logging without orders"
                  : "Submitting live orders"}
              </span>
            </div>
          </div>
          <Button
            size="sm"
            variant={shadowMode ? "warning" : "outline"}
            onClick={() => updateConfig("shadow_mode", !shadowMode)}
            className="shrink-0 min-w-[72px]"
          >
            {shadowMode ? "ON" : "OFF"}
          </Button>
        </div>

        <Separator />

        {/* Smart Grid */}
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-3 min-w-0">
            <div
              className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border ${
                gridOn
                  ? "border-ranging/30 bg-ranging/10"
                  : "border-border bg-surface-2"
              }`}
            >
              <Grid3x3
                className={`size-4 ${
                  gridOn ? "text-ranging" : "text-muted-foreground"
                }`}
                strokeWidth={2.5}
              />
            </div>
            <div className="min-w-0">
              <span className="block text-sm font-bold text-foreground">
                Smart Grid
              </span>
              <span className="block text-[11px] text-muted-foreground truncate">
                {gridOn ? "S/R-based recovery enabled" : "Single entry only"}
              </span>
            </div>
          </div>
          <Button
            size="sm"
            variant={gridOn ? "default" : "outline"}
            onClick={() => {
              const next = !gridOn;
              setGridOn(next);
              updateConfig("grid_enabled", next);
            }}
            className="shrink-0 min-w-[72px]"
          >
            {gridOn ? "ON" : "OFF"}
          </Button>
        </div>

        <Separator />

        {/* Info footer */}
        <div className="rounded-lg border border-border bg-surface-2 px-4 py-3">
          <p className="text-[11px] font-medium leading-relaxed text-muted-foreground">
            Changes apply to the{" "}
            <span className="font-bold text-foreground">next cycle</span>. Grid
            uses predicted S/R levels with anti-martingale sizing.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
