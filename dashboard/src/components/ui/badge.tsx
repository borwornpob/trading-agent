import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-bold uppercase tracking-wider transition-colors",
  {
    variants: {
      variant: {
        default:
          "border-primary/20 bg-primary text-primary-foreground",
        secondary:
          "border-border bg-secondary text-secondary-foreground",
        destructive:
          "border-destructive/20 bg-destructive text-destructive-foreground",
        outline:
          "text-foreground",
        gold:
          "border-primary/25 bg-accent text-primary",
        long:
          "border-long/25 bg-long/10 text-long",
        short:
          "border-short/25 bg-short/10 text-short",
        flat:
          "border-flat/25 bg-flat/10 text-flat",
        ranging:
          "border-ranging/25 bg-ranging/10 text-ranging",
        volatile:
          "border-volatile/25 bg-volatile/10 text-volatile",
        trending:
          "border-trending-up/25 bg-trending-up/10 text-trending-up",
        event:
          "border-destructive/25 bg-destructive/10 text-destructive animate-pulse",
        shadow:
          "border-primary/30 bg-accent text-primary",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <span className={cn(badgeVariants({ variant }), className)} {...props} />
  )
}

export { Badge, badgeVariants }
