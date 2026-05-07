import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { test, expect } from "@playwright/test";

const __dirname = dirname(fileURLToPath(import.meta.url));
const mockPath = join(__dirname, "fixtures", "dashboard-mock.json");
const mockJson = readFileSync(mockPath, "utf-8");

test.describe("dashboard capture", () => {
  test("full-page bento screenshot (mocked API)", async ({ page }) => {
    await page.route("**/api/dashboard", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: mockJson,
      });
    });
    await page.route("**/api/dashboard/refresh", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: "{}",
      });
    });
    await page.route("**/api/config", async (route) => {
      if (route.request().method() === "PUT") {
        await route.fulfill({ status: 200, body: "{}" });
        return;
      }
      await route.continue();
    });

    await page.goto("/", { waitUntil: "networkidle" });
    await expect(page.getByRole("heading", { name: /Gold Agent/i })).toBeVisible();

    await page.screenshot({
      path: join(__dirname, "screenshots", "bento-fullpage.png"),
      fullPage: true,
    });
  });
});
