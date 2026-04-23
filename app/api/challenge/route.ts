import { NextResponse } from "next/server";

import { getMLServiceUrl } from "@/lib/ml-service";

export async function GET() {
  try {
    const response = await fetch(getMLServiceUrl("/api/auth/challenge-token"), {
      method: "GET",
      cache: "no-store",
    });

    const data = await response.json();

    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Challenge service unavailable";

    return NextResponse.json(
      {
        success: false,
        message,
      },
      { status: 500 }
    );
  }
}
