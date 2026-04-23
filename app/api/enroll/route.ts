import { NextResponse } from "next/server";

import { connectDB } from "@/lib/mongodb";
import { getMLServiceUrl } from "@/lib/ml-service";
import User from "@/models/User";

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const username = String(formData.get("username") ?? "").trim();
    const video = formData.get("video");

    if (!username) {
      return NextResponse.json(
        { success: false, message: "Username required" },
        { status: 400 }
      );
    }

    if (!(video instanceof File)) {
      return NextResponse.json(
        { success: false, message: "Video file required" },
        { status: 400 }
      );
    }

    const pythonForm = new FormData();
    pythonForm.append("file", video, video.name || "rppg_sample.webm");

    const mlResponse = await fetch(getMLServiceUrl("/api/ml/analyze"), {
      method: "POST",
      body: pythonForm,
      cache: "no-store",
    });

    const mlData = await mlResponse.json();

    if (!mlResponse.ok || !mlData.success) {
      return NextResponse.json(
        {
          success: false,
          message:
            mlData.message ||
            mlData.spoof_reason ||
            "Enrollment analysis failed",
        },
        { status: mlResponse.status || 400 }
      );
    }

    if (!Array.isArray(mlData.embedding) || mlData.embedding.length === 0) {
      return NextResponse.json(
        { success: false, message: "ML service did not return an embedding" },
        { status: 500 }
      );
    }

    await connectDB();
    await User.findOneAndUpdate(
      { username },
      { username, embedding: mlData.embedding },
      { upsert: true, new: true, setDefaultsOnInsert: true }
    );

    return NextResponse.json({
      success: true,
      message: `Enrollment successful for ${username}. Face data stored.`,
      username,
      embeddingLength: mlData.embedding.length,
      coherence_score: mlData.coherence_score ?? 0,
    });
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Enrollment request failed";

    return NextResponse.json(
      { success: false, message },
      { status: 500 }
    );
  }
}
