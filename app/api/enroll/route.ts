import { NextResponse } from "next/server";

import { connectDB } from "@/lib/mongodb";
import { getMLServiceUrl } from "@/lib/ml-service";
import User from "@/models/User";

const TEMPLATE_VERSION = 2;

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

    const mlResponse = await fetch(getMLServiceUrl("/api/ml/enroll-secure"), {
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

    if (
      !Array.isArray(mlData.pulse_signature) ||
      mlData.pulse_signature.length === 0
    ) {
      return NextResponse.json(
        {
          success: false,
          message: "Vital signal profile could not be extracted. Please re-enroll with a live video.",
        },
        { status: 400 }
      );
    }

    await connectDB();
    await User.findOneAndUpdate(
      { username },
      {
        username,
        templateVersion: TEMPLATE_VERSION,
        faceEmbedding: mlData.embedding,
        pulseSignature: mlData.pulse_signature,
        rppgHrBpm: mlData.rppg_hr_bpm ?? 0,
        bcgHrBpm: mlData.bcg_hr_bpm ?? 0,
        enrollmentCoherence: mlData.coherence_score ?? 0,
      },
      { upsert: true, new: true, setDefaultsOnInsert: true }
    );

    return NextResponse.json({
      success: true,
      message: `Enrollment successful for ${username}. Face and vital signature stored.`,
      username,
      embeddingLength: mlData.embedding.length,
      coherence_score: mlData.coherence_score ?? 0,
      rppg_hr_bpm: mlData.rppg_hr_bpm ?? 0,
      bcg_hr_bpm: mlData.bcg_hr_bpm ?? 0,
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
