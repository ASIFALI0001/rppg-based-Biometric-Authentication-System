import { NextResponse } from "next/server";

import { connectDB } from "@/lib/mongodb";
import { getMLServiceUrl } from "@/lib/ml-service";
import User from "@/models/User";

function cosineSimilarity(a: number[], b: number[]) {
  if (a.length === 0 || a.length !== b.length) {
    return 0;
  }

  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  if (normA === 0 || normB === 0) {
    return 0;
  }

  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

const SIMILARITY_THRESHOLD = 0.75;

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const username = String(formData.get("username") ?? "").trim();
    const challengeToken = String(formData.get("challenge_token") ?? "");
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

    if (!challengeToken) {
      return NextResponse.json(
        { success: false, message: "Challenge token required" },
        { status: 400 }
      );
    }

    await connectDB();
    const user = await User.findOne({ username }).lean();

    if (!user) {
      return NextResponse.json(
        {
          success: false,
          message: `User '${username}' not found. Please enroll first.`,
        },
        { status: 404 }
      );
    }

    const pythonForm = new FormData();
    pythonForm.append("file", video, video.name || "rppg_sample.webm");
    pythonForm.append("challenge_token", challengeToken);

    const mlResponse = await fetch(getMLServiceUrl("/api/ml/analyze-full"), {
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
            "Login analysis failed",
          challenge_passed: mlData.challenge_passed ?? false,
          bcg_passed: mlData.bcg_passed ?? false,
          bcg_hr_bpm: mlData.bcg_hr_bpm ?? 0,
          rppg_hr_bpm: mlData.rppg_hr_bpm ?? 0,
          bcg_signal_power: mlData.bcg_signal_power ?? 0,
          coherence_score: mlData.coherence_score ?? 0,
          bcg_freq_match: mlData.bcg_freq_match ?? false,
        },
        { status: mlResponse.status || 401 }
      );
    }

    if (!Array.isArray(mlData.embedding) || mlData.embedding.length === 0) {
      return NextResponse.json(
        { success: false, message: "ML service did not return an embedding" },
        { status: 500 }
      );
    }

    const similarity = cosineSimilarity(
      user.embedding as number[],
      mlData.embedding as number[]
    );

    if (similarity < SIMILARITY_THRESHOLD) {
      return NextResponse.json(
        {
          success: false,
          message: `Face mismatch (similarity=${similarity.toFixed(2)})`,
          face_similarity: Number(similarity.toFixed(4)),
          challenge_passed: mlData.challenge_passed ?? false,
          bcg_passed: mlData.bcg_passed ?? false,
          bcg_hr_bpm: mlData.bcg_hr_bpm ?? 0,
          rppg_hr_bpm: mlData.rppg_hr_bpm ?? 0,
          bcg_signal_power: mlData.bcg_signal_power ?? 0,
          coherence_score: mlData.coherence_score ?? 0,
          bcg_freq_match: mlData.bcg_freq_match ?? false,
        },
        { status: 401 }
      );
    }

    return NextResponse.json({
      success: true,
      message: `Welcome ${username}! Liveness verified.`,
      username,
      face_similarity: Number(similarity.toFixed(4)),
      challenge_passed: mlData.challenge_passed ?? false,
      bcg_passed: mlData.bcg_passed ?? false,
      bcg_hr_bpm: mlData.bcg_hr_bpm ?? 0,
      rppg_hr_bpm: mlData.rppg_hr_bpm ?? 0,
      bcg_signal_power: mlData.bcg_signal_power ?? 0,
      coherence_score: mlData.coherence_score ?? 0,
      bcg_freq_match: mlData.bcg_freq_match ?? false,
    });
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Login request failed";

    return NextResponse.json(
      { success: false, message },
      { status: 500 }
    );
  }
}
