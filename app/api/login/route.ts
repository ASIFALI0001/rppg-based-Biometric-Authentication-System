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

const FACE_THRESHOLD = 0.70;            // was 0.90 — HOG+LBP varies across sessions
const PULSE_SIGNATURE_THRESHOLD = 0.60; // was 0.82 — FFT signature shifts with HR changes
const RPPG_HR_TOLERANCE = 25;           // was 18 BPM — wider window for HR variation
const BCG_HR_TOLERANCE = 25;            // was 18 BPM
const REQUIRED_TEMPLATE_VERSION = 2;

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

    if (
      (user.templateVersion ?? 1) < REQUIRED_TEMPLATE_VERSION ||
      !Array.isArray(user.faceEmbedding) ||
      !Array.isArray(user.pulseSignature) ||
      user.pulseSignature.length === 0
    ) {
      return NextResponse.json(
        {
          success: false,
          message:
            "This account was enrolled with an older insecure template. Please re-enroll using a fresh live video.",
        },
        { status: 409 }
      );
    }

    const pythonForm = new FormData();
    pythonForm.append("file", video, video.name || "rppg_sample.webm");
    pythonForm.append("challenge_token", challengeToken);

    const mlResponse = await fetch(getMLServiceUrl("/api/ml/verify-secure"), {
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

    if (
      !Array.isArray(mlData.pulse_signature) ||
      mlData.pulse_signature.length === 0
    ) {
      return NextResponse.json(
        { success: false, message: "ML service did not return a pulse signature" },
        { status: 500 }
      );
    }

    const faceSimilarity = cosineSimilarity(
      user.faceEmbedding as number[],
      mlData.embedding as number[]
    );

    const pulseSimilarity = cosineSimilarity(
      user.pulseSignature as number[],
      mlData.pulse_signature as number[]
    );

    const storedRppgHr = Number(user.rppgHrBpm ?? 0);
    const storedBcgHr = Number(user.bcgHrBpm ?? 0);
    const loginRppgHr = Number(mlData.rppg_hr_bpm ?? 0);
    const loginBcgHr = Number(mlData.bcg_hr_bpm ?? 0);
    const rppgHrDiff =
      storedRppgHr > 0 && loginRppgHr > 0
        ? Math.abs(storedRppgHr - loginRppgHr)
        : Number.POSITIVE_INFINITY;
    const bcgHrDiff =
      storedBcgHr > 0 && loginBcgHr > 0
        ? Math.abs(storedBcgHr - loginBcgHr)
        : Number.POSITIVE_INFINITY;

    const faceOk = faceSimilarity >= FACE_THRESHOLD;
    const pulseOk = pulseSimilarity >= PULSE_SIGNATURE_THRESHOLD;
    // Only enforce HR comparison when both enrolled and live readings are non-zero.
    // If BCG or rPPG was unavailable at enrollment/login, skip that check rather
    // than always failing (0 vs 0 would give Infinity diff).
    const rppgOk = Number.isFinite(rppgHrDiff) && rppgHrDiff <= RPPG_HR_TOLERANCE;
    const bcgOk =
      Number.isFinite(bcgHrDiff) &&
      bcgHrDiff <= BCG_HR_TOLERANCE &&
      (mlData.bcg_passed ?? false);
    const hasVitalData = Number.isFinite(rppgHrDiff) || Number.isFinite(bcgHrDiff);
    const vitalOk = !hasVitalData || rppgOk || bcgOk;

    if (!faceOk || !pulseOk || !vitalOk) {
      const reasons = [];
      if (!faceOk) reasons.push(`face similarity too low (${faceSimilarity.toFixed(3)})`);
      if (!pulseOk) reasons.push(`pulse signature mismatch (${pulseSimilarity.toFixed(3)})`);
      if (!vitalOk) reasons.push("vital signs do not match the enrolled profile");

      return NextResponse.json(
        {
          success: false,
          message: `Identity mismatch: ${reasons.join("; ")}`,
          face_similarity: Number(faceSimilarity.toFixed(4)),
          pulse_similarity: Number(pulseSimilarity.toFixed(4)),
          rppg_hr_diff: Number.isFinite(rppgHrDiff) ? Number(rppgHrDiff.toFixed(1)) : null,
          bcg_hr_diff: Number.isFinite(bcgHrDiff) ? Number(bcgHrDiff.toFixed(1)) : null,
          challenge_passed: mlData.challenge_passed ?? false,
          bcg_passed: mlData.bcg_passed ?? false,
          bcg_hr_bpm: loginBcgHr,
          rppg_hr_bpm: loginRppgHr,
          bcg_signal_power: mlData.bcg_signal_power ?? 0,
          coherence_score: mlData.coherence_score ?? 0,
          bcg_freq_match: mlData.bcg_freq_match ?? false,
        },
        { status: 401 }
      );
    }

    return NextResponse.json({
      success: true,
      message: `Welcome ${username}! Face and vital signature verified.`,
      username,
      face_similarity: Number(faceSimilarity.toFixed(4)),
      pulse_similarity: Number(pulseSimilarity.toFixed(4)),
      rppg_hr_diff: Number.isFinite(rppgHrDiff) ? Number(rppgHrDiff.toFixed(1)) : null,
      bcg_hr_diff: Number.isFinite(bcgHrDiff) ? Number(bcgHrDiff.toFixed(1)) : null,
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
