'use client';

import { useEffect, useRef, useState } from 'react';
import { ArrowRight, CheckCircle2, LoaderCircle, Radar, ScanFace, Shield, Sparkles } from 'lucide-react';

import BiometricWebcamArea from './biometric-webcam-area';
import { BiometricTelemetry } from './biometric-telemetry';
import BiometricControls from './biometric-controls';
import { BiometricFeedback } from './biometric-feedback';
import { BiometricHeader } from './biometric-header';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';

const ENROLL_CHALLENGES = ['blink', 'head_turn'];

const CHALLENGE_META: Record<string, { label: string; duration: number }> = {
  blink: { label: 'Blink your eyes clearly', duration: 3000 },
  head_turn: { label: 'Turn head left then right', duration: 4000 },
  smile: { label: 'Smile broadly', duration: 3000 },
  look_up: { label: 'Look up briefly', duration: 3000 },
};

function buildSteps(challenges: string[]) {
  return [
    { label: 'Hold still - scanning face...', duration: 2000 },
    ...challenges.map((c) => CHALLENGE_META[c] ?? { label: c, duration: 3000 }),
    { label: 'Hold still - BCG capture...', duration: 2000 },
  ];
}

type BcgResult = {
  bcg_hr_bpm: number;
  rppg_hr_bpm: number;
  bcg_signal_power: number;
  bcg_passed: boolean;
  freq_match: boolean;
  coherence_score: number;
  challenge_passed: boolean;
};

type AnalysisStep = {
  label: string;
  detail: string;
};

const ANALYSIS_STEPS: Record<'enroll' | 'reenroll' | 'login', AnalysisStep[]> = {
  enroll: [
    { label: 'Uploading capture', detail: 'Sending enrollment video to the secure ML service.' },
    { label: 'Extracting face template', detail: 'Building a stronger multi-frame identity descriptor.' },
    { label: 'Profiling pulse signature', detail: 'Deriving the enrolled vital baseline from the live clip.' },
    { label: 'Saving secure profile', detail: 'Writing the face + vital signature template to MongoDB.' },
  ],
  reenroll: [
    { label: 'Uploading capture', detail: 'Sending re-enrollment video to the secure ML service.' },
    { label: 'Refreshing face template', detail: 'Rebuilding the face descriptor from a fresh guided capture.' },
    { label: 'Refreshing pulse signature', detail: 'Replacing the stored vital baseline for this identity.' },
    { label: 'Updating secure profile', detail: 'Overwriting the previous biometric template in MongoDB.' },
  ],
  login: [
    { label: 'Uploading capture', detail: 'Sending login video for secure verification.' },
    { label: 'Checking challenge response', detail: 'Confirming blink and head-turn sequence against the issued prompt.' },
    { label: 'Verifying liveness', detail: 'Running rPPG, BCG, and anti-spoof checks.' },
    { label: 'Matching identity', detail: 'Comparing live face and vital signature against the enrolled profile.' },
  ],
};

export function BiometricAuth() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const stepTimersRef = useRef<ReturnType<typeof setTimeout>[]>([]);
  const analysisIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const analysisTimeoutsRef = useRef<ReturnType<typeof setTimeout>[]>([]);

  const [status, setStatus] = useState('Initializing camera...');
  const [isScanning, setIsScanning] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [scanResult, setScanResult] = useState<'success' | 'fail' | null>(null);
  const [username, setUsername] = useState('');
  const [usernameError, setUsernameError] = useState('');
  const [challengeStep, setChallengeStep] = useState<string | null>(null);
  const [bcgResult, setBcgResult] = useState<BcgResult | null>(null);
  const [pendingChallenges, setPendingChallenges] = useState<string[] | null>(null);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [analysisLogs, setAnalysisLogs] = useState<string[]>([]);
  const [analysisPhase, setAnalysisPhase] = useState<string>('');

  const challengeTokenRef = useRef<string>('');
  const activeChallengesRef = useRef<string[]>([]);
  const enrollChallengeKey = ENROLL_CHALLENGES.join('|');

  const runChallengeInstructions = (challenges: string[]) => {
    const steps = buildSteps(challenges);
    let elapsed = 0;
    setChallengeStep(steps[0].label);

    steps.forEach((step) => {
      const timer = setTimeout(() => {
        setChallengeStep(step.label);
        setStatus(step.label);
      }, elapsed);

      stepTimersRef.current.push(timer);
      elapsed += step.duration;
    });
  };

  const clearTimers = () => {
    stepTimersRef.current.forEach(clearTimeout);
    stepTimersRef.current = [];
    setChallengeStep(null);
  };

  const stopAnalysisTracking = () => {
    if (analysisIntervalRef.current) {
      clearInterval(analysisIntervalRef.current);
      analysisIntervalRef.current = null;
    }
    analysisTimeoutsRef.current.forEach(clearTimeout);
    analysisTimeoutsRef.current = [];
  };

  const startAnalysisTracking = (action: 'enroll' | 'reenroll' | 'login') => {
    stopAnalysisTracking();
    const steps = ANALYSIS_STEPS[action];
    setIsAnalyzing(true);
    setAnalysisProgress(8);
    setAnalysisPhase(steps[0].label);
    setAnalysisLogs([`Started: ${steps[0].detail}`]);

    steps.forEach((step, index) => {
      const timeout = setTimeout(() => {
        setAnalysisPhase(step.label);
        setAnalysisLogs((current) => {
          const next = [...current, `${step.label}: ${step.detail}`];
          return next.slice(-5);
        });
      }, index * 1400);
      analysisTimeoutsRef.current.push(timeout);
    });

    analysisIntervalRef.current = setInterval(() => {
      setAnalysisProgress((current) => (current >= 92 ? current : current + 7));
    }, 650);
  };

  const finishAnalysisTracking = (success: boolean) => {
    stopAnalysisTracking();
    setAnalysisProgress(100);
    setAnalysisPhase(success ? 'Verification complete' : 'Verification failed');
    setAnalysisLogs((current) => {
      const next = [...current, success ? 'Completed: all required checks finished.' : 'Stopped: one or more checks failed.'];
      return next.slice(-5);
    });
    setTimeout(() => {
      setIsAnalyzing(false);
      setAnalysisProgress(0);
      setAnalysisPhase('');
      setAnalysisLogs([]);
    }, 1400);
  };

  useEffect(() => {
    return () => {
      stopAnalysisTracking();
    };
  }, []);

  const fetchChallengeToken = async (): Promise<boolean> => {
    try {
      setStatus('Generating random challenges...');
      const res = await fetch('/api/challenge');
      const data = await res.json();

      if (!data.token || !data.challenges) {
        setStatus('Failed to get challenge token - try again.');
        return false;
      }

      challengeTokenRef.current = data.token;
      activeChallengesRef.current = data.challenges;
      setPendingChallenges(data.challenges);
      return true;
    } catch (err: any) {
      setStatus(`Server error: ${err.message}`);
      return false;
    }
  };

  const handleStartScan = async (action: 'enroll' | 'reenroll' | 'login') => {
    if (!username.trim()) {
      setUsernameError('Please enter a username');
      return;
    }

    setUsernameError('');

    if (!videoRef.current?.srcObject) {
      return;
    }

    if (action === 'login') {
      const ok = await fetchChallengeToken();
      if (!ok) {
        return;
      }
    } else {
      challengeTokenRef.current = '';
      activeChallengesRef.current = ENROLL_CHALLENGES;
    }

    setIsScanning(true);
    setIsAnalyzing(false);
    setScanResult(null);
    setBcgResult(null);
    setPendingChallenges(action !== 'login' ? ENROLL_CHALLENGES : null);
    chunksRef.current = [];

    setStatus('Preparing...');
    runChallengeInstructions(activeChallengesRef.current);

    const steps = buildSteps(activeChallengesRef.current);
    const duration = steps.reduce((sum, current) => sum + current.duration, 0);

    const stream = videoRef.current.srcObject as MediaStream;
    const recorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
    recorderRef.current = recorder;

    recorder.ondataavailable = (event) => {
      if (event.data?.size > 0) {
        chunksRef.current.push(event.data);
      }
    };

    recorder.onstop = () => {
      setIsScanning(false);
      clearTimers();
      setStatus('Analysing rPPG + challenges + BCG...');
      startAnalysisTracking(action);
      const blob = new Blob(chunksRef.current, { type: 'video/webm' });
      sendToBackend(blob, action);
    };

    recorder.start();

    setTimeout(() => {
      if (recorder.state === 'recording') {
        recorder.stop();
      }
    }, duration);
  };

  const sendToBackend = async (blob: Blob, action: 'enroll' | 'reenroll' | 'login') => {
    const formData = new FormData();
    formData.append('video', blob, 'rppg_sample.webm');
    formData.append('username', username.trim());

    if (action === 'login') {
      formData.append('challenge_token', challengeTokenRef.current);
    } else if (action === 'reenroll') {
      formData.append('force_reenroll', 'true');
    }

    try {
      const endpoint = action === 'login' ? '/api/login' : '/api/enroll';
      const res = await fetch(endpoint, { method: 'POST', body: formData });
      const data = await res.json().catch(async () => {
        const text = await res.text();
        throw new Error(`Non-JSON response (${res.status}): ${text}`);
      });

      if (data.success) {
        finishAnalysisTracking(true);
        setScanResult('success');

        if (action === 'login') {
          setBcgResult({
            bcg_hr_bpm: data.bcg_hr_bpm ?? 0,
            rppg_hr_bpm: data.rppg_hr_bpm ?? 0,
            bcg_signal_power: data.bcg_signal_power ?? 0,
            bcg_passed: data.bcg_passed ?? false,
            freq_match: data.bcg_freq_match ?? false,
            coherence_score: data.coherence_score ?? 0,
            challenge_passed: data.challenge_passed ?? false,
          });

          const bcgInfo = data.bcg_hr_bpm ? ` | BCG: ${data.bcg_hr_bpm} BPM` : '';
          setStatus(`Welcome ${username}! Liveness verified.${bcgInfo}`);
        } else {
          setStatus(
            action === 'reenroll'
              ? `Re-enrollment successful for ${username}. Profile updated.`
              : `Enrollment successful for ${username}. Face data stored.`
          );
        }
      } else {
        finishAnalysisTracking(false);
        setScanResult('fail');
        setStatus(data.message || data.spoof_reason || 'Access denied');
      }
    } catch (err: any) {
      finishAnalysisTracking(false);
      console.error('Backend error:', err);
      setStatus(`Error: ${err.message}`);
      setScanResult('fail');
    }
  };

  return (
    <div className="mx-auto grid w-full max-w-7xl gap-5 lg:grid-cols-[1.02fr_0.98fr] lg:gap-8">
      <section className="glass-panel relative order-2 overflow-hidden rounded-[28px] border border-slate-700/70 p-5 shadow-[0_30px_120px_rgba(0,0,0,0.45)] sm:p-7 lg:order-1 lg:rounded-[32px] lg:p-8">
        <div className="pointer-events-none absolute -left-16 top-8 h-40 w-40 rounded-full bg-emerald-400/10 blur-3xl" />
        <div className="pointer-events-none absolute -right-12 bottom-0 h-48 w-48 rounded-full bg-cyan-400/10 blur-3xl" />

        <div className="relative space-y-6">
          <BiometricHeader />

          <div className="flex flex-wrap gap-3">
            <div className="rounded-full border border-emerald-400/25 bg-emerald-400/12 px-4 py-2 text-xs font-medium uppercase tracking-[0.22em] text-emerald-200">
              Secure onboarding
            </div>
            <div className="rounded-full border border-cyan-400/25 bg-cyan-400/12 px-4 py-2 text-xs font-medium uppercase tracking-[0.22em] text-cyan-100">
              Replay-resistant auth
            </div>
          </div>

          <div className="grid gap-3 sm:grid-cols-3">
            <div className="rounded-2xl border border-slate-700/70 bg-slate-950/70 p-4">
              <div className="mb-2 flex items-center gap-2 text-emerald-300">
                <Radar className="h-4 w-4" />
                <span className="text-xs uppercase tracking-[0.2em]">Signal Layer</span>
              </div>
              <p className="text-sm leading-6 text-slate-200">Captures rPPG coherence across facial regions in real time.</p>
            </div>
            <div className="rounded-2xl border border-slate-700/70 bg-slate-950/70 p-4">
              <div className="mb-2 flex items-center gap-2 text-cyan-200">
                <Sparkles className="h-4 w-4" />
                <span className="text-xs uppercase tracking-[0.2em]">Challenge Layer</span>
              </div>
              <p className="text-sm leading-6 text-slate-200">Injects randomized prompts to reject prerecorded attacks.</p>
            </div>
            <div className="rounded-2xl border border-slate-700/70 bg-slate-950/70 p-4">
              <div className="mb-2 flex items-center gap-2 text-white">
                <Shield className="h-4 w-4 text-emerald-300" />
                <span className="text-xs uppercase tracking-[0.2em]">Identity Layer</span>
              </div>
              <p className="text-sm leading-6 text-slate-200">Matches live embeddings against enrolled biometric profiles.</p>
            </div>
          </div>
        </div>
      </section>

      <section className="glass-panel relative order-1 overflow-hidden rounded-[28px] border border-slate-700/70 p-4 shadow-[0_25px_90px_rgba(0,0,0,0.42)] sm:p-5 lg:order-2 lg:rounded-[32px] lg:p-6">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(45,212,191,0.05),transparent_30%)]" />

        <div className="relative space-y-4 sm:space-y-5">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between sm:gap-4">
            <div className="space-y-1">
              <div className="mb-2 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.26em] text-emerald-300/80">
                <ScanFace className="h-3.5 w-3.5" />
                VitalSign ID Console
              </div>
              <h2 className="max-w-xl text-xl font-semibold leading-tight text-white sm:text-2xl">
                Authenticate with live cardiovascular evidence
              </h2>
            </div>
            <div className="w-fit rounded-2xl border border-slate-700/70 bg-slate-950/80 px-3 py-2 text-left sm:text-right">
              <div className="text-[11px] uppercase tracking-[0.22em] text-slate-500">Mode</div>
              <div className="text-sm font-medium text-emerald-300">Face + rPPG</div>
            </div>
          </div>

          <div className="rounded-3xl border border-slate-700/70 bg-slate-950/80 p-4 sm:p-5">
            <div className="mb-3 flex items-center justify-between gap-3">
              <div>
                <Label htmlFor="username" className="text-[11px] font-semibold uppercase tracking-[0.24em] text-slate-300">
                  Identity label
                </Label>
                <p className="mt-1 text-xs text-slate-400">Use the same username for enrollment and authentication.</p>
              </div>
              <ArrowRight className="mt-1 h-4 w-4 text-emerald-300/80" />
            </div>

            <Input
              id="username"
              type="text"
              placeholder="Enter your username"
              value={username}
              onChange={(e) => {
                setUsername(e.target.value);
                setUsernameError('');
              }}
              disabled={isScanning}
              className={`h-12 rounded-2xl border-slate-700 bg-slate-900/95 text-base text-white placeholder:text-slate-500 ${usernameError ? 'border-red-500' : ''}`}
            />

            {usernameError && <p className="mt-2 text-sm text-red-400">{usernameError}</p>}
          </div>

          <BiometricWebcamArea videoRef={videoRef} status={status} setStatus={setStatus} />

          {pendingChallenges && !isScanning && !isAnalyzing && (
            <div className="rounded-3xl border border-cyan-400/25 bg-slate-950/90 px-4 py-4">
              <p className="mb-2 text-xs font-semibold uppercase tracking-[0.22em] text-cyan-200">
                {pendingChallenges.join('|') === enrollChallengeKey ? 'Enrollment capture sequence' : 'Randomized liveness sequence'}
              </p>
              <ol className="space-y-2">
                {pendingChallenges.map((challenge, index) => (
                  <li key={challenge} className="flex items-center gap-3 text-sm text-cyan-50">
                    <span className="flex h-6 w-6 items-center justify-center rounded-full border border-cyan-300/30 bg-cyan-300/15 text-[11px] font-semibold text-cyan-100">
                      {index + 1}
                    </span>
                    <span className="text-slate-100">{CHALLENGE_META[challenge]?.label ?? challenge}</span>
                  </li>
                ))}
              </ol>
              <p className="mt-3 text-xs text-slate-300">
                {pendingChallenges.join('|') === enrollChallengeKey
                  ? 'Enrollment uses guided motion so we capture a stronger multi-frame face and pulse template.'
                  : 'Start login to begin recording. Complete each prompt in sequence.'}
              </p>
            </div>
          )}

          {isScanning && challengeStep && (
            <div className="rounded-3xl border border-emerald-400/30 bg-slate-950/90 px-4 py-4 text-center">
              <p className="animate-pulse text-sm font-semibold text-emerald-300">{challengeStep}</p>
              <p className="mt-1 text-xs text-slate-300">
                Camera is actively recording rPPG, challenge compliance, and BCG motion.
              </p>
            </div>
          )}

          {isAnalyzing && (
            <div className="rounded-3xl border border-emerald-400/25 bg-slate-950/95 px-4 py-4">
              <div className="mb-3 flex items-center justify-between gap-3">
                <div className="flex items-center gap-2 text-emerald-300">
                  <LoaderCircle className="h-4 w-4 animate-spin" />
                  <span className="text-sm font-semibold">{analysisPhase || 'Running biometric checks'}</span>
                </div>
                <span className="text-xs font-semibold text-slate-300">{analysisProgress}%</span>
              </div>
              <div className="mb-4 h-2 overflow-hidden rounded-full bg-slate-800">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-cyan-400 via-emerald-300 to-emerald-400 transition-[width] duration-500"
                  style={{ width: `${analysisProgress}%` }}
                />
              </div>
              <div className="space-y-2">
                {analysisLogs.map((log, index) => (
                  <div key={`${log}-${index}`} className="flex items-start gap-2 text-xs text-slate-300">
                    <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-emerald-300" />
                    <span>{log}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <BiometricTelemetry isScanning={isScanning} bcgResult={bcgResult} />

          <BiometricControls
            isScanning={isScanning}
            onEnroll={() => handleStartScan('enroll')}
            onReenroll={() => handleStartScan('reenroll')}
            onLogin={() => handleStartScan('login')}
          />

          {scanResult && (
            <BiometricFeedback
              result={scanResult}
              message={status}
              onClose={() => {
                setScanResult(null);
                setBcgResult(null);
              }}
            />
          )}
        </div>
      </section>
    </div>
  );
}
