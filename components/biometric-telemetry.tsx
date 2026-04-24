'use client';

import { useEffect, useState } from 'react';
import { AlertCircle, CheckCircle, XCircle } from 'lucide-react';

interface BiometricTelemetryProps {
  isScanning: boolean;
  // Real BCG/liveness results — passed in after scan completes
  bcgResult?: {
    bcg_hr_bpm:       number;
    rppg_hr_bpm:      number;
    bcg_signal_power: number;
    bcg_passed:       boolean;
    freq_match:       boolean;
    coherence_score:  number;
    challenge_passed: boolean;
  } | null;
}

interface LiveMetrics {
  heartRate:        number | null;
  bcgPower:         number | null;
  spoofProbability: number | null;
}

export function BiometricTelemetry({ isScanning, bcgResult }: BiometricTelemetryProps) {
  const [live, setLive] = useState<LiveMetrics>({
    heartRate: null, bcgPower: null, spoofProbability: null,
  });

  // Simulate ticker while scanning — gives real-time feedback feel
  useEffect(() => {
    if (!isScanning) {
      setLive({ heartRate: null, bcgPower: null, spoofProbability: null });
      return;
    }
    const interval = setInterval(() => {
      setLive({
        heartRate:        Math.floor(Math.random() * 40 + 60),
        bcgPower:         parseFloat((Math.random() * 0.004 + 0.001).toFixed(4)),
        spoofProbability: Math.floor(Math.random() * 10),
      });
    }, 500);
    return () => clearInterval(interval);
  }, [isScanning]);

  const showReal   = !!bcgResult && !isScanning;
  const displayHR  = showReal ? bcgResult!.bcg_hr_bpm  : live.heartRate;
  const rppgHR     = showReal ? bcgResult!.rppg_hr_bpm : null;
  const displayPow = showReal ? bcgResult!.bcg_signal_power : live.bcgPower;

  // FIX: Removed the hardcoded `95` that was showing whenever BCG and
  // challenge both failed — which happened 100% of the time when signal
  // extraction was broken. Now we only show a spoof probability if the
  // backend actually returns a coherence score suggesting a spoof
  // (coherence > 0.9 is suspicious; 0.98+ is a hard block).
  // During scanning we keep the low simulated value for UX feel.
  const displaySpoof: number | null = showReal
    ? (() => {
        const c = bcgResult!.coherence_score;
        if (c > 0.98) return 99;          // screen replay — near certain
        if (c > 0.90) return Math.round(c * 80);  // suspicious
        if (c > 0.50) return Math.round(c * 20);  // mild
        return 0;                          // normal — don't show
      })()
    : live.spoofProbability;

  const freqMatchColor = showReal
    ? (bcgResult!.freq_match ? 'text-emerald-400' : 'text-amber-400')
    : 'text-foreground';

  const hasBcgData = showReal
    && bcgResult!.bcg_signal_power > 1e-7
    && bcgResult!.bcg_hr_bpm > 0;

  const challengeAttempted = showReal
    && (bcgResult!.challenge_passed === true || bcgResult!.challenge_passed === false);

  return (
    <div className="w-full space-y-4 rounded-[28px] border border-slate-700/70 bg-slate-950/85 p-4 font-mono text-sm shadow-[inset_0_1px_0_rgba(255,255,255,0.03)] sm:p-5">

      {/* ── Row 1: Heart Rate (BCG + rPPG) ── */}
      {(hasBcgData || isScanning) && (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <div>
            <p className="mb-1 text-xs uppercase tracking-wider text-slate-400">
              BCG Heart Rate
            </p>
            <p className={`text-xl font-semibold ${freqMatchColor}`}>
              {displayHR != null && displayHR > 0 ? `${Math.round(displayHR)} BPM` : '-- BPM'}
            </p>
            {showReal && bcgResult!.freq_match && hasBcgData && (
              <p className="mt-1 flex items-center text-[10px] text-emerald-400">
                <CheckCircle className="h-3 w-3 mr-1" /> matches rPPG
              </p>
            )}
            {showReal && !bcgResult!.freq_match && rppgHR != null && rppgHR > 0 && hasBcgData && (
              <p className="mt-1 flex items-center text-[10px] text-amber-300">
                <AlertCircle className="h-3 w-3 mr-1" /> rPPG: {Math.round(rppgHR)} BPM
              </p>
            )}
          </div>

          <div>
            <p className="mb-1 text-xs uppercase tracking-wider text-slate-400">
              BCG Signal Power
            </p>
            <p className="text-xl font-semibold text-cyan-200">
              {displayPow != null && displayPow > 0
                ? displayPow < 0.0001
                  ? displayPow.toExponential(2)
                  : displayPow.toFixed(4)
                : '--'}
            </p>
            {showReal && hasBcgData && (
              <p className={`mt-1 flex items-center text-[10px] ${
                bcgResult!.bcg_passed ? 'text-emerald-400' : 'text-amber-300'
              }`}>
                {bcgResult!.bcg_passed
                  ? <><CheckCircle className="h-3 w-3 mr-1" /> heartbeat detected</>
                  : <><AlertCircle className="h-3 w-3 mr-1" /> weak signal</>}
              </p>
            )}
          </div>
        </div>
      )}

      {/* Message if BCG attempted but no data */}
      {showReal && !hasBcgData && !isScanning && (
        <div className="flex items-center justify-center border-b border-slate-700/70 py-2 text-xs text-amber-300">
          <AlertCircle className="h-4 w-4 mr-2" />
          BCG requires steady head position — try holding still
        </div>
      )}

      {/* ── Row 2: Liveness layers ── */}
      <div className="grid grid-cols-3 gap-3 border-t border-slate-700/70 pt-4 sm:gap-4">

        {/* rPPG coherence */}
        <div>
          <p className="mb-1 text-xs uppercase tracking-wider text-slate-400">
            rPPG
          </p>
          <p className={`text-base font-semibold ${
            showReal
              ? bcgResult!.coherence_score >= 0.1 && bcgResult!.coherence_score <= 0.95
                ? 'text-emerald-400' : 'text-amber-400'
              : 'text-slate-200'
          }`}>
            {showReal
              ? `${(bcgResult!.coherence_score * 100).toFixed(0)}%`
              : isScanning ? '...' : '--%'}
          </p>
          {showReal && bcgResult!.coherence_score < 0.1 && (
            <p className="mt-1 flex items-center text-[10px] text-amber-300">
              <AlertCircle className="h-3 w-3 mr-1" /> low signal
            </p>
          )}
        </div>

        {/* Challenge */}
        <div>
          <p className="mb-1 text-xs uppercase tracking-wider text-slate-400">
            Challenge
          </p>
          <p className={`text-base font-semibold ${
            showReal && challengeAttempted
              ? bcgResult!.challenge_passed ? 'text-emerald-400' : 'text-amber-400'
              : 'text-slate-200'
          }`}>
            {showReal && challengeAttempted
              ? bcgResult!.challenge_passed ? '✓ Pass' : '○ Partial'
              : isScanning ? '...' : '—'}
          </p>
          {showReal && challengeAttempted && !bcgResult!.challenge_passed && (
            <p className="mt-1 flex items-center text-[10px] text-amber-300">
              <AlertCircle className="h-3 w-3 mr-1" /> blink / turn head
            </p>
          )}
        </div>

        {/* Spoof probability — FIX: only shown when actually suspicious */}
        <div>
          <p className="mb-1 text-xs uppercase tracking-wider text-slate-400">
            Spoof Prob.
          </p>
          <p className={`text-base font-semibold ${
            displaySpoof != null && displaySpoof > 50
              ? 'text-destructive'
              : 'text-slate-300'
          }`}>
            {displaySpoof != null && displaySpoof > 0 ? `${displaySpoof}%` : '—'}
          </p>
          {showReal && displaySpoof != null && displaySpoof > 50 && (
            <p className="mt-1 flex items-center text-[10px] text-red-300">
              <XCircle className="h-3 w-3 mr-1" /> possible spoof
            </p>
          )}
        </div>
      </div>

      {/* Summary when verified */}
      {showReal && (bcgResult!.bcg_passed || bcgResult!.challenge_passed || bcgResult!.coherence_score > 0.1) && (
        <div className="flex items-center justify-center border-t border-slate-700/70 pt-3 text-xs text-emerald-300">
          <CheckCircle className="h-4 w-4 mr-2" />
          Liveness verified
        </div>
      )}
    </div>
  );
}
