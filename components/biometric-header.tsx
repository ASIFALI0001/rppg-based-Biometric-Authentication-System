import { Activity, ShieldCheck, Sparkles } from 'lucide-react';

export function BiometricHeader() {
  return (
    <div className="flex flex-col gap-5">
      <div className="flex items-center gap-3">
        <div className="animate-pulse-glow flex h-12 w-12 items-center justify-center rounded-2xl border border-emerald-400/30 bg-emerald-400/10 text-emerald-300 shadow-[0_0_30px_rgba(45,212,191,0.18)]">
          <ShieldCheck className="h-6 w-6" />
        </div>
        <div>
          <div className="mb-1 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.28em] text-emerald-300/75">
            <Sparkles className="h-3.5 w-3.5" />
            VitalSign ID
          </div>
          <h1 className="text-2xl font-semibold tracking-tight text-white sm:text-3xl lg:text-4xl">
            Passive pulse biometrics with active liveness proof
          </h1>
        </div>
      </div>

      <div className="grid gap-3 text-sm text-slate-200 sm:grid-cols-3">
        <div className="rounded-2xl border border-slate-700/70 bg-slate-950/70 px-4 py-3">
          <div className="mb-1 font-medium text-white">rPPG Signal</div>
          <div className="text-xs leading-5 text-slate-300">Reads pulse signatures from facial color variation.</div>
        </div>
        <div className="rounded-2xl border border-slate-700/70 bg-slate-950/70 px-4 py-3">
          <div className="mb-1 font-medium text-white">Challenge Proof</div>
          <div className="text-xs leading-5 text-slate-300">Random motion prompts block replayed recordings.</div>
        </div>
        <div className="rounded-2xl border border-slate-700/70 bg-slate-950/70 px-4 py-3">
          <div className="mb-1 flex items-center gap-2 font-medium text-white">
            <Activity className="h-4 w-4 text-emerald-300" />
            Vital motion
          </div>
          <div className="text-xs leading-5 text-slate-300">BCG cross-check confirms sub-pixel live head motion.</div>
        </div>
      </div>
    </div>
  );
}
