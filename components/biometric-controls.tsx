"use client";

import { Button } from "@/components/ui/button";
import { RefreshCcw, ScanFace, UserPlus } from "lucide-react";

interface ControlsProps {
  isScanning: boolean;
  onEnroll: () => void;
  onReenroll: () => void;
  onLogin: () => void;
}

export default function BiometricControls({ isScanning, onEnroll, onReenroll, onLogin }: ControlsProps) {
  return (
    <div className="mt-2 grid w-full grid-cols-1 gap-3 sm:flex sm:flex-row sm:justify-center">
      <Button 
        onClick={onEnroll} 
        disabled={isScanning}
        className={`h-12 w-full transition-all sm:max-w-[190px] ${
          isScanning ? "cursor-not-allowed bg-slate-700/70 text-slate-300" : "border border-cyan-300/20 bg-cyan-500/90 text-slate-950 shadow-[0_10px_30px_rgba(34,211,238,0.22)] hover:bg-cyan-400"
        }`}
      >
        <UserPlus className="mr-2 h-5 w-5" />
        {isScanning ? "Scanning..." : "1. Enroll Face"}
      </Button>

      <Button
        onClick={onReenroll}
        disabled={isScanning}
        className={`h-12 w-full transition-all sm:max-w-[190px] ${
          isScanning ? "cursor-not-allowed bg-slate-700/70 text-slate-300" : "border border-amber-300/20 bg-amber-300/90 text-slate-950 shadow-[0_10px_30px_rgba(252,211,77,0.16)] hover:bg-amber-200"
        }`}
      >
        <RefreshCcw className="mr-2 h-5 w-5" />
        {isScanning ? "Scanning..." : "2. Re-enroll"}
      </Button>

      <Button 
        onClick={onLogin} 
        disabled={isScanning}
        className={`h-12 w-full transition-all sm:max-w-[190px] ${
          isScanning ? "cursor-not-allowed bg-slate-700/70 text-slate-300" : "border border-emerald-300/20 bg-emerald-400 text-slate-950 shadow-[0_10px_30px_rgba(74,222,128,0.2)] hover:bg-emerald-300"
        }`}
      >
        <ScanFace className="mr-2 h-5 w-5" />
        {isScanning ? "Scanning..." : "3. Login"}
      </Button>
    </div>
  );
}
