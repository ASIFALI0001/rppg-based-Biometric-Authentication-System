"use client";

import { useEffect, RefObject } from "react";

interface WebcamProps {
  videoRef: RefObject<HTMLVideoElement | null>;
  status: string;
  setStatus: (status: string) => void;
}

export default function BiometricWebcamArea({ videoRef, status, setStatus }: WebcamProps) {
  
  // Turn on the camera when this component loads
  useEffect(() => {
    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, frameRate: { ideal: 30 } },
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setStatus("Align face and click Start Scan");
      } catch (err) {
        setStatus("Camera access denied.");
        console.error(err);
      }
    }
    
    startCamera();

    // Cleanup: Turn off camera when user leaves the page
    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [videoRef, setStatus]);

  return (
    <div className="relative w-full min-h-[25rem] overflow-hidden rounded-[28px] border border-slate-700/70 bg-slate-950 shadow-[inset_0_1px_0_rgba(255,255,255,0.04),0_12px_50px_rgba(0,0,0,0.35)] sm:min-h-[28rem] lg:min-h-0 lg:aspect-video">
      {/* The actual video feed */}
      <video
        ref={videoRef as any}
        autoPlay
        playsInline
        muted
        className="h-full w-full scale-x-[-1] object-cover"
        /* scale-x-[-1] mirrors the video so it feels like a mirror */
      />

      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(45,212,191,0.08),transparent_35%),linear-gradient(180deg,transparent_0%,rgba(3,7,18,0.28)_100%)]" />
      <div className="scan-animation absolute left-0 right-0 h-px bg-gradient-to-r from-transparent via-emerald-300 to-transparent shadow-[0_0_18px_rgba(110,231,183,0.95)]" />

      {/* Dashed alignment box overlay */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        <div className="h-[18rem] w-[13rem] rounded-[999px] border border-dashed border-emerald-300/45 shadow-[0_0_60px_rgba(45,212,191,0.1)] sm:h-80 sm:w-56 lg:h-64 lg:w-48" />
      </div>

      {/* Status Text Overlay */}
      <div className="absolute bottom-4 left-3 right-3 text-center pointer-events-none sm:left-0 sm:right-0">
        <span className="inline-flex max-w-full rounded-full border border-emerald-300/20 bg-slate-950/92 px-4 py-2 text-center font-mono text-xs text-emerald-300 backdrop-blur-md sm:text-sm">
          {status}
        </span>
      </div>
    </div>
  );
}
