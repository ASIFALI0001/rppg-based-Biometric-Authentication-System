import { Alert, AlertDescription } from '@/components/ui/alert';
import { CheckCircle2, XCircle, X } from 'lucide-react';

interface BiometricFeedbackProps {
  result: 'success' | 'fail';
  message: string;
  onClose: () => void;
}

export function BiometricFeedback({ result, message, onClose }: BiometricFeedbackProps) {
  if (result === 'success') {
    return (
      <Alert className="rounded-2xl border-emerald-400/25 bg-emerald-500/10 text-emerald-50">
        <CheckCircle2 className="h-5 w-5 text-emerald-400" />
        <AlertDescription className="ml-2 text-emerald-100">
          {message}
        </AlertDescription>
        <button 
          onClick={onClose} 
          className="absolute top-2 right-2 text-emerald-400 hover:opacity-70"
        >
          <X className="h-4 w-4" />
        </button>
      </Alert>
    );
  }
  
  if (result === 'fail') {
    return (
      <Alert className="rounded-2xl border-red-400/25 bg-red-500/10 text-red-50">
        <XCircle className="h-5 w-5 text-red-400" />
        <AlertDescription className="ml-2 text-red-100">
          {message}
        </AlertDescription>
        <button 
          onClick={onClose} 
          className="absolute top-2 right-2 text-red-400 hover:opacity-70"
        >
          <X className="h-4 w-4" />
        </button>
      </Alert>
    );
  }
  
  return null;
}
