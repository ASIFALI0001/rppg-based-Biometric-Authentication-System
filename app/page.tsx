import { BiometricAuth } from '@/components/biometric-auth';

export default function Home() {
  return (
    <main className="relative min-h-screen overflow-hidden bg-background px-4 py-10 sm:px-6 lg:px-8">
      <BiometricAuth />
    </main>
  );
}
