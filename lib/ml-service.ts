const ML_SERVICE_URL =
  process.env.PYTHON_ML_SERVICE_URL ?? "http://127.0.0.1:8000";

export function getMLServiceUrl(path: string) {
  return `${ML_SERVICE_URL}${path}`;
}
