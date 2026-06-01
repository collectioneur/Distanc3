type ToastListener = (message: string) => void;

let listener: ToastListener | null = null;

export function showToast(message: string): void {
  listener?.(message);
}

export function setToastListener(fn: ToastListener | null): void {
  listener = fn;
}
