/**
 * Lightweight `filename::functionName` call tracing, used by the `debug-turn`
 * and `debug-seeds` commands (and anything else that opts in) to make the
 * end-to-end flow visible without a debugger.
 *
 * Enabled by `KORA_TRACE=1` (the debug commands set it themselves). Disabled
 * everywhere else, so `traceCall` is a no-op on the hot benchmark path.
 */

export function isTraceEnabled(): boolean {
  return process.env.KORA_TRACE === "1";
}

export function enableTrace(): void {
  process.env.KORA_TRACE = "1";
}

let depth = 0;

/** Log a single `file::function` step. */
export function traceCall(file: string, fn: string, detail?: string): void {
  if (!isTraceEnabled()) return;
  const indent = "  ".repeat(depth);
  const suffix = detail ? ` ${detail}` : "";
  console.error(`[trace] ${indent}${file}::${fn}${suffix}`);
}

/** Log entry/exit around an async step, indenting nested steps. */
export async function traceStep<T>(
  file: string,
  fn: string,
  body: () => Promise<T>,
  detail?: string
): Promise<T> {
  if (!isTraceEnabled()) return body();
  traceCall(file, fn, detail);
  depth++;
  try {
    return await body();
  } finally {
    depth--;
  }
}

/** Pretty-print a labelled JSON payload to stdout. */
export function printPayload(label: string, payload: unknown): void {
  console.log(`\n===== ${label} =====`);
  console.log(JSON.stringify(payload, null, 2));
  console.log(`===== end ${label} =====\n`);
}
