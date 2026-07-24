import { spawn } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
export interface BridgeCallContext {
  runId: string;
  stepId: string;
}

export interface BridgeResponse {
  status: "ok" | "error";
  protocol?: string;
  command?: string;
  data?: unknown;
  error?: {
    type?: string;
    message?: string;
  };
}

const bridgeRoot = (): string => {
  const currentDir = dirname(fileURLToPath(import.meta.url));
  const agentRoot = resolve(currentDir, "../..");
  return resolve(agentRoot, "..");
};

export const callThetaBridge = async (
  command: string,
  input: unknown,
  context: BridgeCallContext
): Promise<BridgeResponse> => {
  const cwd = bridgeRoot();
  const python = process.env.THETA_AGENT_BRIDGE_PYTHON || "python";
  const payload = JSON.stringify({
    command,
    input,
    context: {
      runId: context.runId,
      stepId: context.stepId
    }
  });

  return new Promise((resolvePromise) => {
    const child = spawn(python, ["-m", "theta_agent_bridge"], {
      cwd,
      stdio: ["pipe", "pipe", "pipe"],
      env: {
        ...process.env,
        PYTHONIOENCODING: "utf-8",
        PYTHONPATH: [cwd, process.env.PYTHONPATH].filter(Boolean).join(process.platform === "win32" ? ";" : ":")
      }
    });

    let stdout = "";
    let stderr = "";

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    child.on("error", (error) => {
      resolvePromise({
        status: "error",
        command,
        error: {
          type: error.name,
          message: error.message
        }
      });
    });
    child.on("close", (code) => {
      const trimmed = stdout.trim();
      if (!trimmed) {
        resolvePromise({
          status: "error",
          command,
          error: {
            type: "EmptyBridgeResponse",
            message: stderr || `Bridge exited with code ${code}`
          }
        });
        return;
      }

      try {
        resolvePromise(JSON.parse(trimmed) as BridgeResponse);
      } catch (error) {
        resolvePromise({
          status: "error",
          command,
          error: {
            type: "InvalidBridgeJson",
            message: error instanceof Error ? error.message : "Bridge returned invalid JSON"
          },
          data: {
            stdout: trimmed,
            stderr
          }
        });
      }
    });

    child.stdin.end(payload, "utf8");
  });
};
