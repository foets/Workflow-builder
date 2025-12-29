'use client';

import React from 'react';
import type { AuthStatus as AuthStatusType } from '@/types/workflow';

interface AuthStatusProps {
  authStatus: AuthStatusType | null;
  onCheckAgain: () => void;
  onExecute?: () => void;
}

export default function AuthStatus({ authStatus, onCheckAgain, onExecute }: AuthStatusProps) {
  if (!authStatus) return null;

  const entries = Object.entries(authStatus.toolkits || {});
  if (entries.length === 0) return null;

  return (
    <div className="p-3 border-b border-[var(--border)] bg-[var(--bg-secondary)]">
      <div className="flex items-center justify-between gap-3 mb-2">
        <h3 className="text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wide">
          Required Connections
        </h3>

        <div className="flex items-center gap-2">
          <button
            onClick={onCheckAgain}
            className="text-[10px] px-2 py-1 rounded bg-[var(--bg-tertiary)] border border-[var(--border)] text-[var(--text-secondary)] hover:text-white hover:border-blue-500/40 transition-colors"
          >
            Check Again
          </button>
          {authStatus.all_connected && onExecute && (
            <button
              onClick={onExecute}
              className="text-[10px] px-2 py-1 rounded bg-emerald-600 hover:bg-emerald-700 text-white transition-colors"
            >
              ▶ Execute
            </button>
          )}
        </div>
      </div>

      <div className="space-y-2">
        {entries.map(([toolkit, status]) => (
          <div
            key={toolkit}
            className="flex items-center justify-between gap-3 p-2 rounded-lg bg-[var(--bg-tertiary)] border border-[var(--border)]"
          >
            <div className="min-w-0">
              <div className="text-sm font-medium text-[var(--text-primary)] truncate">
                {status.label || toolkit}
              </div>
              {status.error && (
                <div className="text-[11px] text-amber-300 mt-0.5">
                  {status.error}
                </div>
              )}
            </div>

            <div className="flex items-center gap-2 flex-shrink-0">
              {status.connected ? (
                <span className="text-[11px] text-emerald-400">Connected</span>
              ) : status.connect_url ? (
                <a
                  href={status.connect_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-[11px] px-2 py-1 rounded bg-blue-600 hover:bg-blue-700 text-white transition-colors"
                >
                  Connect
                </a>
              ) : (
                <button
                  onClick={onCheckAgain}
                  className="text-[11px] px-2 py-1 rounded bg-blue-600 hover:bg-blue-700 text-white transition-colors"
                  title="Fetch a Connect link"
                >
                  Connect
                </button>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}


