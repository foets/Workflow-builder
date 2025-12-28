'use client';

import React from 'react';

export type NavigationProps = {
  onHomeClick?: () => void;
};

export default function Navigation({ onHomeClick }: NavigationProps) {
  return (
    <nav className="h-14 bg-[var(--bg-secondary)] border-b border-[var(--border)] flex items-center px-6 justify-between">
      <button
        type="button"
        onClick={onHomeClick}
        className="flex items-center gap-3 rounded-md focus:outline-none focus:ring-2 focus:ring-[var(--accent)] focus:ring-offset-2 focus:ring-offset-[var(--bg-secondary)]"
        title="Home"
      >
        <div className="text-2xl" aria-hidden="true">
          🔄
        </div>
        <h1 className="text-lg font-semibold hover:opacity-90">AI Workflow Builder</h1>
      </button>
      
      <div className="flex items-center gap-3">
        <span className="text-xs text-[var(--text-secondary)]">Demo Mode</span>
        <div className="w-2 h-2 rounded-full bg-[var(--success)]" />
      </div>
    </nav>
  );
}


