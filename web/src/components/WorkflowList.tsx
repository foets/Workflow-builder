'use client';

import React from 'react';
import { WorkflowFileSummary } from '@/types/workflow';

interface WorkflowListProps {
  workflows: WorkflowFileSummary[];
  onSelectWorkflow: (workflowId: string) => void;
  onExecuteWorkflow: (workflowId: string) => void;
  onDeleteWorkflow: (workflowId: string) => void;
  selectedWorkflowId: string | null;
}

export default function WorkflowList({ 
  workflows, 
  onSelectWorkflow, 
  onExecuteWorkflow,
  onDeleteWorkflow,
  selectedWorkflowId 
}: WorkflowListProps) {
  if (workflows.length === 0) {
    return (
      <div className="p-4 text-center text-[var(--text-secondary)]">
        <div className="text-3xl mb-3 opacity-40">📁</div>
        <p className="text-sm">No workflows saved yet.</p>
        <p className="text-xs mt-2 leading-relaxed">
          Describe an automation in the chat to create your first workflow.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-2 p-2 overflow-x-hidden">
      {workflows.map((wf) => (
        <div
          key={wf.id}
          className={`p-3 rounded-lg border cursor-pointer transition-all ${
            selectedWorkflowId === wf.id
              ? 'bg-blue-500/10 border-blue-500/50'
              : 'bg-[var(--bg-tertiary)] border-[var(--border)] hover:border-blue-500/30 hover:bg-[var(--bg-tertiary)]/80'
          }`}
          onClick={() => onSelectWorkflow(wf.id)}
        >
          <div className="flex-1 min-w-0">
            <h4 className="font-medium text-sm truncate text-[var(--text-primary)]">
              {wf.name}
            </h4>
            {wf.description && (
              <p className="text-xs text-[var(--text-secondary)] mt-1 line-clamp-2">
                {wf.description}
              </p>
            )}
          </div>
          
          <div className="mt-2 pt-2 border-t border-[var(--border)]">
            <div className="flex items-center justify-end gap-1 flex-wrap">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onSelectWorkflow(wf.id);
                }}
                className="text-[10px] px-2 py-1 bg-[var(--bg-secondary)] border border-[var(--border)] text-[var(--text-secondary)] hover:text-white hover:border-blue-500/40 rounded transition-colors"
                title="Open"
              >
                Open
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onExecuteWorkflow(wf.id);
                }}
                className="text-[10px] px-2 py-1 bg-emerald-600 hover:bg-emerald-700 text-white rounded transition-colors"
                title="Run"
              >
                ▶ Run
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDeleteWorkflow(wf.id);
                }}
                className="text-[10px] px-2 py-1 bg-red-600 hover:bg-red-700 text-white rounded transition-colors"
                title="Delete"
              >
                🗑
              </button>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}


