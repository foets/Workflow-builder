'use client';

import React, { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface WorkflowBuilderProps {
  workflowId: string | null;
  content: string;
  workflowJson: any | null;
  onSave: (jsonText: string) => Promise<{ ok: true } | { ok: false; error: string }>;
  onExecute?: () => void;
  isExecuting?: boolean;
  stepCount?: number;
  currentExecutionStep?: number;
}

/**
 * WorkflowBuilder
 *
 * - View mode: renders Markdown (a VIEW of the workflow)
 * - Edit mode: edits Workflow JSON (the SOURCE OF TRUTH) and persists via API
 */
export default function WorkflowBuilder({ 
  workflowId,
  content,
  workflowJson,
  onSave,
  onExecute,
  isExecuting = false,
  stepCount = 0,
  currentExecutionStep = -1
}: WorkflowBuilderProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [editableJsonText, setEditableJsonText] = useState('');
  const [saveError, setSaveError] = useState<string | null>(null);

  // Sync with props
  useEffect(() => {
    if (!workflowJson) {
      setEditableJsonText('');
      return;
    }
    try {
      setEditableJsonText(JSON.stringify(workflowJson, null, 2));
    } catch {
      setEditableJsonText(String(workflowJson));
    }
  }, [workflowId, workflowJson]);

  // Save handler
  const handleSave = async () => {
    setSaveError(null);
    const res = await onSave(editableJsonText);
    if (res.ok) {
      setIsEditing(false);
      return;
    }
    setSaveError(res.error || 'Failed to save workflow.');
  };

  // Cancel handler
  const handleCancel = () => {
    if (workflowJson) {
      try {
        setEditableJsonText(JSON.stringify(workflowJson, null, 2));
      } catch {
        setEditableJsonText(String(workflowJson));
      }
    } else {
      setEditableJsonText('');
    }
    setSaveError(null);
    setIsEditing(false);
  };

  // Empty state
  if (!content && !workflowId) {
    return (
      <div className="h-full flex items-center justify-center bg-[var(--bg-primary)] p-6">
        <div className="text-center max-w-md">
          <div className="text-7xl mb-6 opacity-40">📝</div>
          <h2 className="text-xl font-semibold mb-4 text-[var(--text-primary)]">
            No Workflow Selected
          </h2>
          <p className="text-[var(--text-secondary)] mb-6 leading-relaxed">
            Describe your automation in the chat, or select a saved workflow from the sidebar.
          </p>
          <div className="text-left bg-[var(--bg-secondary)] rounded-xl p-5 border border-[var(--border)]">
            <p className="text-sm font-medium text-[var(--text-primary)] mb-3">Try saying:</p>
            <ul className="text-sm text-[var(--text-secondary)] space-y-2">
              <li>• "Create a workflow to extract emails and save to docs"</li>
              <li>• "Build an automation that summarizes Slack messages"</li>
              <li>• "Set up a flow to sync CRM contacts to a spreadsheet"</li>
            </ul>
          </div>
        </div>
      </div>
    );
  }

  const workflowName = (workflowJson && typeof workflowJson?.name === 'string' && workflowJson.name) || workflowId || 'Untitled';

  return (
    <div className="h-full flex flex-col bg-[var(--bg-primary)]">
      {/* Header */}
      <div className="flex-shrink-0 p-4 border-b border-[var(--border)] bg-[var(--bg-secondary)]">
        <div className="flex items-center justify-between gap-4">
          {/* Title and file info */}
          <div className="flex-1 min-w-0">
            <h2 className="font-semibold text-lg text-[var(--text-primary)] truncate">
              {workflowName}
            </h2>
            {workflowId && (
              <p className="text-xs text-[var(--text-secondary)] mt-0.5">
                📁 workflows/{workflowId}.json
              </p>
            )}
        </div>
          
          {/* Action buttons */}
          <div className="flex items-center gap-2 flex-shrink-0">
            {!isExecuting && (
              <>
                {isEditing ? (
                  <>
                    <button
                      onClick={handleCancel}
                      className="px-3 py-1.5 text-sm text-[var(--text-secondary)] hover:text-white transition-colors"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleSave}
                      className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors"
                    >
                      💾 Save
                    </button>
                  </>
                ) : (
                  <>
            <button
                      onClick={() => {
                        setSaveError(null);
                        setIsEditing(true);
                      }}
                      className="px-3 py-1.5 text-sm text-[var(--text-secondary)] hover:text-[var(--accent)] hover:bg-[var(--accent)]/10 rounded-lg transition-colors"
            >
                      ✏️ Edit
            </button>
                    {onExecute && (
            <button
              onClick={onExecute}
                        className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg text-sm font-medium transition-colors flex items-center gap-2"
            >
                        ▶ Run
            </button>
          )}
                  </>
                )}
              </>
            )}
            
            {isExecuting && (
              <div className="flex items-center gap-2 text-blue-400">
                <span className="animate-spin">⚡</span>
                <span className="text-sm font-medium">Executing...</span>
              </div>
            )}
        </div>
      </div>

        {/* Progress bar during execution */}
        {isExecuting && stepCount > 0 && currentExecutionStep >= 0 && (
          <div className="mt-3">
            <div className="h-1.5 bg-[var(--bg-tertiary)] rounded-full overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-blue-500 to-emerald-500 transition-all duration-500"
                style={{ width: `${((currentExecutionStep + 1) / stepCount) * 100}%` }}
              />
            </div>
            <p className="text-xs text-[var(--text-secondary)] mt-1">
              Step {currentExecutionStep + 1} of {stepCount}
            </p>
          </div>
        )}
          </div>

      {/* Content Area */}
      <div className="flex-1 overflow-y-auto">
        {isEditing ? (
          /* Edit Mode */
          <div className="h-full p-4">
            <div className="mb-3 text-xs text-[var(--text-secondary)] flex items-center gap-2">
              <span>📝</span>
              <span>Edit workflow JSON (source of truth). Saved immediately when you click Save.</span>
            </div>
            {saveError && (
              <div className="mb-3 text-xs text-red-300 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
                {saveError}
              </div>
            )}
            <textarea
              value={editableJsonText}
              onChange={(e) => setEditableJsonText(e.target.value)}
              className="w-full h-[calc(100%-2.5rem)] p-4 bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg 
                         text-sm font-mono text-[var(--text-primary)] resize-none focus:outline-none focus:border-[var(--accent)]
                         leading-relaxed"
              spellCheck={false}
            />
          </div>
        ) : (
          /* View Mode - Rendered Markdown */
          <div className="p-6 workflow-content">
            <div className="prose prose-invert prose-lg max-w-none
              [&>h1]:text-2xl [&>h1]:font-bold [&>h1]:text-white [&>h1]:mb-4 [&>h1]:pb-3 [&>h1]:border-b [&>h1]:border-gray-700
              [&>h2]:text-xl [&>h2]:font-semibold [&>h2]:text-blue-400 [&>h2]:mt-8 [&>h2]:mb-4
              [&>h3]:text-base [&>h3]:font-semibold [&>h3]:text-gray-300 [&>h3]:mt-4 [&>h3]:mb-2
              [&>p]:text-gray-400 [&>p]:my-2 [&>p]:leading-relaxed
              [&>blockquote]:border-l-4 [&>blockquote]:border-blue-500 [&>blockquote]:bg-gray-800/50 [&>blockquote]:py-2 [&>blockquote]:px-4 [&>blockquote]:my-4 [&>blockquote]:rounded-r-lg [&>blockquote]:italic [&>blockquote]:text-gray-400
              [&>ul]:my-3 [&>ul]:pl-4 [&>ul>li]:text-gray-400 [&>ul>li]:my-1
              [&>hr]:border-gray-700 [&>hr]:my-8
              [&_code]:bg-gray-800 [&_code]:px-2 [&_code]:py-0.5 [&_code]:rounded [&_code]:text-emerald-400 [&_code]:text-sm [&_code]:font-mono
              [&_pre]:bg-gray-900 [&_pre]:border [&_pre]:border-gray-700 [&_pre]:rounded-lg [&_pre]:p-4 [&_pre]:my-3 [&_pre]:overflow-x-auto
              [&_pre_code]:bg-transparent [&_pre_code]:p-0 [&_pre_code]:text-gray-300
              [&_table]:w-full [&_table]:my-3
              [&_th]:bg-gray-800 [&_th]:px-4 [&_th]:py-2 [&_th]:text-left [&_th]:text-sm [&_th]:font-semibold [&_th]:text-gray-300 [&_th]:border [&_th]:border-gray-700
              [&_td]:px-4 [&_td]:py-2 [&_td]:border [&_td]:border-gray-700 [&_td]:text-gray-400
              [&_strong]:text-white [&_strong]:font-semibold
              [&_a]:text-blue-400 [&_a]:no-underline hover:[&_a]:underline
              [&_sub]:text-gray-500 [&_sub]:text-xs
            ">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {content}
              </ReactMarkdown>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
