'use client';

import React, { useState, useCallback, useRef, useEffect } from 'react';
import Navigation from '@/components/Navigation';
import ChatPanel from '@/components/ChatPanel';
import AuthStatus from '@/components/AuthStatus';
import WorkflowBuilder from '@/components/WorkflowBuilder';
import WorkflowList from '@/components/WorkflowList';
import { AuthStatus as AuthStatusType, Message, Workflow, WorkflowFileSummary } from '@/types/workflow';
import { Client } from '@langchain/langgraph-sdk';

// API configuration
const LANGGRAPH_API_URL = process.env.NEXT_PUBLIC_LANGGRAPH_API_URL || 'http://localhost:2025';
const WORKFLOW_API_URL = process.env.NEXT_PUBLIC_WORKFLOW_API_URL || 'http://localhost:2026';

export default function Home() {
  // Chat state
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const threadIdRef = useRef<string | null>(null);
  const [chatMode, setChatMode] = useState<'build' | 'configure'>('build');

  // Workflow state (JSON-first, Markdown view)
  const [currentWorkflowId, setCurrentWorkflowId] = useState<string | null>(null);
  const [workflowMarkdown, setWorkflowMarkdown] = useState<string>('');
  const [currentWorkflow, setCurrentWorkflow] = useState<Workflow | null>(null);
  const [savedWorkflows, setSavedWorkflows] = useState<WorkflowFileSummary[]>([]);
  const [authStatus, setAuthStatus] = useState<AuthStatusType | null>(null);
  const [missingConfigKeys, setMissingConfigKeys] = useState<string[]>([]);
  const [workflowApiOk, setWorkflowApiOk] = useState<boolean | null>(null); // null=unknown, true=ok, false=offline

  // UI state
  const [showWorkflowList, setShowWorkflowList] = useState(true);
  const [isExecuting, setIsExecuting] = useState(false);
  const [currentExecutionStep, setCurrentExecutionStep] = useState(-1);

  // Fetch workflows from file API on mount
  useEffect(() => {
    fetchWorkflows();
  }, []);

  // Fetch all workflows from the file API
  const fetchWorkflows = useCallback(async () => {
    try {
      const response = await fetch(`${WORKFLOW_API_URL}/workflows`);
      if (response.ok) {
        const workflows = await response.json();
        setSavedWorkflows(workflows);
        setWorkflowApiOk(true);
      } else {
        setWorkflowApiOk(false);
      }
    } catch (error) {
      console.error('Failed to fetch workflows:', error);
      // API not available - that's okay, we'll show empty list
      setWorkflowApiOk(false);
    }
  }, []);

  // If Workflow API is temporarily offline, keep retrying in the background so the UI "heals" once it starts.
  useEffect(() => {
    if (workflowApiOk !== false) return;
    const t = window.setInterval(() => {
      fetchWorkflows();
    }, 3000);
    return () => window.clearInterval(t);
  }, [workflowApiOk, fetchWorkflows]);

  const clearAllWorkflows = useCallback(async () => {
    const ok = window.confirm('Delete ALL saved workflows? This cannot be undone.');
    if (!ok) return;

    try {
      const res = await fetch(`${WORKFLOW_API_URL}/workflows`, { method: 'DELETE' });
      if (!res.ok) throw new Error('Failed to delete workflows');

      // Reset local UI state + refresh list
      setSavedWorkflows([]);
      setCurrentWorkflowId(null);
      setWorkflowMarkdown('');
      setCurrentWorkflow(null);
      setAuthStatus(null);
      setMissingConfigKeys([]);
      setChatMode('build');
      setCurrentExecutionStep(-1);
      setIsExecuting(false);
      await fetchWorkflows();
    } catch (e) {
      console.error('Failed to clear workflows:', e);
      const errorMessage: Message = {
        id: `msg_${Date.now()}_error`,
        role: 'assistant',
        content: `## ⚠️ Could not clear workflows\n\nMake sure the Workflow API is running on \`${WORKFLOW_API_URL}\`.`,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    }
  }, [fetchWorkflows]);

  // Clean response for display (remove any legacy blocks if they appear)
  const cleanResponseForDisplay = useCallback((content: string): string => {
    return content
      .replace(/WORKFLOW_MARKDOWN_START\n[\s\S]*?\nWORKFLOW_MARKDOWN_END\n*/g, '')
      .replace(/WORKFLOW_SCHEMA_START\n[\s\S]*?\nWORKFLOW_SCHEMA_END\n*/g, '')
      .replace(/WORKFLOW_CONFIRMED_START\n[\s\S]*?\nWORKFLOW_CONFIRMED_END\n*/g, '')
      .replace(/WORKFLOW_DISPLAY_START\n[\s\S]*?\nWORKFLOW_DISPLAY_END\n*/g, '')
      .trim();
  }, []);

  // Run a mode-specific agent call and stream state updates
  const runAgent = useCallback(async (
    mode: 'build' | 'preflight' | 'configure' | 'execute',
    content: string,
    workflowId?: string | null,
    opts?: {
      streamAssistantMessages?: boolean;
      onAssistantMessage?: (raw: string) => void;
    }
  ) => {
    const client = new Client({ apiUrl: LANGGRAPH_API_URL });

    if (!threadIdRef.current) {
      const thread = await client.threads.create();
      threadIdRef.current = thread.thread_id;
    }

    const input: any = {
      mode,
      messages: [{ role: 'user', content }],
    };
    if (workflowId) input.current_workflow_id = workflowId;

    const stream = client.runs.stream(threadIdRef.current, 'workflow_builder', { input });

    let lastAssistantContent = '';
    let lastState: any = null;
    const streamAssistantMessages = !!opts?.streamAssistantMessages;
    const onAssistantMessage = opts?.onAssistantMessage;
    let lastSeenMessagesLen: number | null = null;
    let streamedAny = false;

    for await (const event of stream) {
      if (event.event !== 'values') continue;
      const data: any = event.data || {};
      lastState = data;

      // State-driven UI updates
      if (data?.mode === 'build' || data?.mode === 'configure') {
        setChatMode(data.mode);
      }
      if (typeof data.current_workflow_id === 'string') {
        setCurrentWorkflowId(data.current_workflow_id);
      }
      if (data.workflow && typeof data.workflow === 'object') {
        setCurrentWorkflow(data.workflow as Workflow);
      }
      if (typeof data.workflow_markdown === 'string') {
        setWorkflowMarkdown(data.workflow_markdown);
      }
      if (Array.isArray(data.missing_config_keys)) {
        setMissingConfigKeys(data.missing_config_keys.filter((k: any) => typeof k === 'string'));
      }
      if (data.auth_status) {
        setAuthStatus(data.auth_status);
      }
      if (mode === 'execute' && typeof data.execution_cursor === 'number') {
        // cursor is 0-based for next step; for progress bar we use current step index
        setCurrentExecutionStep(Math.max(0, data.execution_cursor));
      }

      // Pull last assistant message for chat display
      const msgs = data.messages;
      if (Array.isArray(msgs) && msgs.length > 0) {
        // Optional: stream assistant messages incrementally (used for Execute progress).
        if (streamAssistantMessages && typeof onAssistantMessage === 'function') {
          if (lastSeenMessagesLen === null) {
            // First snapshot: establish baseline to avoid re-appending old thread history.
            lastSeenMessagesLen = msgs.length;
          } else {
            if (msgs.length < lastSeenMessagesLen) lastSeenMessagesLen = msgs.length;
            const newMsgs = msgs.slice(lastSeenMessagesLen);
            lastSeenMessagesLen = msgs.length;
            for (const m of newMsgs) {
              const isAssistant = m?.type === 'ai' || m?.role === 'assistant';
              if (!isAssistant) continue;
              const toolCalls = (m?.tool_calls || m?.toolCalls) as any;
              const rawContent = typeof m?.content === 'string' ? m.content : JSON.stringify(m?.content ?? '');
              // Skip tool-call-only messages (no readable content).
              if (Array.isArray(toolCalls) && toolCalls.length > 0 && !rawContent.trim()) continue;
              if (!rawContent.trim()) continue;
              onAssistantMessage(rawContent);
              streamedAny = true;
            }
          }
        }

        const lastMsg = msgs[msgs.length - 1];
        if (lastMsg.type === 'ai' || lastMsg.role === 'assistant') {
          lastAssistantContent = typeof lastMsg.content === 'string'
            ? lastMsg.content
            : JSON.stringify(lastMsg.content);
        }
      }
    }

    return { lastAssistantContent, lastState, streamedAny };
  }, []);

  const appendAssistantMessage = useCallback((raw: string) => {
    const cleaned = cleanResponseForDisplay(raw || '');
    if (!cleaned) return;
    const assistantMessage: Message = {
      id: `msg_${Date.now()}_ai`,
      role: 'assistant',
      content: cleaned,
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, assistantMessage]);
  }, [cleanResponseForDisplay]);

  const runPreflight = useCallback(async (workflowId: string) => {
    const res = await runAgent('preflight', 'Check connections', workflowId);
    appendAssistantMessage(res.lastAssistantContent || '');
    // Return both auth_status and missing_config_keys for flow gating
    const authStatus = (res.lastState?.auth_status || null) as AuthStatusType | null;
    const missingKeys = Array.isArray(res.lastState?.missing_config_keys)
      ? res.lastState.missing_config_keys.filter((k: any) => typeof k === 'string')
      : [];
    return { authStatus, missingKeys };
  }, [appendAssistantMessage, runAgent]);

  const runExecute = useCallback(async (workflowId: string) => {
    const res = await runAgent('execute', 'Execute workflow', workflowId, {
      streamAssistantMessages: true,
      onAssistantMessage: appendAssistantMessage,
    });
    // Fallback if the stream didn't emit a readable assistant message.
    if (!res.streamedAny) appendAssistantMessage(res.lastAssistantContent || '');
    return res.lastState || null;
  }, [appendAssistantMessage, runAgent]);

  // Send message to LangGraph agent
  const sendMessage = useCallback(async (content: string) => {
    const userMessage: Message = {
      id: `msg_${Date.now()}`,
      role: 'user',
      content,
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const modeToUse: 'build' | 'configure' = chatMode === 'configure' ? 'configure' : 'build';
      const result = await runAgent(modeToUse, content, currentWorkflowId);
      appendAssistantMessage(result.lastAssistantContent || '');

      // Refresh workflow list (new workflow may have been created)
      fetchWorkflows();

      // If we just built/updated a workflow and it needs config, immediately transition into Configure.
      const missing = Array.isArray(result.lastState?.missing_config_keys)
        ? result.lastState.missing_config_keys.filter((k: any) => typeof k === 'string')
        : [];
      const wfIdFromState = typeof result.lastState?.current_workflow_id === 'string'
        ? result.lastState.current_workflow_id
        : currentWorkflowId;
      if (modeToUse === 'build' && wfIdFromState && missing.length > 0) {
        const cfg = await runAgent('configure', 'Configure workflow', wfIdFromState);
        appendAssistantMessage(cfg.lastAssistantContent || '');
      }

    } catch (error) {
      console.error('Error calling agent:', error);

      const errorMessage: Message = {
        id: `msg_${Date.now()}_error`,
        role: 'assistant',
        content: `## ⚠️ Connection Issue

Unable to reach the workflow agent. Please ensure:
1. The LangGraph server is running (\`langgraph dev --port 2025\`)
2. The workflow API is running (\`python api_server.py\`)

Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  }, [cleanResponseForDisplay, fetchWorkflows, currentWorkflowId, runAgent, chatMode, appendAssistantMessage]);

  // Save workflow content to file
  const handleSaveWorkflow = useCallback(async (content: string) => {
    if (!currentWorkflowId) {
      return { ok: false as const, error: 'No workflow selected.' };
    }

    let parsed: any;
    try {
      parsed = JSON.parse(content);
    } catch {
      return { ok: false as const, error: 'Invalid JSON. Please fix syntax and try again.' };
    }

    try {
      const res = await fetch(`${WORKFLOW_API_URL}/workflows/${currentWorkflowId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ workflow: parsed }),
      });
      if (!res.ok) {
        let detail = 'Failed to save workflow.';
        try {
          const body = await res.json();
          if (body?.detail) detail = String(body.detail);
        } catch {
          // ignore
        }
        return { ok: false as const, error: detail };
      }

      const data = await res.json();
      if (typeof data?.markdown === 'string') setWorkflowMarkdown(data.markdown);
      if (data?.workflow && typeof data.workflow === 'object') setCurrentWorkflow(data.workflow as Workflow);
      fetchWorkflows();
      return { ok: true as const };
    } catch (e) {
      console.error('Failed to save workflow:', e);
      return { ok: false as const, error: `Could not reach Workflow API (${WORKFLOW_API_URL}).` };
    }
  }, [currentWorkflowId, fetchWorkflows]);

  // Select a workflow from the sidebar
  const handleSelectWorkflow = useCallback(async (workflowId: string) => {
    try {
      const response = await fetch(`${WORKFLOW_API_URL}/workflows/${workflowId}`);
      if (response.ok) {
        const data = await response.json();
        setCurrentWorkflowId(workflowId);
        setWorkflowMarkdown(data.markdown || '');
        setCurrentWorkflow(data.workflow || null);
        setMissingConfigKeys([]);
        setChatMode('build');

        const loadMessage: Message = {
          id: `msg_${Date.now()}_load`,
          role: 'assistant',
          content: `## 📂 Loaded: ${data.name}

Workflow loaded: \`${workflowId}\`

You can:
- **Modify** through chat
- **Run** using the ▶ Run button (will check auth first)
- **Modify** through chat`,
          timestamp: new Date(),
        };
        setMessages(prev => [...prev, loadMessage]);
      }
    } catch (error) {
      console.error('Failed to load workflow:', error);
    }
  }, [sendMessage]);

  const handleDeleteWorkflow = useCallback(async (workflowId: string) => {
    const ok = window.confirm(`Delete workflow ${workflowId}?`);
    if (!ok) return;

    try {
      const res = await fetch(`${WORKFLOW_API_URL}/workflows/${workflowId}`, { method: 'DELETE' });
      if (!res.ok) throw new Error('Failed to delete workflow');

      // If currently open, clear selection/panels
      setSavedWorkflows(prev => prev.filter(w => w.id !== workflowId));
      if (currentWorkflowId === workflowId) {
        setCurrentWorkflowId(null);
        setWorkflowMarkdown('');
        setCurrentWorkflow(null);
        setAuthStatus(null);
        setMissingConfigKeys([]);
        setChatMode('build');
        setCurrentExecutionStep(-1);
        setIsExecuting(false);
      }

      await fetchWorkflows();
    } catch (e) {
      console.error('Failed to delete workflow:', e);
      const errorMessage: Message = {
        id: `msg_${Date.now()}_error`,
        role: 'assistant',
        content: `## ⚠️ Could not delete workflow\n\nMake sure the Workflow API is running on \`${WORKFLOW_API_URL}\`.`,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    }
  }, [currentWorkflowId, fetchWorkflows]);

  // Execute a workflow
  const handleExecuteWorkflow = useCallback(async (filename?: string) => {
    try {
      const targetId = filename || currentWorkflowId;
      if (!targetId) return;

      // 1) Connections check
      const preflightMsg: Message = {
        id: `msg_${Date.now()}_preflight`,
        role: 'assistant',
        content: `## 🔌 Connections\n\nChecking required connections...`,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, preflightMsg]);

      const pre = await runPreflight(targetId);

      // 2) If all connected, either configure (if needed) or execute
      if (pre?.authStatus?.all_connected) {
        // Use fresh missing_config_keys from preflight response, not stale React state
        const freshMissingKeys = pre.missingKeys || [];
        if (freshMissingKeys.length > 0) {
          // Update React state for UI display
          setMissingConfigKeys(freshMissingKeys);

          const cfgMsg: Message = {
            id: `msg_${Date.now()}_need_cfg`,
            role: 'assistant',
            content: `## 🧩 Configuration Needed\n\nThis workflow needs a bit more configuration before it can run:\n\n- ${freshMissingKeys.map((k: string) => `**${k}**`).join('\\n- ')}\n\nI'll guide you through picking these values now.`,
            timestamp: new Date(),
          };
          setMessages(prev => [...prev, cfgMsg]);

          const cfg = await runAgent('configure', 'Configure workflow', targetId);
          appendAssistantMessage(cfg.lastAssistantContent || '');
          return;
        }

        setIsExecuting(true);
        setCurrentExecutionStep(0);
        const execMsg: Message = {
          id: `msg_${Date.now()}_exec`,
          role: 'assistant',
          content: `## ⚡ Executing\n\nStarting workflow execution...`,
          timestamp: new Date(),
        };
        setMessages(prev => [...prev, execMsg]);
        await runExecute(targetId);
      } else {
        const waitMsg: Message = {
          id: `msg_${Date.now()}_wait_auth`,
          role: 'assistant',
          content: `## 🔗 Action Required\n\nConnect the missing accounts above, then click **Check Again**.`,
          timestamp: new Date(),
        };
        setMessages(prev => [...prev, waitMsg]);
      }

    } catch (error) {
      console.error('Execution error:', error);

      const errorMessage: Message = {
        id: `msg_${Date.now()}_error`,
        role: 'assistant',
        content: `## ⚠️ Execution Error

Failed to execute workflow. Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    }

    setIsExecuting(false);
    setCurrentExecutionStep(-1);
  }, [currentWorkflowId, runPreflight, runExecute, missingConfigKeys, runAgent, appendAssistantMessage]);

  // Create a new workflow (reset state)
  const handleNewWorkflow = useCallback(() => {
    setMessages([]);
    setCurrentWorkflowId(null);
    setWorkflowMarkdown('');
    setCurrentWorkflow(null);
    setAuthStatus(null);
    setMissingConfigKeys([]);
    setChatMode('build');
    setCurrentExecutionStep(-1);
    setIsExecuting(false);
    threadIdRef.current = null;
  }, []);

  // Count steps in current workflow
  const stepMatches = workflowMarkdown.match(/## Step \d+:/g);
  const stepCount = stepMatches ? stepMatches.length : 0;

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      <Navigation
        onHomeClick={() => {
          handleNewWorkflow();
          setShowWorkflowList(true);
        }}
      />

      <div className="flex-1 flex overflow-hidden">
        {/* Saved Workflows Sidebar */}
        {showWorkflowList && (
          <div className="w-[220px] border-r border-[var(--border)] bg-[var(--bg-secondary)] flex-shrink-0 flex flex-col min-h-0 overflow-hidden">
            <div className="p-3 border-b border-[var(--border)] flex items-center justify-between">
              <h3 className="text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wide">
                Saved Workflows
              </h3>
              <div className="flex items-center gap-2">
                <span
                  className={`inline-block h-2 w-2 rounded-full ${workflowApiOk === true
                    ? 'bg-emerald-500'
                    : workflowApiOk === false
                      ? 'bg-red-500'
                      : 'bg-amber-400'
                    }`}
                  title={
                    workflowApiOk === true
                      ? 'Workflow API: online'
                      : workflowApiOk === false
                        ? 'Workflow API: offline'
                        : 'Workflow API: unknown'
                  }
                />
                <button
                  onClick={fetchWorkflows}
                  className="text-[var(--text-secondary)] hover:text-white text-sm"
                  title="Refresh"
                >
                  ↻
                </button>
                <button
                  onClick={clearAllWorkflows}
                  className="text-[var(--text-secondary)] hover:text-red-300 text-sm"
                  title="Clear all"
                >
                  🗑
                </button>
                <button
                  onClick={() => setShowWorkflowList(false)}
                  className="text-[var(--text-secondary)] hover:text-white text-sm"
                  title="Hide"
                >
                  ✕
                </button>
              </div>
            </div>

            {/* New Workflow Button */}
            <div className="p-2 border-b border-[var(--border)]">
              <button
                onClick={handleNewWorkflow}
                className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white rounded-lg text-sm font-medium transition-colors"
              >
                <span>+</span>
                <span>New Workflow</span>
              </button>
            </div>

            <div className="flex-1 overflow-y-auto overflow-x-hidden">
              <WorkflowList
                workflows={savedWorkflows}
                onSelectWorkflow={handleSelectWorkflow}
                onExecuteWorkflow={handleExecuteWorkflow}
                onDeleteWorkflow={handleDeleteWorkflow}
                selectedWorkflowId={currentWorkflowId}
              />
              {workflowApiOk === false && (
                <div className="px-3 py-2 text-xs text-[var(--text-secondary)] border-t border-[var(--border)]">
                  Workflow API is offline. Start it with:
                  <div className="mt-1 font-mono text-[10px] bg-[var(--bg-tertiary)] px-2 py-1 rounded">
                    cd agent &amp;&amp; python api_server.py
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Sidebar toggle when hidden */}
        {!showWorkflowList && (
          <button
            onClick={() => setShowWorkflowList(true)}
            className="w-8 border-r border-[var(--border)] bg-[var(--bg-secondary)] flex items-center justify-center hover:bg-[var(--bg-tertiary)] transition-colors"
            title="Show workflows"
          >
            <span className="text-[var(--text-secondary)] text-sm">›</span>
          </button>
        )}

        {/* Chat Panel */}
        <div className="w-[420px] border-r border-[var(--border)] flex-shrink-0 flex flex-col min-h-0 overflow-hidden">
          <AuthStatus
            authStatus={authStatus}
            onCheckAgain={() => currentWorkflowId && runPreflight(currentWorkflowId)}
            // Always run the full gated flow (connections -> execute), even from the AuthStatus panel.
            onExecute={() => handleExecuteWorkflow(currentWorkflowId || undefined)}
          />
          <div className="flex-1 min-h-0">
            <ChatPanel
              messages={messages}
              onSendMessage={sendMessage}
              isLoading={isLoading}
            />
          </div>
        </div>

        {/* Workflow Builder (Markdown Editor) */}
        <div className="flex-1 overflow-hidden">
          <WorkflowBuilder
            workflowId={currentWorkflowId}
            content={workflowMarkdown}
            workflowJson={currentWorkflow}
            onSave={handleSaveWorkflow}
            onExecute={() => handleExecuteWorkflow()}
            isExecuting={isExecuting}
            stepCount={stepCount}
            currentExecutionStep={currentExecutionStep}
          />
        </div>
      </div>
    </div>
  );
}
