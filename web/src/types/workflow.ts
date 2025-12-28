export interface WorkflowStep {
  id: string;
  order: number;
  name: string;
  description: string;
  instructions: string;
  tool: string;
  tool_params: Record<string, unknown>;
  toolkit: string;
  inputs: string[];
  outputs: string;
  execution_method?: 'internal' | 'tool_router';
  requires_connection?: boolean;
  status: 'pending' | 'running' | 'completed' | 'failed';
}

export interface Workflow {
  id: string;
  name: string;
  description: string;
  created_at: string;
  updated_at: string;
  steps: WorkflowStep[];
  status: 'draft' | 'ready' | 'executing' | 'completed' | 'failed';
  required_toolkits: string[];
  config?: Record<string, unknown>;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

// File-based workflow summary (from API)
export interface WorkflowFileSummary {
  id: string;
  name: string;
  description: string;
  step_count: number;
  updated_at: string;
  status: string;
  required_toolkits: string[];
}

export interface ToolkitAuthStatus {
  connected: boolean;
  connect_url: string | null;
  label: string;
  error?: string;
}

export interface AuthStatus {
  all_connected: boolean;
  toolkits: Record<string, ToolkitAuthStatus>;
}



