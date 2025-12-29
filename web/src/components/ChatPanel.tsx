'use client';

import React, { useState, useRef, useEffect, useMemo, useLayoutEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Message } from '@/types/workflow';

interface ChatPanelProps {
  messages: Message[];
  onSendMessage: (content: string) => void;
  isLoading: boolean;
}

// Detect and extract auth links from content
function extractAuthLinks(content: string): { cleanContent: string; authLinks: string[] } {
  const authLinks: string[] = [];

  // Match various auth URL patterns from Composio and OAuth providers
  const authPatterns = [
    // Composio patterns
    /https?:\/\/[^\s<>"]*composio[^\s<>"]*/gi,
    /https?:\/\/[^\s<>"]*auth[^\s<>"]*/gi,
    /https?:\/\/[^\s<>"]*oauth[^\s<>"]*/gi,
    /https?:\/\/[^\s<>"]*connect[^\s<>"]*/gi,
    /https?:\/\/[^\s<>"]*authorize[^\s<>"]*/gi,
    // Google OAuth
    /https?:\/\/accounts\.google\.[^\s<>"]+/gi,
    // Generic login
    /https?:\/\/[^\s<>"]*login[^\s<>"]*/gi,
    // API connection URLs
    /https?:\/\/[^\s<>"]*\.api\.[^\s<>"]+connect[^\s<>"]*/gi,
  ];

  authPatterns.forEach(pattern => {
    const matches = content.match(pattern);
    if (matches) {
      matches.forEach(match => {
        // Clean up the URL (remove trailing punctuation and markdown)
        let cleanUrl = match
          .replace(/[),.\s\]>]+$/, '')
          .replace(/\)$/, '')
          .replace(/\]$/, '');

        // Skip if it's just a domain without path or looks incomplete
        if (cleanUrl.length > 20 && !authLinks.includes(cleanUrl)) {
          authLinks.push(cleanUrl);
        }
      });
    }
  });

  // Also check for markdown links with auth-related text
  const markdownLinkPattern = /\[([^\]]*)\]\((https?:\/\/[^)]+)\)/gi;
  let match;
  while ((match = markdownLinkPattern.exec(content)) !== null) {
    const linkText = match[1].toLowerCase();
    const url = match[2].replace(/[),.\s]+$/, '');

    // Check if link text suggests authentication
    const authKeywords = ['connect', 'authenticate', 'login', 'authorize', 'oauth', 'sign in', 'link account'];
    if (authKeywords.some(kw => linkText.includes(kw)) && !authLinks.includes(url)) {
      authLinks.push(url);
    }
  }

  // Check for "click here" or "this link" patterns followed by URLs
  const clickHerePattern = /(?:click\s+(?:here|this)|follow\s+this\s+link|use\s+this\s+link)[^\n]*?(https?:\/\/[^\s<>"]+)/gi;
  while ((match = clickHerePattern.exec(content)) !== null) {
    const url = match[1].replace(/[),.\s]+$/, '');
    if (!authLinks.includes(url)) {
      authLinks.push(url);
    }
  }

  return { cleanContent: content, authLinks };
}

// Auth button component
function AuthButton({ url }: { url: string }) {
  const getButtonLabel = (url: string): string => {
    if (url.includes('google')) return '🔗 Connect Google Account';
    if (url.includes('slack')) return '🔗 Connect Slack';
    if (url.includes('zoho')) return '🔗 Connect Zoho';
    if (url.includes('notion')) return '🔗 Connect Notion';
    if (url.includes('gmail')) return '🔗 Connect Gmail';
    return '🔗 Complete Authentication';
  };

  return (
    <a
      href={url}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-2 mt-3 px-4 py-2.5 bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-white rounded-lg font-medium text-sm transition-all shadow-lg hover:shadow-xl transform hover:scale-[1.02]"
    >
      {getButtonLabel(url)}
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
      </svg>
    </a>
  );
}

// Markdown renderer component
function MarkdownContent({ content }: { content: string }) {
  const { cleanContent, authLinks } = useMemo(() => extractAuthLinks(content), [content]);

  return (
    <div className="markdown-content min-w-0 break-words">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          // Style headings
          h1: ({ children }) => (
            <h1 className="text-lg font-bold mb-2 mt-3 first:mt-0">{children}</h1>
          ),
          h2: ({ children }) => (
            <h2 className="text-base font-semibold mb-2 mt-3 first:mt-0">{children}</h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-sm font-semibold mb-1.5 mt-2 first:mt-0">{children}</h3>
          ),
          // Style paragraphs
          p: ({ children }) => (
            <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>
          ),
          // Style lists
          ul: ({ children }) => (
            <ul className="list-disc list-inside mb-2 space-y-1 ml-1">{children}</ul>
          ),
          ol: ({ children }) => (
            <ol className="list-decimal list-inside mb-2 space-y-1 ml-1">{children}</ol>
          ),
          li: ({ children }) => (
            <li className="leading-relaxed">{children}</li>
          ),
          // Style code
          code: ({ className, children, ...props }) => {
            const isInline = !className;
            if (isInline) {
              return (
                <code className="bg-[var(--bg-tertiary)] px-1.5 py-0.5 rounded text-xs font-mono" {...props}>
                  {children}
                </code>
              );
            }
            return (
              <code className={`${className} block bg-[var(--bg-tertiary)] p-3 rounded-lg text-xs font-mono overflow-x-auto my-2`} {...props}>
                {children}
              </code>
            );
          },
          pre: ({ children }) => (
            <pre className="bg-[var(--bg-tertiary)] p-3 rounded-lg text-xs font-mono overflow-x-auto my-2">
              {children}
            </pre>
          ),
          // Style blockquotes
          blockquote: ({ children }) => (
            <blockquote className="border-l-3 border-[var(--accent)] pl-3 my-2 italic opacity-90">
              {children}
            </blockquote>
          ),
          // Style links - but NOT auth links (those get buttons)
          a: ({ href, children }) => {
            // Check if this is an auth link - render differently
            if (href && authLinks.includes(href)) {
              return null; // Will be rendered as button below
            }
            return (
              <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="text-[var(--accent)] hover:underline"
              >
                {children}
              </a>
            );
          },
          // Style tables
          table: ({ children }) => (
            <table className="w-full border-collapse my-2 text-xs">{children}</table>
          ),
          th: ({ children }) => (
            <th className="border border-[var(--border)] px-2 py-1 bg-[var(--bg-tertiary)] font-semibold text-left">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="border border-[var(--border)] px-2 py-1">{children}</td>
          ),
          // Style horizontal rules
          hr: () => <hr className="my-3 border-[var(--border)]" />,
          // Style strong/bold
          strong: ({ children }) => (
            <strong className="font-semibold">{children}</strong>
          ),
          // Style emphasis/italic
          em: ({ children }) => (
            <em className="italic">{children}</em>
          ),
        }}
      >
        {cleanContent}
      </ReactMarkdown>

      {/* Render auth buttons */}
      {authLinks.length > 0 && (
        <div className="flex flex-wrap gap-2 mt-3">
          {authLinks.map((url, index) => (
            <AuthButton key={index} url={url} />
          ))}
        </div>
      )}
    </div>
  );
}

export default function ChatPanel({ messages, onSendMessage, isLoading }: ChatPanelProps) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const MAX_TEXTAREA_HEIGHT_PX = 180;

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useLayoutEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    // Auto-grow textarea up to MAX_TEXTAREA_HEIGHT_PX
    el.style.height = 'auto';
    const nextHeight = Math.min(el.scrollHeight, MAX_TEXTAREA_HEIGHT_PX);
    el.style.height = `${nextHeight}px`;
    el.style.overflowY = el.scrollHeight > MAX_TEXTAREA_HEIGHT_PX ? 'auto' : 'hidden';
  }, [input]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      onSendMessage(input.trim());
      setInput('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter submits; Shift+Enter inserts newline (standard chat UX)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (input.trim() && !isLoading) {
        onSendMessage(input.trim());
        setInput('');
      }
    }
  };

  const suggestions = [
    "When a new document arrives in Gmail, save it to a specific folder in Google Drive",
    "Create a workflow that sends a Slack message when a new row is added to Google Sheets",
    "Summarize new Notion pages and send the summary via email",
    "Sync Google Docs to Notion automatically when documents are updated",
  ];

  return (
    <div className="flex flex-col h-full min-h-0 bg-[var(--bg-primary)]">
      {/* Header */}
      <div className="p-4 border-b border-[var(--border)]">
        <h2 className="text-sm font-semibold text-[var(--text-secondary)] uppercase tracking-wide">
          Chat with AI
        </h2>
      </div>

      {/* Messages */}
      <div className="flex-1 min-h-0 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center py-8">
            <div className="text-4xl mb-4">🤖</div>
            <h3 className="text-lg font-medium mb-2">Welcome to AI Workflow Builder</h3>
            <p className="text-sm text-[var(--text-secondary)] mb-6">
              Describe the workflow you want to create, and I'll help you build it.
            </p>
            <div className="space-y-2">
              <p className="text-xs text-[var(--text-secondary)] uppercase tracking-wide mb-2">
                Try these examples:
              </p>
              {suggestions.map((suggestion, i) => (
                <button
                  key={i}
                  onClick={() => onSendMessage(suggestion)}
                  className="block w-full text-left p-3 bg-[var(--bg-secondary)] rounded-lg text-sm hover:bg-[var(--bg-tertiary)] transition-colors"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex min-w-0 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[85%] min-w-0 break-words rounded-2xl px-4 py-3 ${message.role === 'user'
                  ? 'bg-[var(--accent)] text-white'
                  : 'bg-[var(--bg-secondary)] text-[var(--text-primary)]'
                }`}
            >
              {message.role === 'user' ? (
                <p className="text-sm whitespace-pre-wrap break-words">{message.content}</p>
              ) : (
                <div className="text-sm min-w-0">
                  <MarkdownContent content={message.content} />
                </div>
              )}
              <p className="text-xs opacity-60 mt-2">
                {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </p>
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-[var(--bg-secondary)] rounded-2xl px-4 py-3">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-[var(--accent)] rounded-full animate-pulse" />
                <div className="w-2 h-2 bg-[var(--accent)] rounded-full animate-pulse" style={{ animationDelay: '150ms' }} />
                <div className="w-2 h-2 bg-[var(--accent)] rounded-full animate-pulse" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="p-4 border-t border-[var(--border)]">
        <div className="flex gap-2 items-end">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Describe your workflow..."
            disabled={isLoading}
            rows={1}
            className="flex-1 bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl px-4 py-3 text-sm leading-relaxed focus:outline-none focus:border-[var(--accent)] transition-colors disabled:opacity-50 resize-none"
          />
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="px-6 py-3 bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white rounded-xl font-medium text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Send
          </button>
        </div>
        <p className="mt-2 text-[11px] text-[var(--text-secondary)]">
          Press <strong className="font-semibold">Enter</strong> to send, <strong className="font-semibold">Shift+Enter</strong> for a new line.
        </p>
      </form>
    </div>
  );
}


