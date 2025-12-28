import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'AI Workflow Builder',
  description: 'Create automated workflows with natural language',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  )
}








