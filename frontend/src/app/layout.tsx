import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Agentic RAG Chatbot | LangGraph + pgvector + MCP",
  description:
    "Multi-Agent RAG system with Guardrail Agent, Data Retrieval Agent (pgvector, PostgreSQL, MCP Tools), and Answer Generation Agent.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
