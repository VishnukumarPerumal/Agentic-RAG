# Agentic RAG Architecture & Product Deployment Guide

## 1. System Overview
The Agentic RAG platform combines a three-stage LangGraph workflow with PostgreSQL 18 and pgvector 0.8.6.
All incoming user queries first pass through the **Guardrail Agent**, which screens for prompt injections,
destructive SQL commands, harmful requests, and sensitive PII such as SSNs or credit card numbers.

## 2. Nexus GPU Inference Server X1 (SKU: AI-SRV-100)
The Nexus GPU Inference Server X1 features 4 high-bandwidth tensor GPUs with 320 GB total VRAM,
dual redundant 2000W Titanium power supplies, and pre-installed CUDA + vLLM + LangGraph runtimes.
Recommended deployment temperature is 18°C to 24°C in a standard 4U rackmount enclosure.
Annual hardware maintenance plan is included for the first 3 years.

## 3. VectorCloud Enterprise Plan (SKU: VEC-CLD-200)
VectorCloud Enterprise provides managed PostgreSQL clusters with the `pgvector` extension pre-configured.
It uses HNSW (`vector_cosine_ops`) indexing for sub-10ms approximate nearest neighbor search across millions
of 384-dimensional or 1536-dimensional embeddings.

## 4. Onboarding & Support Escalation Procedure
New enterprise customers receive a dedicated deployment engineer within 24 hours of order confirmation.
For urgent technical issues, customers can trigger a Severity-1 ticket which guarantees a 15-minute
response SLA from the Site Reliability Engineering (SRE) team.
