# Agent Suite

15 specialized AI agents for sales, marketing, content, operations, and personal productivity.

## Agents

| # | Agent | Description |
|---|-------|-------------|
| 1 | SEO Analyst | Keyword research, competitor analysis, backlink monitoring, technical SEO audits |
| 2 | Lead Scraping + Enrichment | Web scraping for leads, data enrichment, deduplication, lead scoring |
| 3 | CRM + Inbox Management | Contact management, email triage, follow-up scheduling, pipeline tracking |
| 4 | Sales Call Analyst | Transcription, sentiment analysis, key moments extraction, action items |
| 5 | Research Agent | AI/news summaries, topic monitoring, report generation, trend analysis |
| 6 | Content Production | Blog posts, social media, newsletters, video scripts, SEO-optimized content |
| 7 | Content Repurposing | Blog-to-social, video-to-text, podcast-to-blog, cross-platform adaptation |
| 8 | Second Brain | Knowledge base, note linking, spaced repetition, context retrieval |
| 9 | Executive Assistant | Calendar management, meeting prep, travel planning, priority triage |
| 10 | Business Analyst + KPI | Dashboard creation, KPI tracking, anomaly detection, report generation |
| 11 | Operations Employee | Task management, SOP tracking, vendor management, process automation |
| 12 | Customer Support | Ticket triage, response drafting, escalation handling, FAQ management |
| 13 | Shopify Assistant | Product management, order tracking, inventory alerts, customer service |
| 14 | Investment Analyst | Market research, stock screening, portfolio analysis, risk assessment |
| 15 | Advisory Council + NotebookLM | Multi-perspective advice, document analysis, decision support |

## Quick Start

```bash
# Install dependencies
pip install -e .

# Copy environment template
cp .env.example .env
# Edit .env with your API keys

# Run an agent
python scripts/run_agent.py seo_analyst --domain example.com
```

## Project Structure

- `core/` -- Shared infrastructure (LLM setup, tools, integrations, storage)
- `agents/` -- Individual agent modules (one per agent)
- `workflows/` -- Multi-agent orchestration workflows
- `tests/` -- Test suite
- `scripts/` -- Utility scripts
