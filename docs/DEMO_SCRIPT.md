# DocMind — Demo Script (2–3 minutes)

**Setup before you start:** DocMind running (`streamlit run app.py`), a
realistic sample PDF ready, Wi-Fi toggle accessible, Performance tab
pre-visited once so metrics aren't empty.

A ready-to-use synthetic sample document (a fictional 3-page quarterly
report — no real company/financial data) ships in this repo at
`assets/sample_documents/sample_quarterly_report.pdf`. It's designed to
give clean, demonstrable answers for the Q&A, summarization, and
insights steps below (revenue growth, named risks, dated action items).

---

**1. Introduce DocMind (15s)**
> "This is DocMind — a private document assistant. Instead of uploading
> your confidential PDFs to a cloud AI service, DocMind answers questions
> about them entirely on your own device."

**2. Upload a realistic PDF (15s)**
> Upload the sample document live. Point out the sidebar: "Private
> Offline Mode" is on by default, and the model status shows the
> embedding model and local LLM are both ready — no API key involved.

**3. Ask a question (25s)**
> Type a real question about the document's content. Click Ask.
> "Notice DocMind only answers from the document — if I ask something
> not in the document, it says so instead of guessing." (Optionally
> demonstrate this with a second, out-of-scope question.)
> Point at the Sources line: "Page 3" etc.

**4. Show source pages (10s)**
> "Every answer cites exactly where it came from in the document — no
> black box."

**5. Generate a summary (15s)**
> Switch to the Summarize tab, click Generate Summary. Briefly scroll
> through the executive summary / key points / action items.

**6. Show insights (15s)**
> Switch to Insights tab, click Extract Insights. Point out entities,
> dates, numbers pulled out automatically.

**7. Open Performance dashboard (20s)**
> Switch to Performance tab. "These aren't marketing numbers — this is
> the real, measured latency for this exact request: embedding time,
> retrieval time, generation time, and which execution provider is
> actually running the embedding model."

**8. Demonstrate offline capability (20s)**
> Turn off Wi-Fi live (or show it already off). Ask another question.
> "Still works — nothing about Private Offline Mode needs the internet."

**9. Explain Snapdragon/QNN optimization (25s)**
> "The embedding pipeline runs through ONNX Runtime with a priority
> order: Qualcomm's QNN execution provider for the Hexagon NPU first,
> then DirectML, then CPU — whichever is actually available on this
> machine, detected automatically, never assumed. On this laptop, that's
> [CPU / QNN — say the real, current value from the dashboard]." *(Only
> ever state what the dashboard currently shows.)*

**10. Show benchmark evidence (15s)**
> If available: briefly show the AI Hub device-cloud profiling report or
> `scripts/benchmark_results.json` as backing evidence for any
> Snapdragon-specific performance claim.

**11. Switch to Cloud Comparison Mode briefly (15s, optional)**
> Ask the same question in Cloud Comparison Mode. "Same question, same
> answer quality roughly, but now it went to Groq's cloud API — you can
> see the privacy and offline-capability difference directly."

**12. Final impact statement (10s)**
> "DocMind shows that private, capable document AI doesn't need to live
> in the cloud — Snapdragon's on-device AI makes that practical for
> everyday use, not just a lab demo."

---

**Total: ~3 minutes.** Cut step 11 first if running over time; it's the
least essential to the core pitch.
