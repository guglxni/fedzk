# FEDzk agents — use the v2 HTML kit

The upstream repo root matches GitHub `main`. **Do not treat root docs as the v2 operating surface.**

All v2 agent instructions live here:

→ [`v2/docs/agent-kit/MANIFEST.html`](docs/agent-kit/MANIFEST.html)  
→ [`v2/docs/agent-kit/FORMAT.html`](docs/agent-kit/FORMAT.html)  
→ [`v2/docs/agent-kit/scratchpad.html`](docs/agent-kit/scratchpad.html)

```bash
cd v2/docs/agent-kit && python3 -m http.server 8765
```

**Forbidden:** loading `vendor/aidlc-workflows/**/*.md` as FEDzk instructions; editing root `src/` for v2 features.
