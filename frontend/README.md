# Frontend

The chat page of the Movie Research Copilot: Next.js with TypeScript and Tailwind, exported as static files
(`output: "export"`). It has no server of its own; the browser calls the FastAPI backend and reads the answer as a
stream. Setup and the whole system are described in the [main README](../README.md).

```bash
npm install
npm run dev     # http://localhost:3000, expects the backend on http://localhost:8000
npm run build   # type check and static export into out/
npm run lint
npm run types   # regenerate app/api-types.ts from openapi.json
```

`NEXT_PUBLIC_API_BASE_URL` sets another backend address; it is built into the files (see `.env.example`).

| File | What it does |
|---|---|
| `app/page.tsx` | The chat: sends a question, reads the Server-Sent Events, keeps the conversation |
| `app/SourceList.tsx`, `app/citations.ts` | Cited sources first, numbered as in the answer; the rest collapsed |
| `app/RagPanel.tsx`, `app/ToolCallList.tsx`, `app/UsagePanel.tsx` | The collapsible panels under each answer |
| `app/api.ts`, `app/api-types.ts` | The API's types, generated from the backend's schemas |

**API types.** `openapi.json` is written by `python scripts/export_openapi.py` (repo root) and `api-types.ts` is
generated from it. After changing a backend schema, run that script and `npm run types`, and commit both files; the
tests and CI fail when either is out of date.
