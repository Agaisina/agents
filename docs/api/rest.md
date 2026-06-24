# REST API

The FastAPI application exposes three endpoints.

## Endpoints

### `POST /v1/chat`
Invoke the agent graph and return the final response in one shot.

**Request body** — `ChatRequest`:

| Field | Type | Description |
|---|---|---|
| `query` | `str` | Employee's travel request (2–5000 chars) |
| `thread_id` | `str` | Unique conversation ID for checkpointing |

**Response** — `ChatResponse`:

| Field | Type | Description |
|---|---|---|
| `answer` | `str` | Final employee-facing report |
| `thread_id` | `str` | Echoed from the request |
| `trace_id` | `str \| null` | LangFuse trace ID |

---

### `POST /v1/chat/stream`
Stream graph node updates and final tokens via **NDJSON** (newline-delimited JSON).

Each line is a JSON object with a `type` field:

| `type` | Extra fields | Meaning |
|---|---|---|
| `node_update` | `node: str` | A graph node finished executing |
| `token` | `content: str` | A word/token from the final response |
| `interrupted` | — | Graph paused; awaiting manager approval |
| `error` | `content: str` | An error occurred during streaming |

---

### `POST /v1/chat/resume`
Resume a suspended workflow after a manager decision.

**Request body** — `ResumeRequest`:

| Field | Type | Description |
|---|---|---|
| `thread_id` | `str` | ID of the suspended conversation |
| `approve` | `bool` | `true` to approve, `false` to reject |
| `feedback` | `str \| null` | Optional manager comment |

---

## Rate limiting

All endpoints share a **sliding-window** rate limit: **20 requests per 60 seconds** per client IP. Exceeding the limit returns HTTP 429.

## Auto-generated Swagger UI

When the server is running, interactive documentation is available at:

```
http://localhost:8000/docs      ← Swagger UI
http://localhost:8000/redoc     ← ReDoc
```
