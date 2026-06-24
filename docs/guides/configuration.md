# Configuration

All settings live in `settings.yaml` and are managed by [Dynaconf](https://www.dynaconf.com/).

## Environments

The active environment is set via the `APP_ENV` variable (default: `local`).

```bash
APP_ENV=gemini poetry run uvicorn src.api.app:app
```

## settings.yaml structure

```yaml
default:
  langfuse:
    enabled: false

local:                       # OpenAI / local LLM
  models:
    provider: "openai"
    model_id: "gpt-4o-mini"
  database:
    type: "sqlite"

gemini:                      # Google Gemini
  models:
    provider: "gemini"
    model_id: "gemini-1.5-flash"
```

## Secret injection

Secrets are **never** stored in `settings.yaml`. Use environment variables with the `APP_` prefix:

| Env var | Setting path | Description |
|---|---|---|
| `APP_TOKEN` | `settings.token` | LLM API key |
| `APP_DATABASE__SUPABASE_URL` | `settings.database.supabase_url` | Supabase project URL |
| `APP_DATABASE__SUPABASE_KEY` | `settings.database.supabase_key` | Supabase service-role key |
| `APP_DATABASE__POSTGRES_URI` | `settings.database.postgres_uri` | Full PostgreSQL URI |
| `APP_LANGFUSE__PUBLIC_KEY` | `settings.langfuse.public_key` | LangFuse public key |
| `APP_LANGFUSE__SECRET_KEY` | `settings.langfuse.secret_key` | LangFuse secret key |

## LLM Provider selection

The provider is resolved in order:

1. `settings.models.provider` (from `settings.yaml` based on `APP_ENV`)
2. `LLM_PROVIDER` environment variable (legacy override)
3. Falls back to `openai`

To add a new provider, register it in `_PROVIDER_REGISTRY` inside `src/agents/application/factory.py`.
