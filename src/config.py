from dynaconf import Dynaconf
from pyprojroot import here

PROJECT_ROOT = here()

_settings_path = PROJECT_ROOT / "settings.yaml"
if not _settings_path.exists():
    raise FileNotFoundError(
        f"Configuration file not found: {_settings_path}. "
        "Ensure settings.yaml is present at the project root."
    )

settings = Dynaconf(
    # Multiple settings files for different environments
    environments=True,
    env_switcher="APP_ENV",

    # Structured, non-secret configs go here
    settings_files=['settings.yaml'],

    # Automatically and securely loads the .env file into the environment
    load_dotenv=True,

    # Set the root path
    root_path=str(PROJECT_ROOT),

    # Prefix  environment variables
    envvar_prefix="APP"
)

_REQUIRED_SETTINGS = [
    ("models.model_id", "APP_ENV or environment-specific models.model_id in settings.yaml"),
    ("database.type", "APP_ENV-specific database.type in settings.yaml"),
]

_missing = [
    (key, hint)
    for key, hint in _REQUIRED_SETTINGS
    if not settings.get(key)
]
if _missing:
    lines = "\n".join(f"  - {key}  (set via: {hint})" for key, hint in _missing)
    raise RuntimeError(
        f"Missing required configuration values:\n{lines}\n"
        "Check settings.yaml and your environment variables."
    )