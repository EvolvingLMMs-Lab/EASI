"""Custom render platforms for the dummy simulator (for testing)."""

from easi.core.render_platform import EnvVars, HeadlessPlatform


class DummyCustomPlatform(HeadlessPlatform):
    """A trivial custom platform that adds a marker env var."""

    @property
    def name(self) -> str:
        return "dummy_custom"

    def get_env_vars(self) -> EnvVars:
        return EnvVars(replace={"DUMMY_CUSTOM_PLATFORM": "1"})
