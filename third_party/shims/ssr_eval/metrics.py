"""Import-only stub. See the package docstring."""


class AudioMetrics:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "ssr_eval is stubbed: the real package is not installable "
            "(it depends on mysql-python). A2SB only uses it for validation "
            "metrics, so reaching this means an evaluation path is running "
            "where only `predict` was expected."
        )
