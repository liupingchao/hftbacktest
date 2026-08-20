"""Stable errors for the Research Package Trust Kernel."""

from __future__ import annotations


class TrustKernelError(RuntimeError):
    """Fail-closed kernel error with a stable machine-readable code."""

    def __init__(self, code: str, location: str, detail: str) -> None:
        self.code = code
        self.location = location
        self.detail = detail
        super().__init__(f"{code} at {location}: {detail}")

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "location": self.location,
            "detail": self.detail,
        }
