"""Auto-resolve missing Python deps that HF transformers reports at load time.

Transformers raises a structured error when a remote-code modeling file
imports a package that's absent from the runtime, e.g.::

    "This modeling file requires the following packages that were not
     found in your environment: timm. Run `pip install timm`"

We parse that, install the requested package(s) from a static allowlist,
then retry the load. Avoids the per-VLM-swap dance of editing
``entrypoint.sh`` and rebuilding the container.

Hard constraints:
  * Allowlist-only — never install arbitrary packages a malicious model
    might request via ``trust_remote_code``.
  * Bounded retries (cap N attempts; the same package is never installed
    twice within one call chain).
  * Disabled when ``WEDA_AUTO_PIP=false`` so deployments that pin every
    dep at image-build time get no surprise side effects.
  * pip stdout/stderr captured into ``/tmp/dep_resolver.log`` for audit.

Usage::

    from app.modules.dep_resolver import auto_install

    @auto_install
    def load_model():
        return AutoModelForCausalLM.from_pretrained(...)
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from functools import wraps
from typing import Callable, Iterable, TypeVar

from app.utils.logger import get_logger

logger = get_logger()

T = TypeVar("T")
_MISSING_PACKAGES_RX = re.compile(
    r"requires the following packages? that (?:is|are|was|were) not found in your environment:\s*"
    r"([A-Za-z0-9_,\-\s]+?)(?=\.|\n|$)",
    re.IGNORECASE,
)

_AUTO_INSTALL_ALLOW: frozenset[str] = frozenset({
    "timm",
    "open_clip",
    "open_clip_torch",
    "torchvision",
    "einops",
    "sentencepiece",
    "tiktoken",
    "qwen_vl_utils",
    "decord",
    "av",
    "Pillow",
    "scipy",
    "accelerate",
    "tokenizers",
})

_AUTO_INSTALL_ALLOW: frozenset[str] = frozenset({
    "timm",
    "open_clip",
    "open_clip_torch",
    "torchvision",
    "einops",
    "sentencepiece",
    "tiktoken",
    "qwen_vl_utils",
    "decord",
    "av",
    "Pillow",
    "scipy",
    "accelerate",
    "tokenizers",
})

_IMPORT_TO_PIP: dict[str, str] = {
    "open_clip": "open_clip_torch",
    "qwen_vl_utils": "qwen-vl-utils",
    "av": "av",
}

_LOCK = threading.Lock()
_SESSION_INSTALLED: set[str] = set()
_LOG_PATH = "/tmp/dep_resolver.log"


def _enabled() -> bool:
    return os.environ.get("WEDA_AUTO_PIP", "true").lower() not in ("false", "0", "no")


def _missing_from_message(msg: str) -> list[str]:
    m = _MISSING_PACKAGES_RX.search(msg)
    if m is None:
        return []
    raw = m.group(1)
    return [p.strip() for p in re.split(r"[,\s]+", raw) if p.strip()]


def _install(pkg: str) -> bool:
    pip_name = _IMPORT_TO_PIP.get(pkg, pkg)
    logger.info("[dep_resolver] auto-installing %s (pip: %s)", pkg, pip_name)
    try:
        with open(_LOG_PATH, "a") as log:
            log.write(f"\n=== installing {pip_name} ===\n")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--no-cache-dir", pip_name],
                stdout=log, stderr=subprocess.STDOUT,
                timeout=300,
            )
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.error("[dep_resolver] pip install %s failed: %s", pip_name, e)
        return False
    if result.returncode != 0:
        logger.error(
            "[dep_resolver] pip install %s returned %d (see %s)",
            pip_name, result.returncode, _LOG_PATH,
        )
        return False
    logger.info("[dep_resolver] %s installed.", pip_name)
    return True


def auto_install(loader: Callable[..., T], *, max_retries: int = 6) -> Callable[..., T]:
    """Wrap a loader so missing-package errors auto-resolve via pip.

    The wrapped callable will:
      1. Call ``loader(*args, **kwargs)`` normally.
      2. On exception, parse the message for the transformers
         "requires the following packages…" diagnostic.
      3. For each requested package that's in the allowlist and hasn't
         been installed in this session, run ``pip install``.
      4. Retry the load up to ``max_retries`` times.
      5. Re-raise the original exception when retries are exhausted, the
         requested package isn't allowlisted, or pip itself fails.
    """
    @wraps(loader)
    def _wrapped(*args, **kwargs) -> T:
        if not _enabled():
            return loader(*args, **kwargs)

        local_attempts: dict[str, int] = {}
        for attempt in range(max_retries):
            try:
                return loader(*args, **kwargs)
            except Exception as exc:
                missing = _missing_from_message(str(exc))
                if not missing:
                    raise

                to_install: list[str] = []
                for pkg in missing:
                    if pkg not in _AUTO_INSTALL_ALLOW:
                        logger.error(
                            "[dep_resolver] '%s' is not in the auto-install allowlist. "
                            "Install manually or add to _AUTO_INSTALL_ALLOW.",
                            pkg,
                        )
                        raise

                    if local_attempts.get(pkg, 0) >= 1:
                        logger.error(
                            "[dep_resolver] '%s' was installed but the model "
                            "still reports it missing. Aborting.", pkg,
                        )
                        raise
                    local_attempts[pkg] = local_attempts.get(pkg, 0) + 1

                    with _LOCK:
                        if pkg in _SESSION_INSTALLED:
                            continue
                        _SESSION_INSTALLED.add(pkg)
                    to_install.append(pkg)

                for pkg in to_install:
                    if not _install(pkg):
                        raise

        raise RuntimeError(
            f"dep_resolver: max retries ({max_retries}) exceeded — "
            "model still reports missing packages."
        )

    return _wrapped
