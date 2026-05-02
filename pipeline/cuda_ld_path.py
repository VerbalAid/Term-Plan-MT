"""Make pip-installed NVIDIA shared libraries visible before ``import torch`` / bitsandbytes.

PyTorch+cu130 wheels expect ``libnvJitLink.so.13`` at dynamic-load time. The wheel ships it
under ``site-packages/nvidia/cu13/lib`` (and related paths). Without prepending those
directories to ``LD_LIBRARY_PATH``, bitsandbytes fails with missing ``libnvJitLink``.

Call :func:`ensure_cuda_pip_libs_visible` once at process startup, before importing torch.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

_DONE = False


def site_packages_roots() -> list[Path]:
    """Likely ``site-packages`` dirs (venv layouts differ; ``site.getsitepackages()`` can miss one)."""
    import site

    roots: list[Path] = []
    for r in (*site.getsitepackages(), site.getusersitepackages()):
        if r:
            roots.append(Path(r).resolve())
    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    for base in (Path(sys.prefix) / "lib", Path(sys.prefix) / "lib64"):
        sp = base / f"python{ver}" / "site-packages"
        if sp.is_dir():
            roots.append(sp.resolve())
    seen: set[str] = set()
    out: list[Path] = []
    for p in roots:
        s = str(p)
        if s not in seen:
            seen.add(s)
            out.append(p)
    return out


def nvidia_lib_dirs_with_nvjitlink() -> list[Path]:
    """Directories under site-packages that contain ``libnvJitLink.so*`` (prefer cu13)."""
    dirs: list[Path] = []
    for base in site_packages_roots():
        for rel in ("nvidia/cu13/lib", "nvidia/nvjitlink/lib", "nvidia/cuda_nvjitlink/lib"):
            d = base / rel
            if d.is_dir() and any(d.glob("libnvJitLink.so*")):
                rd = d.resolve()
                if rd not in dirs:
                    dirs.append(rd)
        nvidia = base / "nvidia"
        if nvidia.is_dir():
            for child in nvidia.iterdir():
                lib = child / "lib"
                if lib.is_dir() and any(lib.glob("libnvJitLink.so*")):
                    rd = lib.resolve()
                    if rd not in dirs:
                        dirs.append(rd)
    return dirs


def ensure_cuda_pip_libs_visible() -> None:
    """Prepend nvJitLink dirs to ``LD_LIBRARY_PATH`` and preload ``libnvJitLink.so.13`` when present."""
    global _DONE
    if _DONE:
        return
    extra = [str(p) for p in nvidia_lib_dirs_with_nvjitlink()]
    if not extra:
        _DONE = True
        return
    prev = os.environ.get("LD_LIBRARY_PATH", "")
    parts = extra + ([prev] if prev else [])
    os.environ["LD_LIBRARY_PATH"] = ":".join(parts)
    try:
        RTLD_GLOBAL = ctypes.RTLD_GLOBAL  # type: ignore[attr-defined]
    except AttributeError:
        RTLD_GLOBAL = 256
    mode = ctypes.DEFAULT_MODE | RTLD_GLOBAL
    for d in extra:
        for name in ("libnvJitLink.so.13", "libnvJitLink.so.12"):
            so = Path(d) / name
            if so.is_file():
                try:
                    ctypes.CDLL(str(so), mode=mode)
                except OSError:
                    pass
                break
    _DONE = True
