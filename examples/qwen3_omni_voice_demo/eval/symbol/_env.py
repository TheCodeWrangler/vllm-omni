"""Import this FIRST, before transformers or librosa, from every entry point.

Two environment landmines, both of which fail at import time and are easy to hit:

1. `transformers` eagerly imports `torchaudio` (for Parakeet's RNNT loss). A broken torchaudio
   build then takes down anything that imports transformers. Nothing here uses torchaudio, so
   stub it rather than requiring a working install.
2. `librosa.pyin` / `librosa.effects.split` compile through numba, which needs a writable cache
   directory. The default sits next to the installed package and is often not writable.
"""
import importlib.machinery
import os
import sys
import types

_m = types.ModuleType("torchaudio")
_m.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)
sys.modules.setdefault("torchaudio", _m)

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
