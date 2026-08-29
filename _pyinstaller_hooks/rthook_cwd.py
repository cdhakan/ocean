# PyInstaller runtime hook — set writable CWD before app starts
import sys, os
from pathlib import Path
if getattr(sys, 'frozen', False):
    _out = Path.home() / 'Documents' / 'OCEAN_Output'
    _out.mkdir(parents=True, exist_ok=True)
    os.chdir(_out)
