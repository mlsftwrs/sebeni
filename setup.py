import os
import shutil
import stat
import sys
import urllib.request
from pathlib import Path
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install


def _install_kveritas_if_missing():
    """Auto-install k-veritas binary if not already available on PATH."""
    if shutil.which("kveritas") is not None:
        return

    # Only auto-download for Linux x86_64 / amd64
    if sys.platform != "linux":
        return

    url = "https://github.com/27-GROUP/kveritas-releases/raw/main/bin/kveritas-linux-amd64"
    print(f"Installing kveritas from {url}...")

    # Determine candidate target directories
    target_dirs = [
        Path(sys.prefix) / "bin",
        Path("/usr/local/bin"),
        Path.home() / ".local" / "bin",
    ]

    installed = False
    for tdir in target_dirs:
        try:
            tdir.mkdir(parents=True, exist_ok=True)
            target_bin = tdir / "kveritas"
            req = urllib.request.Request(url, headers={"User-Agent": "sebeni-installer"})
            with urllib.request.urlopen(req, timeout=30) as resp, open(target_bin, "wb") as f:
                f.write(resp.read())
            current_mode = target_bin.stat().st_mode
            target_bin.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            print(f"kveritas successfully installed to {target_bin}")
            installed = True
            break
        except Exception as exc:
            # If permission denied or directory unwritable, attempt next candidate
            continue

    if not installed:
        print("Note: Could not auto-install kveritas to PATH. You can install it manually from:", url)


class PostInstallCommand(install):
    """Post-installation for installation mode."""

    def run(self):
        super().run()
        try:
            _install_kveritas_if_missing()
        except Exception as e:
            print(f"Warning during kveritas installation: {e}")


class PostDevelopCommand(develop):
    """Post-installation for development mode."""

    def run(self):
        super().run()
        try:
            _install_kveritas_if_missing()
        except Exception as e:
            print(f"Warning during kveritas installation: {e}")


setup(
    cmdclass={
        "install": PostInstallCommand,
        "develop": PostDevelopCommand,
    }
)
