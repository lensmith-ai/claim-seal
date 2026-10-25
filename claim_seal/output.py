"""Writing a result file without clobbering or following links.

write_text() creates a NEW regular file (O_CREAT | O_EXCL | O_NOFOLLOW, mode 0600 before the
umask). With force=True it may replace an existing REGULAR file. A symlink is always refused,
force or not, and so is anything that is not a regular file (a directory, a FIFO, a device).
Only the last path component is protected from links; the directories leading to it are the
caller's choice.
"""
from __future__ import annotations

import errno
import os
import stat


class OutputRefused(ValueError):
    """The output path is a symlink, not a regular file, or exists and force was not given."""


_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)   # a FIFO with no reader fails instead of hanging
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)


def check_out_path(path: str, *, force: bool = False) -> None:
    """Refuse early, before any work is done for a file that could not be written."""
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(st.st_mode):
        raise OutputRefused(f"{path} is a symlink; refusing to write through it (--force does "
                            "not change this)")
    if not stat.S_ISREG(st.st_mode):
        raise OutputRefused(f"{path} exists and is not a regular file")
    if not force:
        raise OutputRefused(f"{path} already exists; pass --force to replace it")


def write_text(path: str, text: str, *, force: bool = False, _skip_precheck: bool = False) -> None:
    """Write `text` (UTF-8) to `path` under the rules in the module docstring."""
    if not _skip_precheck:          # tests skip it to prove the open itself refuses a link
        check_out_path(path, force=force)
    flags = os.O_WRONLY | os.O_CREAT | _NOFOLLOW | _NONBLOCK | _CLOEXEC
    flags |= os.O_TRUNC if force else os.O_EXCL
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError:
        raise OutputRefused(f"{path} already exists; pass --force to replace it") from None
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.EMLINK, errno.ENXIO):
            raise OutputRefused(f"{path} is a symlink or not a regular file; refusing to write") from None
        raise
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OutputRefused(f"{path} is not a regular file")
    except BaseException:
        os.close(fd)
        raise
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
