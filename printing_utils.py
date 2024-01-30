"""
Helper functions to make training/verification output prettier.
"""

import contextlib
import os
import sys


@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as fnull:
        original_stdout = sys.stdout
        sys.stdout = fnull
        try:
            yield
        finally:
            sys.stdout = original_stdout


def delete_last_line():
    """Deletes the last line in the STDOUT"""
    sys.stdout.write('\x1b[1A')
    sys.stdout.write('\x1b[2K')
