# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
# Copyright (c) 2025 Aaryan Guglani and FEDzk Contributors
# Licensed under FSL-1.1-Apache-2.0. See LICENSE for details.

"""
Main entry point for the FEDzk package.

This module redirects to the CLI implementation in cli.py.
"""

from fedzk.cli import main

app = main

if __name__ == "__main__":
    main()
