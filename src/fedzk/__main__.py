# Copyright (c) 2025 Aaryan Guglani and FedZK Contributors
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 Aaryan Guglani and FedZK Contributors
# SPDX-License-Identifier: MIT

"""
Main entry point for the FedZK package.

This module redirects to the CLI implementation in cli.py.
"""

from fedzk.cli import main

app = main

if __name__ == "__main__":
    main()
