# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Version-compatible access to the PhysX tensor view API.

Isaac Sim 6 exposes the full view classes from ``omni.physics.tensors.api``.
Isaac Sim 5.1 exposes the same API from the older private ``impl.api`` path.
Keep this compatibility decision in one place so the shared Gina vendored source
can retain its validated 5.1 path while the 6.0.1 port is exercised.
"""

try:
    import omni.physics.tensors.api as physx
except ImportError:  # Isaac Sim 5.1 compatibility
    import omni.physics.tensors.impl.api as physx

__all__ = ["physx"]
