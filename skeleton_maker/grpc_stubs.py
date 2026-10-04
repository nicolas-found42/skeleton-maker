# SPDX-License-Identifier: MIT
"""Regenerate the vendored gRPC stubs from the bundled protos.

The stubs in ``skeleton_maker/_gen`` are generated at install or first use, so
they are not committed. Run this after editing anything under ``proto/``:

    python -m skeleton_maker.grpc_stubs

Requires ``grpcio-tools`` (the ``dev`` extra).
"""

from .nim import GEN_DIR, _generate_stubs

if __name__ == "__main__":
    _generate_stubs()
    print(f"stubs written to {GEN_DIR}")
