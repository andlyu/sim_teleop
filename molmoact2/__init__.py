"""MolmoAct2 policy integration for sim_teleop.

Layout:
  adapter.py          state/action unit conversion (Genesis rad <-> model scale)
  client/             Mac-side, no GPU deps: Policy seam, StubPolicy, HTTP client
  server/             GPU-side: FastAPI host for MolmoAct2-SO100_101 + Vast setup
"""
