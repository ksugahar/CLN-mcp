"""3D round-conductor CLN, T-Omega formulation (current vector potential T, reduced magnetic scalar potential Omega): maintained in CLN-mcp.

This development copy was superseded and is intentionally not executable.
The maintained implementation is validation/cln3d/CLN_T_Omega.py in the public CLN-mcp
repository (https://github.com/ksugahar/CLN-mcp), together with its shared
solver helper validation/cln3d/_iccg.py, the six-case check
validation/validate_3d.py and the notebook docs/06_3d_round_wire.ipynb.

Why the public version: it pairs each vector space with a scalar space of
order + 1 (so the gradient kernel lies in the test space), uses matched
quadrature dx(bonus_intorder=4) on curved elements, adds a gradient penalty
alpha G G^T, and rejects a projected-out load component above 1e-8. The
reviewed baseline is one stage (R0, L1, R2) at s0 = 0; additional verified
three-stage round-wire settings are recorded in docs/10_research_followup.ipynb
and docs/data/higher_stages.json. There is no universal higher-stage guarantee.

Original notebook by N. Tanimoto.
"""

raise SystemExit("Superseded: run validation/cln3d/CLN_T_Omega.py from the public CLN-mcp repository.")