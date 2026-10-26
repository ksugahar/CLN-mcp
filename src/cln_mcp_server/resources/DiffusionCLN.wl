(* ::Package:: *)
(* Physical (current/flux-source) CLN for  -div(a grad u) + b du/dt = f,  u = 0 on the outer boundary,
   on the symmetric 1-D domains: slab |x| < L (d = 1), disk/cylinder r < L (d = 2), sphere r < L (d = 3).
   Uniform source f = 1/|Omega| (unit total). Algorithm of Tanimoto et al., IEEJ SA-25-014 (2025).

   DiffusionCLN[d, n, a, b, L]  -> <|"R" -> {R0, R2, ...}, "C" -> {C1, C3, ...}, "modes" -> {That_1, That_3, ...} in \[FormalR]|>
   ExactZ[d, a, b, L][s]        -> exact port impedance  int f u / F  for uniform heating
   RCLadderZ[R, C, s]           -> 1/(1/R0 + 1/(1/(s C1) + 1/(1/R2 + ...)))
   Measures: d = 1 per unit area of the slab faces (both halves), d = 2 per unit length, d = 3 total. *)
BeginPackage["DiffusionCLN`"];
DiffusionCLN::usage = "DiffusionCLN[d, n, a, b, L] gives the RC-CLN elements of the first n stages.";
ExactZ::usage = "ExactZ[d, a, b, L] gives the exact port impedance as a pure function of s.";
RCLadderZ::usage = "RCLadderZ[R, C, s] evaluates the current-source RC ladder.";
Begin["`Private`"];

meas[d_] := Switch[d, 1, 2, 2, 2 Pi, 3, 4 Pi];                 (* |S^{d-1}| (slab: two halves) *)
vol[d_, L_] := Switch[d, 1, 2 L, 2, Pi L^2, 3, 4/3 Pi L^3];
int[d_, L_][g_] := meas[d] Integrate[g r^(d - 1), {r, 0, L}];
(* -a r^(1-d) (r^(d-1) u')' = q,  u(L) = 0, regular at r = 0 *)
solvePoisson[d_, a_, L_][q_] := Module[{flux, du, u},
  flux = -Integrate[Expand[q r^(d - 1)], r]/a;               (* r^(d-1) u' *)
  du = Expand[flux/r^(d - 1)];
  u = Integrate[du, r];
  Expand[u - (u /. r -> L)]];

DiffusionCLN[d_Integer, n_Integer, a_, b_, L_] := Module[{q, prev = 0, tt, rn, th, cn, R = {}, C = {}, modes = {}},
  q = 1/vol[d, L];
  Do[
   tt = solvePoisson[d, a, L][q];
   rn = Simplify[int[d, L][a D[tt, r]^2]];
   th = Expand[prev + tt/rn];
   cn = Simplify[int[d, L][b th^2]];
   AppendTo[R, rn]; AppendTo[C, cn]; AppendTo[modes, th /. r -> \[FormalR]];
   q = Expand[q - b th/cn]; prev = th,
   {n}];
  <|"R" -> R, "C" -> C, "modes" -> modes|>];

ExactZ[d_, a_, b_, L_] := Function[s, Module[{g = Sqrt[s b/a], f = 1/vol[d, L], mean},
    (* u = f/(b s) (1 - phi(r)/phi(L)) with phi = cosh, I0, sinh(g r)/r ; mean of phi over Omega *)
    mean = Switch[d,
      1, Tanh[g L]/(g L),
      2, 2 BesselI[1, g L]/(g L BesselI[0, g L]),
      3, 3 (g L Coth[g L] - 1)/(g L)^2];
    f/(b s) (1 - mean)]];

RCLadderZ[R_List, C_List, s_] := Module[{z = Last[R]},
  Do[z = 1/(1/R[[i]] + 1/(1/(s C[[i]]) + z)), {i, Length[R] - 1, 1, -1}]; z];

End[];
EndPackage[];
