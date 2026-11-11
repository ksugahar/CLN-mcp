(* Symmetric slab, unit sheet current in normalized coordinates. *)
SetSystemOptions["ParallelOptions" -> "ParallelThreadNumber" -> 1];

ClearAll[x, u, makeModel, impedance, highBasis, peelHigh, band, errors];

makeModel[n_] := Module[{j, h, a, r, l},
  j = Table[LegendreP[2 k, x], {k, 0, n - 1}];
  h = Expand[Integrate[#, x] - (Integrate[#, x] /. x -> 0)] & /@ j;
  a = Expand[(Integrate[#, x] /. x -> 1) - Integrate[#, x]] & /@ h;
  r = DiagonalMatrix[Table[1/(4 k + 1), {k, 0, n - 1}]];
  l = Table[With[{p = Expand[h[[i]] h[[jIndex]]]},
    Integrate[p, {x, 0, 1}]], {i, n}, {jIndex, n}];
  {r, l, UnitVector[n, 1], {j, a, h}}
];

impedance[r_, l_, b_] := Together[1/(b . LinearSolve[r + u l, b])];

(* Rational, unnormalized L-orthogonal Krylov basis. *)
highBasis[r_, l_, b_] := Module[{solve = LinearSolve[l], q, cols = {}},
  q = solve[b];
  Do[
    If[q == ConstantArray[0, Length[b]], Return[$Failed]];
    AppendTo[cols, q];
    If[k < Length[b],
      q = solve[r . q];
      Do[q = q - v ((v . l . q)/(v . l . v)), {v, cols}]
    ],
    {k, Length[b]}
  ];
  Transpose[cols]
];


(* Alternating magnetic/electric energy recurrence, with the original port b.
   K=L and N=R in the reversed pencil; signs follow the accumulated update. *)
energyBasis[r_, l_, b_] := Module[
  {a = LinearSolve[l, b], e, inductance, resistance, cols = {}},
  Do[
    AppendTo[cols, a];
    inductance = 1/(a . l . a);
    If[k == 1, e = -a inductance, e = e - a inductance];
    resistance = e . r . e;
    If[k < Length[b], a = a + LinearSolve[l, r . e]/resistance],
    {k, Length[b]}];
  Transpose[cols]
];

(* Independent synthesis: extract series L and shunt R at infinity. *)
peelHigh[z_, n_] := Module[{tail = z, ind, res, rem, pairs = {}},
  Do[
    ind = Limit[tail/u, u -> Infinity];
    rem = Together[tail - u ind];
    res = Limit[rem, u -> Infinity];
    If[!TrueQ[ind > 0 && res > 0], Return[$Failed]];
    AppendTo[pairs, {ind, res}];
    If[k < n, tail = Together[1/(1/rem - 1/res)],
      If[Together[rem - res] =!= 0, Return[$Failed]]],
    {k, n}];
  pairs
];

fromPairs[pairs_] := Fold[#2[[1]] u + 1/(1/#2[[2]] + 1/#1) &,
  Last[pairs] . {u, 1}, Reverse[Most[pairs]]];

zExact[w_] := With[{k = Sqrt[I w]}, k Coth[k]];
relativeError[z_, w_] := Abs[(z /. u -> I w)/zExact[w] - 1.];

(* Report all detected intervals on a bounded scan, refining each crossing. *)
band[z_, points_] := Module[{flags, crossings, edges, lo, hi, mid},
  flags = (relativeError[z, #] <= .01 &) /@ points;
  crossings = Table[
    If[flags[[k]] === flags[[k + 1]], Nothing,
      {lo, hi} = Log[points[[{k, k + 1}]]];
      Do[mid = (lo + hi)/2;
        If[(relativeError[z, Exp[mid]] <= .01) === flags[[k]], lo = mid, hi = mid],
        {40}];
      Exp[(lo + hi)/2]],
    {k, Length[points] - 1}];
  edges = Partition[Join[{First[points]}, crossings, {Last[points]}], 2, 1];
  Pick[edges, Table[OddQ[k] === First[flags], {k, Length[edges]}], True]
];

fields[r_, l_, b_, q_, w_, basis_, grid_] := Module[{g, co},
  g = LinearSolve[N[Transpose[q] . (r + I w l) . q, 35],
    Transpose[q] . b];
  co = q . (g/(b . q . g));
  Table[N[(row . co) /. x -> grid, 25], {row, basis}]
];

exactFields[w_, grid_] := Module[{k = Sqrt[I w]},
  N[{k Cosh[k grid]/Sinh[k],
     (Cosh[k] - Cosh[k grid])/(k Sinh[k]), Sinh[k grid]/Sinh[k]}, 25]
];

errors[approx_, exact_] := MapThread[Norm[#1 - #2]/Norm[#2] &, {approx, exact}];

verifySlab[] := Module[
  {n = 8, r, l, b, basis, q, qt, rh, lh, bh, zl, zh, pairs, match,
   grid, frequencies, points, fieldRows, unitFixture, alternate, checks, qe, ze, lowCircuit, profile},
  {r, l, b, basis} = makeModel[n];
  q = highBasis[r, l, b];
  If[q === $Failed, Return[<|"ok" -> False, "error" -> "Krylov breakdown"|>]];
  qt = Transpose[q];
  qe = energyBasis[r, l, b];
  ze = Table[With[{v = qe[[All, 1 ;; m]]}, impedance[Transpose[v] . r . v, Transpose[v] . l . v, Transpose[v] . b]], {m, n}];
  lowCircuit = Fold[(4 #2 + 1) + 1/(1/(u/(4 #2 + 3)) + 1/#1) &, (4 n - 3) + u/(4 n - 1), Reverse[Range[0, n - 2]]];
  {rh, lh, bh} = {qt . r . q, qt . l . q, qt . b};
  zl = Table[impedance[r[[1 ;; m, 1 ;; m]], l[[1 ;; m, 1 ;; m]], b[[1 ;; m]]], {m, n}];
  zh = Table[impedance[rh[[1 ;; m, 1 ;; m]], lh[[1 ;; m, 1 ;; m]], bh[[1 ;; m]]], {m, n}];
  pairs = peelHigh[Last[zl], n];
  If[pairs === $Failed, Return[<|"ok" -> False, "error" -> "Continued-fraction breakdown"|>]];
  match = Table[Together[zh[[m]] - fromPairs[Take[pairs, m]]] === 0, {m, n}];
  unitFixture = 1/(1/u + 1/(1 + 1/(1/u + 1)));
  alternate = 1/(1/2 + 1/(4 u/5 + 1/(25/2 + 5/u)));
  grid = Range[-100, 100]/100;
  frequencies = {1/10, 1, 10, 100, 1000};
  points = N[10^Range[-4, 5, 1/200]];
  fieldRows = Table[Module[{reference, low, high, shortLow, shortHigh},
    reference = exactFields[w, grid];
    low = fields[r, l, b, IdentityMatrix[n], w, basis, grid];
    high = fields[r, l, b, q, w, basis, grid];
    shortLow = fields[r, l, b, IdentityMatrix[n][[All, 1 ;; 3]], w, basis, grid];
    shortHigh = fields[r, l, b, q[[All, 1 ;; 3]], w, basis, grid];
    <|"omega_tau" -> N[w], "full_I" -> N[errors[low, reference]],
      "full_II" -> N[errors[high, reference]], "three_I" -> N[errors[shortLow, reference]],
      "three_II" -> N[errors[shortHigh, reference]],
      "full_I_vs_II" -> N[errors[high, low]],
      "profiles" -> If[w == 100, <|"x" -> grid, "exact" -> ({Re[#], Im[#]} & /@ reference), "I" -> ({Re[#], Im[#]} & /@ shortLow), "II" -> ({Re[#], Im[#]} & /@ shortHigh)|>, Null]|>], {w, frequencies}];
  checks = <|"energy_and_lanczos_prefixes_exact" -> And @@ Table[Together[ze[[m]] - zh[[m]]] === 0, {m, n}],
    "low_circuit_matches_galerkin" -> (Together[lowCircuit - Last[zl]] === 0),
    "basis_ampere" -> And @@ (Expand[#] === 0 & /@ ((D[#, x] & /@ basis[[3]]) - basis[[1]])),
    "basis_potential_curl" -> And @@ (Expand[#] === 0 & /@ ((-D[#, x] & /@ basis[[2]]) - basis[[3]])),
    "full_impedance_exact" -> (Together[Last[zl] - Last[zh]] === 0),
    "all_high_prefixes_match_independent_synthesis" -> And @@ match,
    "high_l_diagonal" -> (lh === DiagonalMatrix[Diagonal[lh]]),
    "high_r_tridiagonal" -> (And @@ Flatten[Table[Abs[i - j] <= 1 || rh[[i, j]] === 0, {i, n}, {j, n}]]),
    "full_fields_agree" -> (Max[Flatten[(#["full_I_vs_II"] &) /@ fieldRows]] < 10^-20),
    "separate_two_stage_fixture_exact" -> (Together[unitFixture - alternate] === 0)|>;
  <|"ok" -> And @@ Values[checks], "checks" -> checks, "runtime" -> $Version,
    "field_order" -> {"J", "A", "H"}, "spatial_samples" -> Length[grid],
    "field_errors" -> fieldRows, "scan_points" -> Length[points],
    "band_threshold" -> .01, "bands" -> Table[
      <|"stages" -> m, "I" -> band[zl[[m]], points], "II" -> band[zh[[m]], points]|>,
      {m, n}], "impedance_samples" -> Table[<|"omega_tau" -> w, "I_errors" -> Table[relativeError[zl[[m]], w], {m, n}], "II_errors" -> Table[relativeError[zh[[m]], w], {m, n}]|>, {w, points[[1 ;; ;; 10]]}],
    "high_L_R_pairs" -> (ToString[#, InputForm] & /@ pairs)|>
];

result = verifySlab[];
Print[StringReplace[ExportString[result, "RawJSON", "Compact" -> True], RegularExpression["0\\.e([+-]?[0-9]+)"] -> "0.0e$1"]];
Exit[If[TrueQ[result["ok"]], 0, 1]];
