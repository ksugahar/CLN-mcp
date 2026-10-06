(* Exact two-state proof of the physical Type1 field recursion. *)
ClearAll["Global`*"];
assumptions = l > 0 && a > 0 && b > 0 && c > 0 && a c > b^2;
simp[z_] := FullSimplify[z, assumptions];
k = IdentityMatrix[2]; m = {{a,b},{b,c}}; f = {Sqrt[l],0};
magnetic = LinearSolve[k,f]; electric = {0,0};
modes = {}; emodes = {}; elements = {};
Do[
  inductance = simp[magnetic.k.magnetic];
  electric = simp[electric + magnetic/inductance];
  resistance = simp[1/(electric.m.electric)];
  AppendTo[modes,magnetic]; AppendTo[emodes,electric];
  elements = Join[elements,{inductance,resistance}];
  magnetic = simp[magnetic-resistance LinearSolve[k,m.electric]],
  {2}];
z = simp[s f.LinearSolve[k+s m,f]];
ladder = elements[[-1]];
Do[ladder = If[OddQ[j],simp[1/(1/(s elements[[j]])+1/ladder)],
                                  simp[elements[[j]]+ladder]],
   {j,Length[elements]-1,1,-1}];
checks = <|
  "ladder_equals_field" -> TrueQ[simp[ladder-z] == 0],
  "magnetic_energy_orthogonality" -> TrueQ[simp[modes[[1]].k.modes[[2]]] == 0],
  "electric_loss_orthogonality" -> TrueQ[simp[emodes[[1]].m.emodes[[2]]] == 0],
  "positive_elements" -> And@@(TrueQ[simp[#>0]]& /@ elements),
  "expected_elements" -> TrueQ[simp[elements-{l,l/a,l b^2/a^2,l b^2/(a (a c-b^2))}] == {0,0,0,0}],
  "termination" -> TrueQ[simp[magnetic] == {0,0}],
  "peak_phasor_force_average" -> TrueQ[FullSimplify[
     Integrate[-(jr Cos[t]-ji Sin[t]) (br Cos[t]-bi Sin[t]),{t,0,2 Pi}]/(2 Pi)
     +(jr br+ji bi)/2] == 0]|>;
If[!And@@Values[checks],Print[checks];Exit[1]];
args = Rest[$ScriptCommandLine];
output = If[Length[args]>0,Last[args],"energy_recursion.json"];
Export[output,<|"scope"->"Exact SPD two-state Type1 proof; not a TEAM 28 force solution",
  "assumptions"->ToString[assumptions,InputForm],
  "elements_LR"->(ToString[#,InputForm]& /@ elements),
  "field_impedance"->ToString[z,InputForm],"checks"->checks,
  "wolfram_version"->$Version,
  "source_sha256"->FileHash[$InputFileName,"SHA256","HexString"]|>,"RawJSON"];
Print["PASS exact energy recursion, ladder identity, orthogonality and positivity"];
