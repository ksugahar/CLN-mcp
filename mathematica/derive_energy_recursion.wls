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
(* Reproducible truncated rational examples: non-identity K and singular M.
   These are exact instances, not a general proof for arbitrary n<N. *)
truncatedCase[kdiag_,mdiag_,steps_] := Module[
  {kk=DiagonalMatrix[kdiag],mm=DiagonalMatrix[mdiag],src,aa,ee,ar={},er={},el={},
   ll,rr,qq,ff,zg,zf,zl,j,series,tests},
  src=ConstantArray[1,Length[kdiag]]; aa=LinearSolve[kk,src]; ee=0 src;
  Do[ll=aa.kk.aa; ee=ee+aa/ll; rr=1/(ee.mm.ee);
     AppendTo[ar,aa]; AppendTo[er,ee]; el=Join[el,{ll,rr}];
     aa=aa-rr LinearSolve[kk,mm.ee],{steps}];
  qq=Transpose[ar]; ff=Transpose[qq].src;
  zg=Together[s ff.LinearSolve[Transpose[qq].kk.qq+s Transpose[qq].mm.qq,ff]];
  zf=Together[s src.LinearSolve[kk+s mm,src]];
  zl=Last[el];
  Do[zl=If[OddQ[j],Together[1/(1/(s el[[j]])+1/zl)],Together[el[[j]]+zl]],
     {j,Length[el]-1,1,-1}];
  series=Normal[Series[zf-zg,{s,0,2 steps+1}]];
  tests=<|"ladder_equals_galerkin"->TrueQ[Together[zl-zg]==0],
    "positive_elements"->And@@(#>0& /@ el),
    "magnetic_orthogonality"->TrueQ[Transpose[qq].kk.qq==DiagonalMatrix[el[[1;; ;;2]]]],
    "electric_orthogonality"->TrueQ[er.mm.Transpose[er]==DiagonalMatrix[1/el[[2;; ;;2]]]],
    "nonzero_remaining_field"->TrueQ[aa.kk.aa>0],
    "moment_contact"->(And@@Table[TrueQ[Coefficient[series,s,j]==0],{j,0,2 steps}] &&
                          TrueQ[Coefficient[series,s,2 steps+1]!=0])|>;
  If[!And@@Values[tests],Print[tests];Exit[1]];
  <|"K_diagonal"->kdiag,"M_diagonal"->mdiag,"source"->src,
    "states"->steps,"full_dimension"->Length[src],"first_error_power"->2 steps+1,
    "elements_LR"->(ToString[#,InputForm]& /@ el),"checks"->tests|>];
instances={truncatedCase[{2,3,5,7,11},{1,2,3,0,0},2],
           truncatedCase[{2,3,5,7,11,13},{1,2,3,4,0,0},3]};
args = Rest[$ScriptCommandLine];
output = If[Length[args]>0,Last[args],"energy_recursion.json"];
record=<|"scope"->"Exact exhausted n=N=2 SPD Type1 proof: K=I and f along an axis are without loss of generality by congruence and rotation. The symbolic part does not prove general truncated n<N or PSD M; exact truncated instances are recorded separately; not a TEAM 28 force solution",
  "assumptions"->ToString[assumptions,InputForm],
  "elements_LR"->(ToString[#,InputForm]& /@ elements),
  "field_impedance"->ToString[z,InputForm],"checks"->checks,
  "truncated_instances_scope"->"Two exact rational truncated cases with PSD M and non-identity K; no general theorem",
  "truncated_instances"->instances,
  "wolfram_version"->$Version,
  "source_sha256"->FileHash[$InputFileName,"SHA256","HexString"]|>;
(* Binary UTF-8 output pins LF on every platform, including Windows. *)
stream=OpenWrite[output,BinaryFormat->True];
BinaryWrite[stream,ToCharacterCode[StringReplace[ExportString[record,"RawJSON"],"\r\n"->"\n"],"UTF-8"],"Byte"];
Close[stream];
Print["PASS exact energy recursion, ladder identity, orthogonality and positivity"];
