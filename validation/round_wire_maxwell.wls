(* Run: wolframscript -file validation/round_wire_maxwell.wls output.json
   The output parent must exist. Timings are observations, never pass criteria. *)
root = DirectoryName[DirectoryName[$InputFileName]];
package = FileNameJoin[{root,"src","cln_mcp_server","resources","RoundWireMaxwell.wl"}];
Get[package];
Clear[x,s];
zero[q_] := TrueQ[Together[q] === 0];
checks = {}; timings = {}; modes = <||>;
check[label_, q_] := AppendTo[checks,<|"name"->label,"passed"->TrueQ[q]|>];
Do[
  {elapsed, model} = AbsoluteTiming[WireModes[route,index,x,s]];
  AppendTo[timings,<|"route"->route,"max_index"->index,"seconds"->elapsed|>];
  check[route<>" construction index "<>ToString[index],AssociationQ[model]];
  If[AssociationQ[model],
    If[index==8, AssociateTo[modes,route->model]];
    If[index==0,check[route<>" resistance termination",zero[model["Z"]-1/Pi]]];
    If[index==1,check[route<>" inductance termination",zero[model["Z"]-(1/Pi+s/(8 Pi))]]]
  ],
  {route,{"E-H","A","T"}},{index,{0,1,8}}];
If[Length[modes]==3,
  reference=modes["E-H"];
  Do[
    m=modes[route];
    Do[check[route<>" vs E-H "<>key,
      And@@(zero /@ Flatten[{m[key]-reference[key]}])],
      {key,{"R","L","E","H","A","T","Z"}}],
    {route,{"A","T"}}];
  check["normalized R coefficients",reference["R"] Pi === {1,3,5,7,9}];
  check["normalized L coefficients",reference["L"] 8 Pi === {1,1/2,1/3,1/4}];
  Do[
    check["A/H differential map "<>ToString[k],zero[-D[reference["A"][[k]],x]-reference["H"][[k]]]];
    check["T/E differential map "<>ToString[k],zero[D[x reference["T"][[k]],x]/x-reference["E"][[k]]]];
    check["regular axis "<>ToString[k],zero[Limit[reference["H"][[k]],x->0]]];
    check["A boundary trace "<>ToString[k],zero[reference["A"][[k]] /. x->1]],
    {k,4}];
  Do[
    fields=reference[key];
    Do[check[key<>" orthogonality "<>ToString[{i,j}],
      zero[Integrate[2 Pi x fields[[i]] fields[[j]],{x,0,1}]]],
      {i,Length[fields]},{j,i-1}],
    {key,{"E","H"}}];
  (* The analytic oracle follows the regular Bessel solution, not ladder extraction. *)
  analytic=Normal[Series[Sqrt[s] BesselI[0,Sqrt[s]]/(2 BesselI[1,Sqrt[s]]),{s,0,8}]];
  check["analytic Bessel contact through degree 8",
    zero[Normal[Series[Pi reference["Z"],{s,0,8}]]-analytic]];
];
check["unknown route rejected",FailureQ[WireModes["unknown",8,x,s]]];
check["noninteger index rejected",FailureQ[WireModes["E-H",1/2,x,s]]];
check["oversized index rejected",FailureQ[WireModes["E-H",9,x,s]]];
passed=And@@Lookup[checks,"passed"];
result=<|"schema"->"cln.round-wire-maxwell.v1","passed"->passed,
  "kernel_version"->$Version,"kernel_system"->$SystemID,
  "package_sha256"->IntegerString[FileHash[package,"SHA256"],16,64],
  "normalization"->"x=r/a; a=mu=sigma=1; per unit axial length",
  "timings"->timings,"checks"->checks,
  "series_contact_degree"->8,
  "derive_stage_cap_timing"->"not measured: maxIndex is not the derive tool stage count",
  "expressions"->AssociationMap[
    Function[route,AssociationMap[Function[key,ToString[modes[route][key],InputForm]],
      {"R","L","E","H","A","T","Z"}]],Keys[modes]]|>;
If[Length[$ScriptCommandLine]<2,Print["Output JSON path required"];Exit[2]];
Export[Last[$ScriptCommandLine],result,"RawJSON"];
If[passed,Print["PASS round-wire Maxwell checks"];Exit[0],Print["FAIL round-wire Maxwell checks"];Exit[1]];
