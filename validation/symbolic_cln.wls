(* Exact scalar synthesis, port/caveat and notebook identity checks. *)
root=DirectoryName[DirectoryName[$InputFileName]];
package=FileNameJoin[{root,"src","cln_mcp_server","resources","SymbolicCLN.wl"}];
Get[package];
checks=<||>;check[label_,q_]:=AssociateTo[checks,label->TrueQ[q]];
zero[q_]:=TrueQ[FullSimplify[Together[q]]==0];
s=Unique["s"];
port=CLNSymbolic`InputPortChecks[];check["all input-port identities",port["passed"]];
KeyValueMap[(check["port: "<>#1,#2])&,port["checks"]];
caveats=CLNSymbolic`CaveatChecks[];
check["all caveat identities",caveats["passed"]];
KeyValueMap[(check["caveat: "<>#1,#2])&,caveats["checks"]];
z=1+4 s/(s+2)+9 s/(s+3);
elements=CLNSymbolic`FosterToCauer[z,s,5];
check["five-element exact termination",ListQ[elements] && Length[elements]==5];
check["exact input round trip",ListQ[elements] && zero[CLNSymbolic`CauerImpedance[elements,s]-z]];
Do[check["invalid budget "<>ToString[b,InputForm],FailureQ[CLNSymbolic`FosterToCauer[z,s,b]]],{b,{0,4,9/2}}];
check["approximate coefficients rejected",FailureQ[CLNSymbolic`FosterToCauer[1+4. s/(s+2),s]]];
check["zero terminal resistance",zero[CLNSymbolic`CauerImpedance[{{"R",1},{"L",1},{"R",0}},s]-1]];
fosterNumericModels={};
Do[
 e=MapIndexed[{If[OddQ[First[#2]],"R","L"],#1}&,values];
 f=CLNSymbolic`CauerToFoster[e,s];
 If[AssociationQ[f],AppendTo[fosterNumericModels,<|"cauer_values"->(ToString[#,InputForm]& /@ values),"dc"->N[f["dc"],17],"tail"->N[f["tail"],17],"branches"->N[f["branches"],17]|>]];
 check["Foster structure "<>ToString[Length[values]],AssociationQ[f] && Length[f["branches"]]==Floor[Length[values]/2]];
 If[AssociationQ[f],back=f["dc"]+f["tail"] s+Total[(#[[2]] s/(s+#[[1]]))& /@ f["branches"]];
  check["exact Foster reconstruction "<>ToString[Length[values]],zero[back-CLNSymbolic`CauerImpedance[e,s]]]],
 {values,{{1,1,1,1,1,1,1},{1,2,3,1,2,3,1,1,5}}}];
check["two-element termination",CLNSymbolic`CauerToFoster[{{"R",1},{"L",2}},s]===<|"dc"->1,"tail"->2,"branches"->{}|>];
(* Notebook identities and concrete conversion outputs for Python record consumers. *)
coords={x,y,t};grad[f_]:=D[f,#]& /@ coords;
curl[v_]:={D[v[[3]],y]-D[v[[2]],t],D[v[[1]],t]-D[v[[3]],x],D[v[[2]],x]-D[v[[1]],y]};
av={x y,y t,t x};phi=x y t;chi=x^2 y+t^3;
ef=-s av-grad[phi];bf=curl[av];tv={y t,x t,x y^2};omega=x t;
vectorZero[v_]:=And@@(zero /@ v);
check["gauge: A-phi E",vectorZero[-s (av+grad[chi])-grad[phi-s chi]-ef]];
check["gauge: A-phi B",vectorZero[curl[av+grad[chi]]-bf]];
check["gauge: T-Omega H",vectorZero[tv+grad[chi]+grad[omega-chi]-tv-grad[omega]]];
check["gauge: T-Omega J",vectorZero[curl[tv+grad[chi]]-curl[tv]]];
terminal=r0+s l1 (r2+tail)/(s l1+r2+tail);
required=Factor[tail/.First[Solve[terminal==target,tail]]];
check["termination inversion",zero[(terminal/.tail->required)-target]];
check["parallel identity",zero[ra s la/(ra+s la)-1/(1/ra+1/(s la))]];
textExact[q_]:=ToString[q,InputForm];
conversionExamples=Table[
 original=z+extra s;e=CLNSymbolic`FosterToCauer[original,s,6];f=CLNSymbolic`CauerToFoster[e,s];
 check[If[extra==0,"base concrete round trip","inductive-tail exact round trip"],
  ListQ[e] && AssociationQ[f] && zero[CLNSymbolic`CauerImpedance[e,s]-original]];
 <|"cauer"->Map[{#[[1]],textExact[#[[2]]]}&,e],"dc"->textExact[f["dc"]],
   "tail"->textExact[f["tail"]],"branches"->Map[textExact,f["branches"],{2}]|>,
 {extra,{0,1/5}}];
passed=And@@Values[checks];
If[Length[$ScriptCommandLine]<2,Print["Output JSON path required"];Exit[2]];
output=Last[$ScriptCommandLine];If[FileExistsQ[output],Print["Fresh output required"];Exit[3]];
result=<|"schema"->"cln.symbolic-migration.v1","passed"->passed,"checks"->checks,
 "scope"->"exact scalar RL synthesis and fixed SPD conjugate-port example; no FE/general passivity certification",
 "kernel"->$Version,"package_sha256"->IntegerString[FileHash[package,"SHA256"],16,64],
 "producer_sha256"->IntegerString[FileHash[$InputFileName,"SHA256"],16,64],
 "conversion_examples"->conversionExamples,"foster_numeric_models"->fosterNumericModels,
 "notebook_expressions"-><|"required_termination"->textExact[required]|>,
 "port_elements"->(ToString[#,InputForm]& /@ port["B_elements"])|>;
If[Export[output,result,"RawJSON"]===$Failed,Exit[4]];
If[passed,Print["PASS symbolic migration"];Exit[0],Print[result];Exit[1]];
