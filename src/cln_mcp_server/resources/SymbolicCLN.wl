(* Exact scalar synthesis and input-port identities; no FE or general passivity claim. *)
BeginPackage["CLNSymbolic`"];
CauerImpedance::usage="CauerImpedance[elements,s] reconstructs an exact alternating series-R/shunt-L impedance.";
FosterToCauer::usage="FosterToCauer[z,s,budget] extracts a finite DC ladder or returns Failure.";
CauerToFoster::usage="CauerToFoster[elements,s] returns exact dc, tail and positive (pole-rate,resistance) branches.";
InputPortChecks::usage="InputPortChecks[] checks a matched three-state conjugate port by independent rational inversion.";
CaveatChecks::usage="CaveatChecks[] checks incompatible magnetic boundary, shifted DC and Euclidean remainder counterexamples.";
Begin["`Private`"];
fail[msg_] := Throw[Failure["CLNSymbolic",<|"MessageTemplate"->msg|>],"CLNSymbolic"];
exact[q_] := If[!FreeQ[q,_Real],fail["Exact coefficients required"],q];
zero[q_] := TrueQ[FullSimplify[Together[q]]==0];
positive[q_] := TrueQ[FullSimplify[q>0]];
nonnegative[q_] := TrueQ[FullSimplify[q>=0]];
CauerImpedance[elements_List,s_Symbol] := Catch[Module[{e=exact[elements],z,j},
 If[Length[e]==0,fail["Empty circuit"]];
 Do[If[!MatchQ[e[[j]],{_String,_}] || e[[j,1]]!=If[OddQ[j],"R","L"],fail["Alternating R/L required"]];
    If[If[OddQ[j],!nonnegative[e[[j,2]]],!positive[e[[j,2]]]],fail["Nonnegative R and positive L required"]],{j,Length[e]}];
 z=If[e[[-1,1]]=="R",e[[-1,2]],s e[[-1,2]]];
 Do[z=Together[If[e[[j,1]]=="R",e[[j,2]]+z,s e[[j,2]] z/(s e[[j,2]]+z)]],{j,Length[e]-1,1,-1}];z
],"CLNSymbolic"];
FosterToCauer[z0_,s_Symbol,budget_:32] := Catch[Module[{z=Together[exact[z0]],tail,e={},r,l,y,remaining,back},
 If[!IntegerQ[budget] || budget<1,fail["Positive integer budget required"]];tail=z;
 While[Length[e]<budget,
  r=FullSimplify[Limit[tail,s->0]];If[!nonnegative[r] || !FreeQ[r,Indeterminate|DirectedInfinity[_]],fail["Irregular DC impedance"]];
  AppendTo[e,{"R",r}];remaining=Together[tail-r];
  If[zero[remaining],back=CauerImpedance[e,s];If[!FailureQ[back] && zero[back-z],Return[e],fail["Round trip failed"]]];
  If[Length[e]==budget,Break[]];y=Together[1/remaining];l=FullSimplify[Limit[s y,s->0]];
  If[!positive[l],fail["Positive inductive extraction required"]];AppendTo[e,{"L",1/l}];remaining=Together[y-l/s];
  If[zero[remaining],back=CauerImpedance[e,s];If[!FailureQ[back] && zero[back-z],Return[e],fail["Round trip failed"]]];
  tail=Together[1/remaining]
 ];fail["Element budget exhausted before termination"]
],"CLNSymbolic"];
CauerToFoster[e_List,s_Symbol] := Catch[Module[{z,dc,tail,q,num,den,poles,branches,derivative,rr,back},
 z=CauerImpedance[e,s];If[FailureQ[z],Return[z]];
 dc=FullSimplify[Limit[z,s->0]];tail=FullSimplify[Limit[z/s,s->Infinity]];
 If[!nonnegative[dc] || !nonnegative[tail],fail["Negative or undecidable static term"]];q=Together[z-dc-tail s];
 If[zero[q],Return[<|"dc"->dc,"tail"->tail,"branches"->{}|>]];
 num=Numerator[q];den=Denominator[q];
 If[Exponent[PolynomialGCD[den,D[den,s]],s]>0,fail["Simple poles required"]];
 poles=s/.Solve[den==0,s,Reals];
 If[Length[poles]!=Exponent[den,s] || !And@@(positive[-#]& /@ poles),fail["Simple negative real poles required"]];
 derivative=D[den,s];branches=Table[
  rr=FullSimplify[(num/(s derivative))/.s->pole];If[!positive[rr],fail["Positive branch resistance required"]];{-pole,rr},{pole,poles}];
 back=dc+tail s+Total[(#[[2]] s/(s+#[[1]]))& /@ branches];
 If[!zero[back-z],fail["Partial fraction round trip failed"]];
 <|"dc"->dc,"tail"->tail,"branches"->SortBy[branches,First]|>
],"CLNSymbolic"];
InputPortChecks[] := Module[{s,km={{4,1,1},{1,3,1},{1,1,2}},nm=DiagonalMatrix[{1,2,3}],kf,nf,lift,c,m,load,correction,total,res,yb,ye,zh,eb,ab,rb,lb,ah,eh,lh,rh,be,he,xe,tail,ll,rr,compact,checks=<||>,check,j},
 check[label_,q_]:=AssociateTo[checks,label->TrueQ[q]];
 kf=km[[2;;,2;;]];nf=nm[[2;;,2;;]];lift=Prepend[-LinearSolve[kf,km[[2;;,1]]],1];
 c=lift.km.lift;m=lift.nm.lift;load=(nm.lift)[[2;;]];
 correction=-LinearSolve[kf+s nf,s load];total=lift+Prepend[correction,0];res=(km+s nm).total;
 check["free residual",And@@(zero /@ Rest[res])];check["static lift orthogonality",zero[lift.km.Prepend[correction,0]]];
 yb=Together[First[res]/s];ye=Together[m-s load.LinearSolve[kf+s nf,load]];
 zh=Together[s UnitVector[3,1].LinearSolve[km+s nm,UnitVector[3,1]]];
 check["lift decomposition",zero[yb-c/s-ye]];check["matched conjugate duality",zero[yb zh-1]];
 eb=lift;ab=0 lift;rb=1/m;be={1/c,rb};
 Do[ab=Together[ab+Prepend[LinearSolve[kf,rb (nm.eb)[[2;;]]],0]];lb=Together[ab.km.ab];
    check["B positive L "<>ToString[j],positive[lb]];eb=Together[eb-ab/lb];rb=Together[1/(eb.nm.eb)];be=Join[be,{lb,rb}],{j,2}];
 ah=LinearSolve[km,UnitVector[3,1]];eh=0 lift;lh=ah.km.ah;he={lh};
 Do[eh=Together[eh-ah/lh];rh=Together[1/(eh.nm.eh)];AppendTo[he,rh];
    If[j<3,ah=Together[LinearSolve[km,km.ah+rh nm.eh]];lh=Together[ah.km.ah];AppendTo[he,lh]],{j,3}];
 tail=zh;xe={};Do[ll=Together[1/Limit[s/tail,s->0]];tail=Together[1/(1/tail-1/(s ll))];rr=Limit[tail,s->0];xe=Join[xe,{ll,rr}];tail=Together[tail-rr],{3}];
 check["scalar extraction terminates",zero[tail]];check["B/H two-index shift",And@@(zero /@ (be-he))];check["recurrence vs rational extraction",And@@(zero /@ (be-xe))];
 compact=Together[lift.km.LinearSolve[km+s nm,nm.lift]];
 check["different port is different",!zero[{1,1}.LinearSolve[kf,{1,1}]-1/c]];
 check["full compact-product counterexample",!zero[compact-ye]];
 check["SPD leading minors",And@@Flatten[Table[Det[a[[1;;j,1;;j]]]>0,{a,{km,nm}},{j,3}]]];
 <|"checks"->checks,"passed"->And@@Values[checks],"B_elements"->be,"H_elements"->he,"scalar_elements"->xe,
   "Y_B"->yb,"Y_E_response"->ye,"Z_H"->zh,"L_excitation"->1/c|>
];
CaveatChecks[] := Module[{r,s,imposed,defect,qr,checks},
 imposed=(r-1)/2;defect=Together[D[r imposed,r]/r-1];
 qr=PolynomialQuotientRemainder[s-1,s+1,s];
 checks=<|
  "regular Ampere solution"->zero[D[r (r/2),r]/r-1],
  "incompatible arbitrary magnetic boundary"->zero[defect+1/(2 r)],
  "negative physical DC of shifted branch"->zero[((s-1)/.s->0)+1],
  "Euclidean quotient"->zero[First[qr]-1],
  "Euclidean remainder need not be positive"->zero[Last[qr]+2]|>;
 <|"checks"->checks,"passed"->And@@Values[checks]|>
];
End[];
EndPackage[];
