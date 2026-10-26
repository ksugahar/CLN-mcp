(* Round-wire modes from cylindrical Maxwell equations.
   See docs/round_wire_maxwell.md for normalization, sources and validity limits.
   x=r/a; a=mu=sigma=1; impedance is per unit axial length.
   maxIndex counts alternating element indices, not R/L pairs.
   The upper bound is operational and NOT a measured runtime guarantee. *)
BeginPackage["RoundWireMaxwell`"];
WireModes::usage = "WireModes[route,maxIndex,x,s] constructs normalized E-H, A, or T modes (indices 0..8).";
Begin["`Private`"];
WireModes[route_, maxIndex_, x_Symbol, s_Symbol] := Module[
  {ampere, faraday, energy, e=1, hprev=0, aprev=0, h, avec, tvec,
   resist, induct, rr={}, ll={}, ee={}, hh={}, aa={}, tt={}, z, j},
  If[!MemberQ[{"E-H","A","T"},route] || !IntegerQ[maxIndex] ||
     !TrueQ[0<=maxIndex<=8], Return[Failure["Input",<|"route"->route,"maxIndex"->maxIndex|>]]];
  ampere[f_] := Module[{u}, Cancel[Integrate[(f /. x->u) u,{u,0,x}]/x]];
  faraday[f_] := Module[{u}, Expand[Integrate[f /. x->u,{u,1,x}]]];
  energy[f_] := Integrate[Expand[2 Pi x f^2],{x,0,1}];
  Do[
    resist=Cancel[1/energy[e]];
    If[!TrueQ[resist>0], Return[Failure["Resistance",<|"index"->2 j|>]]];
    AppendTo[rr,resist]; AppendTo[ee,e];
    If[2 j+1>maxIndex,Break[]];
    Switch[route,
      "E-H", h=Expand[ampere[resist e]+hprev]; avec=-faraday[h]; tvec=ampere[e],
      "A", avec=Expand[-faraday[ampere[resist e]]+aprev];
           h=Expand[-D[avec,x]]; tvec=ampere[e],
      "T", tvec=ampere[e]; h=Expand[resist tvec+hprev]; avec=-faraday[h]
    ];
    induct=Cancel[energy[h]];
    If[!TrueQ[induct>0], Return[Failure["Inductance",<|"index"->2 j+1|>]]];
    AppendTo[ll,induct]; AppendTo[hh,h]; AppendTo[aa,avec]; AppendTo[tt,tvec];
    e=Expand[If[route=="A",e-avec/induct,e+faraday[h/induct]]];
    hprev=h; aprev=avec,
    {j,0,Floor[maxIndex/2]}];
  (* An odd terminal index ends in s L; an even terminal index ends in R. *)
  If[EvenQ[maxIndex], z=Last[rr];
    Do[z=Together[rr[[j]]+1/(1/(s ll[[j]])+1/z)],{j,Length[ll],1,-1}],
    z=Together[Last[rr]+s Last[ll]];
    Do[z=Together[rr[[j]]+1/(1/(s ll[[j]])+1/z)],{j,Length[ll]-1,1,-1}]];
  <|"route"->route,"maxIndex"->maxIndex,"R"->rr,"L"->ll,
    "E"->ee,"H"->hh,"A"->aa,"T"->tt,"Z"->z|>
];
End[];
EndPackage[];
