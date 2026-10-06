(* Independent round-wire, zero-shift series-R/shunt-L extraction.
   Normalized Z/Rb = sqrt(u) I0(sqrt(u))/(2 I1(sqrt(u))).
   This derives seven terminal elements, not 3D physical field modes.
   Modified-Bessel fraction: https://dlmf.nist.gov/10.33.E1 *)
ClearAll["Global`*"];
i0=Sum[u^k/(4^k (k!)^2),{k,0,8}];
i1OverRoot=Sum[u^k/(2 4^k k! (k+1)!),{k,0,8}];
z=Together[i0/(2 i1OverRoot)];
rs={};ls={};
Do[
 r=Limit[z,u->0];AppendTo[rs,r];
 If[j<3,
  rem=Together[z-r];l=Limit[rem/u,u->0];AppendTo[ls,l];
  z=Together[1/(1/rem-1/(u l))];
 ],{j,0,3}];
checks=<|"resistors"->(rs=={1,3,5,7}),"inductors"->(ls=={1/8,1/16,1/24}),
 "I0_power_series"->(i0==Normal[Series[BesselI[0,Sqrt[u]],{u,0,8}]]),
 "I1_power_series"->(i1OverRoot==Normal[Series[BesselI[1,Sqrt[u]]/Sqrt[u],{u,0,8}]])|>;
result=<|"scope"->"Seven zero-shift round-wire terminal elements, independently extracted exact Bessel coefficients",
 "R_normalized"->(ToString[#,InputForm]&/@rs),"L_normalized"->(ToString[#,InputForm]&/@ls),"checks"->checks|>;
output=If[Length[$ScriptCommandLine]>1,Last[$ScriptCommandLine],"round_wire_three_stages.json"];
Export[output,result,"RawJSON"];
Print[result];
If[TrueQ[FileExistsQ[output]] && And@@Values[checks],Exit[0],Exit[1]];

