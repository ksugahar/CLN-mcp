(* Type1 continuum wire, real z>0. Units: R=1/(sigma a^2), L=mu.
   A and T correction solves are written with the same impressed electric field.
   Four elements: L[-1],R[0],L[1],R[2]. No general 3D claim. *)
ClearAll["Global`*"];
$Assumptions=z>0&&0<r<1;
simp[x_]:=FullSimplify[FunctionExpand[x],Assumptions->$Assumptions];
div[f_]:=D[f,r]+f/r;
e=BesselI[0,Sqrt[z] r]/BesselI[0,Sqrt[z]];
h=BesselI[1,Sqrt[z] r]/(Sqrt[z] BesselI[0,Sqrt[z]]);
y=2 Pi (h/.r->1);
w=-D[y,z]; p=y-z w;
rm=1/p; lm=1/(z y);
(* After unit-current normalisation: E[-1]=e/y, H[-1]=h/y.
   Even stage E0=E[-1]/(z L[-1])=e. *)
e1=rm z D[e,z]; h1=rm (h+z D[h,z]);
(* A correction u solves -lap u+z u=rm e, u(1)=0.
   E1=-z u, H1=-u'. T correction t solves
   (div t-rm e,div v)+z(t,v)=0, natural E1(1)=0.
   E1=div t-rm e, H1=t. *)
ua=-e1/z;
aResidual=-div[D[ua,r]]+z ua-rm e;
tResidual=-D[div[h1]-rm e,r]+z h1;
aWeak=r*(D[u[r],r] D[n[r],r]+z u[r] n[r]-f[r] n[r]);
tWeak=r*((div[t[r]]-f[r])*div[v[r]]+z t[r] v[r]);
checks=<|
 "initial_Ampere"->(div[h]-e),
 "initial_Faraday"->(D[e,r]-z h),
 "initial_power_identity"->(D[r e h,r]-r(e^2+z h^2)),
 "E0_normalization"->((e/y)/(z lm)-e),
 "A_correction_equation"->aResidual,
 "A_reconstructed_H1"->(-D[ua,r]-h1),
 "T_correction_equation"->tResidual,
 "T_reconstructed_E1"->(div[h1]-rm e-e1),
 "common_Faraday"->(D[e1,r]-z h1),
 "A_essential_u_zero"->(ua/.r->1),
 "T_natural_E_zero"->(e1/.r->1),
 "A_boundary_identity"->(aWeak-r*(-div[D[u[r],r]]+z u[r]-f[r])*n[r]-D[r D[u[r],r] n[r],r]),
 "T_boundary_identity"->(tWeak-r*(-D[div[t[r]]-f[r],r]+z t[r])*v[r]-D[r*(div[t[r]]-f[r])*v[r],r])
|>;
res=Map[TrueQ[simp[#]==0]&,checks];
KeyValueMap[Print[If[#2,"PASS ","FAIL "],#1]&,res];
wp=90;
rows=Table[
 sv=N[shift,wp];
 ev=N[e/.z->sv,wp]; hv=N[h/.z->sv,wp];
 ev1=N[e1/.z->sv,wp]; hv1=N[h1/.z->sv,wp];
 integ[f_]:=NIntegrate[Evaluate[2 Pi r f],{r,0,1},WorkingPrecision->60,AccuracyGoal->40,PrecisionGoal->35];
 yy=integ[ev^2+sv hv^2];
 lmnum=integ[(ev/yy)^2+sv(hv/yy)^2]/sv;
 r0num=1/integ[ev^2];
 l1num=integ[ev1^2+sv hv1^2]/sv;
 ev2=ev+ev1/(sv l1num); r2num=1/integ[ev2^2];
 (* Independent continued fraction of G=z/Z=zY. Invert extracted values. *)
 jet=Sum[N[D[z y,{z,k}]/.z->sv,wp] q^k/k!,{k,0,3}];
 g0=Coefficient[jet,q,0]; g1=Coefficient[jet,q,1];
 tail=Normal[Series[1/(1/(jet-g0)-1/(q g1)),{q,0,1}]];
 g2=Coefficient[tail,q,0]; g3=Coefficient[tail,q,1];
 field={lmnum,r0num,l1num,r2num}; port={1/g0,1/g1,1/g2,1/g3};
 err=Max[Abs[(field-port)/port]];
 ok=TrueQ[err<10^-28&&Min[field]>0];
 Print[If[ok,"PASS ","FAIL "],"Type1_energy_vs_Cauer z=",shift," relative_error=",N[err,6]];
 <|"z"->ToString[shift,InputForm],"energyElements"->(ToString[#,InputForm]&/@field),"portElements"->(ToString[#,InputForm]&/@port),"relativeError"->ToString[err,InputForm],"passed"->ok|>,
 {shift,{1/10,1,8 Pi^2,100}}];
exportPath=If[Length[$ScriptCommandLine]>1,Last[$ScriptCommandLine],FileNameJoin[{Directory[],"type1_results.json"}]];
exportResult=Export[exportPath,<|"symbolicChecks"->res,"cases"->rows,"scope"->"Type1 continuum circular wire Lminus1 R0 L1 R2, real positive shift"|>,"RawJSON"];
Print["RESULT_FILE ",exportResult];
If[TrueQ[FileExistsQ[exportPath]]&&And@@Join[Values[res],Lookup[rows,"passed"]],Exit[0],Exit[1]];
