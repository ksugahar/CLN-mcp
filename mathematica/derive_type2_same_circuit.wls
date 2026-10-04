(* Codex: exact continuum derivation for the latest notebooks ncln=3.
 Dimensionless radius r/a; z=mu sigma a^2 s. Resistance unit 1/(sigma a^2),
 inductance unit mu. Real z>0, homogeneous conductor, axisymmetric fields. *)
ClearAll["Global`*"];
$Assumptions=z>0&&0<r<1;
e=BesselI[0,Sqrt[z] r]/BesselI[0,Sqrt[z]];
h=BesselI[1,Sqrt[z] r]/(Sqrt[z] BesselI[0,Sqrt[z]]);
y=2 Pi (h/.r->1); zz=1/y;
w=Pi/z (BesselI[1,Sqrt[z]]^2-BesselI[0,Sqrt[z]] BesselI[2,Sqrt[z]])/BesselI[0,Sqrt[z]]^2;
simp[x_]:=FullSimplify[FunctionExpand[x],Assumptions->$Assumptions];
c=y/w;
e2=e+c D[e,z]; h2=h+c D[h,z];
checks=<|
 "initial_Ampere"->(D[h,r]+h/r-e),
 "initial_Faraday"->(D[e,r]-z h),
 "initial_power_boundary_identity"->(D[r e h,r]-r(e^2+z h^2)),
 "A_reconstructed_H2"->(-D[(e-e2)/z,r]+h-c h/z-h2),
 "Bessel_magnetic_energy_primitive"->
 (D[r^2/(2 z BesselI[0,Sqrt[z]]^2)*(BesselI[1,Sqrt[z] r]^2-BesselI[0,Sqrt[z] r] BesselI[2,Sqrt[z] r]),r]-r h^2),
 "admittance_derivative_minus_W"->(D[y,z]+w),
 "unit_current_normalized_E_equals_unit_voltage_E"->((e/y)/(1/y)-e),
 "unit_current_normalized_H_equals_unit_voltage_H"->((h/y)/(1/y)-h),
 "stage2_Ampere"->(D[h2,r]+h2/r-e2),
 "stage2_Faraday_forcing"->(D[e2,r]-z h2-c h),
 "stage2_A_Dirichlet_E_is_one"->((e2/.r->1)-1),
 "stage2_T_Dirichlet_H_is_zero"->(h2/.r->1)
|>;
res=Map[TrueQ[simp[#]==0]&,checks];
KeyValueMap[Print[If[#2,"PASS ","FAIL "],#1]&,res];
(* Initial power identity: integral(e^2+z h^2) = [2Pi r e h]_0^1=y.
 Unit-current fields are (e,h)/y; their power is 1/y=Z.
 T SolveInitial divides by this power, recovering exactly (e,h).
 Stage2 fields above satisfy BOTH notebook Dirichlet conditions.
 Positivity/uniqueness in verify_type2_stage_boundary.wls completes equality.
*)
wp=90;
rows=Table[
 sv=N[shift,wp];
 ev=N[e/.z->sv,wp]; hv=N[h/.z->sv,wp];
 ev2=N[e2/.z->sv,wp]; hv2=N[h2/.z->sv,wp];
 integ[f_]:=NIntegrate[Evaluate[2 Pi r f],{r,0,1},WorkingPrecision->60,AccuracyGoal->40,PrecisionGoal->35];
 r0=1/integ[ev^2+sv hv^2]; hodd1=r0 hv; l1=integ[hodd1^2];
 r2=1/integ[ev2^2+sv hv2^2]; hodd3=r2 hv2+hodd1; l3=integ[hodd3^2];
 (* Independent continued-fraction extraction from exact port Taylor jet. *)
 jet=Sum[N[D[zz,{z,k}]/.z->sv,wp] q^k/k!,{k,0,3}];
 rr0=Coefficient[jet,q,0]; ll1=Coefficient[jet,q,1];
 tail=Normal[Series[1/(1/(jet-rr0)-1/(q ll1)),{q,0,1}]];
 rr2=Coefficient[tail,q,0]; ll3=Coefficient[tail,q,1];
 field={r0,l1,r2,l3}; port={rr0,ll1,rr2,ll3};
 err=Max[Abs[(field-port)/port]];
 bc=Max[Abs[{(ev2/.r->1)-1,hv2/.r->1}]];
 ok=TrueQ[err<10^-28&&bc<10^-28&&Min[field]>0];
 Print[If[ok,"PASS ","FAIL "],"energy_vs_Cauer z=",shift," error=",N[err,5]," BC=",N[bc,5]];
 <|"z"->ToString[shift,InputForm],"energyElements"->(ToString[#,InputForm]&/@field),"portElements"->(ToString[#,InputForm]&/@port),"relativeError"->ToString[err,InputForm],"passed"->ok|>,
 {shift,{1/10,1,8 Pi^2,100}}];
exportPath=If[Length[$ScriptCommandLine]>1,Last[$ScriptCommandLine],FileNameJoin[{Directory[],"type2_same_circuit_results.json"}]];
exportResult=Export[exportPath,<|"symbolicChecks"->res,"cases"->rows,"scope"->"continuum round wire, real positive shift, Type2 R0 L1 R2 L3; no FEM execution"|>,"RawJSON"];
Print["RESULT_FILE ",exportResult];
If[TrueQ[FileExistsQ[exportPath]] && And@@Join[Values[res],Lookup[rows,"passed"]],Exit[0],Exit[1]];



