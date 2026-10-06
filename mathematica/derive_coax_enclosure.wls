ClearAll["Global`*"];
a=1/200;b=3/200;h=3/200;sigma=10^6;mu=4 Pi 10^-7;tau=mu sigma (1/100)^2;
rdc=h/(sigma Pi a^2);lint=mu h/(8 Pi);lair=mu h Log[b/a]/(2 Pi);
zfun[s_]:=rdc Hypergeometric0F1[1,mu sigma a^2 s/4]/Hypergeometric0F1[2,mu sigma a^2 s/4]+s lair;
checks=<|"DCResistance"->TrueQ[zfun[0]==rdc],"DCInductance"->TrueQ[FullSimplify[D[zfun[s],s]/.s->0]==lint+lair],"AirInductance"->TrueQ[FullSimplify[Integrate[mu h/(2 Pi r),{r,a,b}]]==lair],"NoAirLimit"->TrueQ[Limit[mu h Log[bb/a]/(2 Pi),bb->a,Direction->"FromAbove"]==0]|>;
rows=Table[
 jet=N[Normal[Series[zfun[shift+q/tau],{q,0,8}]],80];co=CoefficientList[jet,q];rs={};ls={};
 Do[
  rr=First[co];AppendTo[rs,rr];co[[1]]=0;
  degree=Length[co]-2;
  inv=CoefficientList[Normal[Series[1/(Sum[co[[j+2]] q^j,{j,0,degree}]),{q,0,degree}]],q];
  AppendTo[ls,tau/First[inv]];
  If[Length[inv]>1,co=CoefficientList[Normal[Series[1/Sum[inv[[j+2]] q^j,{j,0,Length[inv]-2}],{q,0,Length[inv]-2}]],q]],
 {4}];
 <|"s0"->N[shift],"Rhat"->N[rs],"L"->N[ls],"passed"->TrueQ[Min[Join[rs,ls]]>0]|>,
 {shift,{0,2 Pi 10000}}];
result=<|"sourceSHA256"->IntegerString[FileHash[$InputFileName,"SHA256"],16,64],"checks"->checks,"cases"->rows,"scope"->"Exact MQS coax, perfectly conducting enclosure and ideal source-break port; no capacitance or shell loss"|>;
If[Length[$ScriptCommandLine]>1,Export[Last[$ScriptCommandLine],result,"RawJSON"]];
Print[result];If[And@@Values[checks]&&And@@Lookup[rows,"passed"],Print["PASS coax oracle" ];Exit[0],Exit[1]];
