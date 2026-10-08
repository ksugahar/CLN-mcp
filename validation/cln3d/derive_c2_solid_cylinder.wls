ClearAll["Global`*"];
kap = Sqrt[(Pi/h)^2 + mu sigma s];
az = mu j0/kap^2 (r - a BesselI[1, kap r]/BesselI[1, kap a]) Sin[Pi z/h];
assumptions = a > 0 && h > 0 && mu > 0 && sigma > 0 && s > 0 && 0 < r < a;
pde = FullSimplify[-(D[az,{r,2}]+D[az,r]/r-az/r^2+D[az,{z,2}])/mu+sigma s az-j0 r Sin[Pi z/h], assumptions];
If[pde =!= 0, Print["FAIL PDE: ",pde]; Exit[1]];
If[FullSimplify[az /. r -> a, assumptions] =!= 0, Print["FAIL radial PEC"]; Exit[1]];
If[FullSimplify[az /. z -> 0, assumptions] =!= 0 || FullSimplify[az /. z -> h, assumptions] =!= 0, Print["FAIL end PEC"]; Exit[1]];
primitive = FullSimplify[D[r^2 BesselI[2, q r]/q,r]-r^2 BesselI[1,q r],r>0&&q>0];
If[primitive =!= 0, Print["FAIL primitive: ",primitive]; Exit[1]];
flux = Pi mu j0^2 h/kap^2 (a^4/4-a^3 BesselI[2,kap a]/(kap BesselI[1,kap a]));
(* Source normalization integral Jtheta dr dz = j0 a^2 h/Pi = 1 A. *)
values = Table[With[{sv=2 Pi I hz},N[s flux /. {s->sv,mu->4 Pi 10^-7,sigma->10^6,a->25/1000,h->20/1000,j0->Pi/((25/1000)^2 (20/1000))},30]],{hz,{1000,10000,100000,1000000}}];
Print["PASS solid-cylinder azimuthal-source PDE, PEC conditions and Bessel primitive"];
Print["Z values: ",InputForm[values]];
Exit[0];
