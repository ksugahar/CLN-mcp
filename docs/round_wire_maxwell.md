# Round-wire Maxwell modes in Mathematica

`src/cln_mcp_server/resources/RoundWireMaxwell.wl` constructs E–H, A and T
representations of an unshifted, homogeneous circular conductor. The three
routes use cylindrical Maxwell integrations and return exact polynomial modes,
ladder coefficients and a finite rational impedance. Mathematica is the
maintained symbolic implementation; no additional SymPy implementation is
required to run it.

The radius coordinate is `x=r/a`, with `a=mu=sigma=1` inside the symbolic
calculation. Resistance and inductance are per unit axial length. To recover
physical coefficients, multiply the returned normalized resistance by
`1/(sigma*a^2)` and inductance by `mu`. The Laplace variable inside the returned
impedance is `s_physical*mu*sigma*a^2`; multiply impedance by `1/(sigma*a^2)`.
Thus `Rdc=1/(Pi*sigma*a^2)` and `Ldc=mu/(8*Pi)`.

`WireModes["E-H",8,x,s]` includes alternating indices 0 through 8: five
resistances and four inductances. This is not eight R/L pairs. The index bound
is operational; a timing at this bound does not establish a general runtime
guarantee or measure the separate MCP derivation tool's stage cap.

Ampère integration uses `H(x)=Integrate[t*J(t),{t,0,x}]/x`, selecting the regular
axis solution. Adding an arbitrary constant to impose a magnetic boundary
value violates the cylindrical differential equation; its homogeneous term is
`C/x`, which regularity excludes. Faraday corrections integrate from `x=1`,
giving homogeneous correction traces. The accumulated electric mode keeps its
initial unit boundary trace. Energy norms use the cross-section measure
`2*Pi*x dx`. Exact zero energy denotes termination; the implementation returns
a failure rather than manufacturing a positive element.

The separate analytic route is the regular modified-Bessel solution:

`Z/Rdc = Sqrt[z]*BesselI[0,Sqrt[z]]/(2*BesselI[1,Sqrt[z]])`,
where `z=s_physical*mu*sigma*a^2`. The validation script compares series through
degree 8, field maps, traces, energy orthogonality and all three formulations.
The coefficient sequences are `R_even/Rdc={1,3,5,7,9}` and
`L_odd/Ldc={1,1/2,1/3,1/4}`. A finite ladder is not the exact impedance over an
arbitrary frequency band. These checks do not qualify general 3D, shifted,
heterogeneous or finite-element implementations.

Run the prepared check with an existing output directory:

```text
wolframscript -file validation/round_wire_maxwell.wls output.json
```

The JSON records individual assertions, rational expressions, kernel identity,
package hash and route timings. Timing is diagnostic and never an accuracy
criterion. No execution result is claimed by this source document.

This implementation is authored from the equations; third-party notebooks and
their presentation code are not included. Background: Kameari et al.,
[IEEE Transactions on Magnetics (2018)](https://doi.org/10.1109/TMAG.2017.2743224).
For the independently evaluated analytic function see
[NIST DLMF §10.25](https://dlmf.nist.gov/10.25.E2).
Hiruma et al.'s
[unit-cell homogenization paper (2024)](https://doi.org/10.1109/TMAG.2024.3486683)
motivates related port and normalization questions; this circular-wire check
does not reproduce or validate that paper's complete unit-cell model.
