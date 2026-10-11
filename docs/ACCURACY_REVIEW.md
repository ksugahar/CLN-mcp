# Reading accuracy claims and formulation comparisons

This guide connects the AC-axis questions to existing reproducible evidence.
It does not infer polynomial order, normalization or termination from a plot.

The [matched formulation study](17_formulation_ranking.ipynb) now traces the
public baseline plot to its producer: order 1, zero shift, maximum error of
R0/L1/R2. It then varies order, coefficient window and real shift separately
on shared 3D meshes. A-T is better for the tested deeper-element and positive-
shift comparisons, while A-phi wins the baseline first-three-element metric.
Neither is a universal ranking of fields, terminal impedance or cost.

[High-order local recovery](18_element_curl_recovery.ipynb) separately tests the
element-by-element part of vector-potential reconstruction at p=2/3. The
global scalar corrections and full CLN integration are distinct gates.
[The integrated A-T study](19_element_cln_integration.ipynb) now verifies
trace/local recovery in the full zero-shift CLN and records the failed direct
one-pass extension on the actual curved mesh.

| Question | Evidence | What remains unresolved |
|---|---|---|
| Does a visually good real/imaginary response imply 1% accuracy? | [Expansion notebook](01_expansion_and_energy.ipynb), [AC producer](../validation/ac_axis_study.py), [recorded samples](data/ac_axis_study.json), [independent evaluator](../validation/validate_ac_axis.py) | Continuous crossing locations between samples; reproduction of a published figure with its precise original conditions |
| Does a higher expansion point improve the whole band? | The same AC study compares real shifts, six/seven-element Type2 terminations, Type1 and a separately labeled DC-preserving rational control | Universal bandwidth or monotonic convergence across element count; the rational control is not a field-derived circuit |
| Is A-T always more accurate than A-Phi or T-Omega? | [3D circular wire](06_3d_round_wire.ipynb), [general conductor](12_general_3d_cln.ipynb), [air cohomology](15_air_cohomology_cln.ipynb) | A universal ranking; unresolved high-frequency layers and deep-element gaps prevent that conclusion |
| Was the plotted calculation p=1? | Each producer and recorded cohort identifies its own spaces/order | The order of an unidentified screenshot; mixed spaces need the order of every unknown |

## Keep the error definitions attached to the plots

Complex impedance error is `abs(Zred-Zref)/abs(Zref)`. Complex admittance
error is `abs(Yred-Yref)/abs(Yref)`. They differ even when Y=1/Z:

    eY = eZ / abs(Zred/Zref).

The AC validator checks this reciprocal identity for Type2. A plot of Re Y
and Im Y, a magnitude plot, a phase plot and a complex relative error plot
answer different questions. Report component errors separately, and use an
absolute error near a zero of the reference component. State the phasor
convention: the stored round-wire admittance has negative imaginary part;
the log plot shows -Im Y.

For a field-generated ladder distinguish:

1. Reduction error: ladder versus its own same-mesh direct solution.
2. Discretization error: direct solution versus the analytic or refined reference.
3. Element error: each R/L value versus its corresponding oracle.

These relative errors have different denominators and cannot simply be added.
Agreement between two formulations or two expansion points is not accuracy
against an independent reference. Internal field agreement and terminal
impedance agreement must also be assessed separately.

## Expansion point, termination and sampled bandwidth

Here s0 is real and nonnegative, q=s-s0, and AC evaluation uses s=j*omega.
The label f0=s0/(2*pi) is a scale convention, not interpolation on the AC
axis. For the round wire, z=tau*s and tau=mu*sigma*a^2; report both physical
parameters and the normalized axis before comparing curves.

A higher real shift can improve high-frequency accuracy while sacrificing
low-frequency accuracy. Six and seven elements do not have the same terminal
branch. Fix ladder type, count and termination before comparing shifts.
Positive elements in q alone do not prove passivity in physical s.

`one_percent_intervals_hz` in the AC JSON contains connected passing intervals
on 601 samples between 10 Hz and 10 MHz. These are sampled endpoints, not
certified continuous bands or interpolated crossings. An interval touching
the sweep boundary is clipped. The producer independently checks continued
fractions against Taylor Pade approximants; it does not digitize a paper's
figure or establish equivalence of every circuit called a T method.

For the recorded Type2 wire (radius 5 mm, sigma=5.8e7 S/m, mu=mu0),
the sampled admittance bands illustrate the termination effect:

| f0 [Hz] | Six elements: <=1% interval [Hz] | Seven elements: <=1% interval [Hz] |
|---|---|---|
| 1,000 | 10–4,897.8 | 10–8,128.3 |
| 10,000 | 7,413.1–12,882.5 | 3,548.1–25,704.0 |
| 100,000 | No passing samples | 39,810.7–245,470.9 |

These values summarize the stored JSON, not the unidentified screenshot.

## Required conditions for a formulation comparison

Record geometry, material data, excitation and port normalization, boundary
conditions, expansion point, ladder type, element count, termination, FE
space/order of each unknown, curved geometry order, quadrature, mesh counts,
skin-depth resolution, backend and true residual. Compare the same physical
field component at the same points. Perform mesh and order refinement before
attributing a difference to the formulation itself.

The [review status](REVIEW_STATUS.md) distinguishes scientific acceptance from
continuum accuracy, publication and dependency migration. In particular,
same-mesh acceptance does not close the C2 loss-convergence or independent
full T-Omega checks. A formulation ranking must retain these limitations.

## Reproduce the existing AC evidence

With numpy and mpmath installed, from the repository root:

    python validation/ac_axis_study.py
    python validation/validate_ac_axis.py
    python tools/policy_lint.py

The first command regenerates the recorded JSON. The validator checks producer
hashes, ladder evaluation, reciprocal/component errors and sampled intervals,
and rejects a corrupted terminal element. Reading the saved notebook needs
no solver rerun. Native FE reproduction has separate dependencies documented
in the notebook reading guide.
