# Repository policy (CLN-mcp)

The tests enforce this policy: `tests/test_policy.py` runs `tools/policy_lint.py`.

| rule | content |
|---|---|
| P1 | Implementations are published **only** as Python (`.py`: the `cln` package, the MCP server, research scripts, tests), Mathematica (`.wls` / `.wl`: symbolic derivations and analytic oracles), or **executed** Jupyter notebooks (`.ipynb`, under `docs/` only: tutorials). |
| P2 | No files of other tool chains: MATLAB (`.m .mlx .slx .mat .mex*`), COMSOL (`.mph`), FEMM (`.fem .ans`), JMAG (`.jmdl .jproj .jfiles .jcf`), FreeFEM (`.edp .idp`), Julia (`.jl`). Knowledge of the lab archive is ported to Python, Mathematica or NGSolve instead. |
| P3 | Published text and code do not name commercial tools (COMSOL, FEMM, JMAG). Papers and analytic solutions are cited; what was validated against commercial tools is not stated. |
| P4 | No absolute local paths (drive letters, user directories) in published files. Standard install locations (`C:\Program Files\...`) are allowed. |
| P5 | Every code cell of a notebook in `docs/` has been executed. |
| P6 | Data files are JSON (preferred) or HDF5 (`.h5` / `.hdf5`) for large arrays. No `.mat`, `.dat`, `.npy`, `.npz` or `.pkl`. Reference data are recomputed by the repository's own scripts, not copied. |
| P7 | Nothing refers to the lab archive: no drive-letter tokens in code, no `<archive>` placeholder, no index of archive files. Every script is self-contained. |
| P8 | No private code or binaries: no EMPY names or imports, no compiled artefacts (`.pyd .dll .exe .so .dylib .obj .o .a`). Drive-letter paths are detected with either slash. |

The lint is a necessary gate, not proof of absence: text that names internal archive folders, private files or unpublished provenance violates P7 even when no rule matches, and is removed in review.

In Japanese (日本語):
- 実装の公開形式は `.py`、`.wls`・`.wl`、実行済みの `docs/*.ipynb` に限る。
- MATLAB・COMSOL・FEMM・JMAG・FreeFEM・Julia のファイルは置かない。
- 公開物には、商用ツールの名前と手元の絶対パスを書かない。
- データは JSON を優先し、大きな配列は HDF5 にする(.mat、.dat などは置かない)。参照データは、リポジトリのスクリプトで再計算して作る。
- 研究アーカイブ(元の MATLAB の結果ファイルなど)には依存しない。スクリプトは単体で動く。
- 非公開のコード(EMPY など)とバイナリ(.pyd、.dll、.exe など)は置かない。
- 配布パッケージ（wheel / sdist）は GitHub Release のみに置き、PyPI へは公開しない。開発時は checkout からインストールできる。
- 違反があればテストが失敗する。ただし lint は必要条件であり、内部フォルダ名や未公開の出所を書いた文章はレビューで取り除く。


## Distribution

CLN-mcp release packages are distributed only as wheel (`.whl`) and source
distribution (`.tar.gz`) assets in [GitHub Releases](https://github.com/ksugahar/CLN-mcp/releases).
CLN-mcp must never be published to PyPI. Development installations from a
repository checkout remain supported; release artifacts are not checked into
the repository.


The sole CUDA source exception is the existing `validation/cln3d/mexcuda/colored_projection.cu`; this permits no compiled MEX or other CUDA source.
