# CLN-mcp

**CLNは、電磁場を少数の抵抗・インダクタンスからなる回路で表現する方法です。** 回路の各段を場のモードと結びつけ、損失や磁気エネルギーの積分から素子値を導きます。

初めて読む方は **[CLNとは何か](docs/00_what_is_cln.ipynb)** から。表皮効果、はしご回路、エネルギー、段数と精度を図と実行例で説明しています。[全ノートの読み順](docs/README.md)も用意しました。

Reproducible Cauer ladder network derivations and finite-element validation notebooks.

| Notebook | Content / verified scope |
|---|---|
| [CLNとは何か](docs/00_what_is_cln.ipynb) | Introductory field/circuit diagrams, skin effect, energy integrals and truncation error |
| [展開点とエネルギー](docs/01_expansion_and_energy.ipynb) | Expansion point, Type1/Type2, normalization and recorded FEM convergence |
| [Type2: A and T give the same circuit](docs/type2_same_circuit.ipynb) | Circular wire, positive real shift, R0 / L1 / R2 / L3; symbolic boundary and normalization proof; energy vs Cauer |
| [Type1: derivation and NGSolve reproduction](docs/type1_same_circuit.ipynb) | Circular wire, positive real shift, L-1 / R0 / L1 / R2; symbolic proof, A/T finite elements, mesh convergence and radia-ngsolve runtime reproduction |

## 定式化と拡張を学ぶ

- [A–φ・T–Ω・A–T、3次元、ゲージと境界条件](docs/02_formulations_gauge_boundary.ipynb)：未知量と物理場を分け、ゲージ変換で場が不変なことを実行確認します。[3次元の丸線再現](docs/06_3d_round_wire.ipynb)では谷本氏の学生ノートをもとに、3定式化の回路定数・物理場・ゲージ処理を比較します。
- [非線形FP-CLNと表面モード](docs/03_nonlinear_fp_surface_modes.ipynb)：一定の線形核と更新する非線形源、表面縮約の役割を説明します。小さなモード系で反復収束とモデル精度を区別します。
- [終端問題・CLN1/CLN2変換・タイル表現](docs/04_termination_conversion_tiles.ipynb)：終端逆算の恒等式、型の命名と回路式の対応、電圧・電流のタイルを扱います。
- [マルチポート化](docs/05_multiport.ipynb)：相互結合、ポート座標変換、仕事保存、相反性と散逸を2ポートの実行例で説明します。

これらは実行済みの解説ノートです。説明用の最小例と、実際の3次元・非線形CLNの定量検証を混同しないよう、検証範囲を各ノートに記載しています。

## 3次元の実行結果

[3次元A–φ・T–Ω・A–T](docs/06_3d_round_wire.ipynb)と[再計算用コード](validation/validate_3d.py)を追加しました。円柱導体内部、線形材料、s₀=0のR₀/L₁/R₂を3定式化×2メッシュで検証します。回路定数と電界・磁束密度の誤差を別々に示し、端子値への規格化、境界条件、右辺射影、ゲージ項の感度も説明します。Codexの自己レビュー済みで、追加部分のClaudeレビューは復帰後に行います。

## 表面モードとFosterの比較実験

[実行済みの比較ノート](docs/07_surface_hybrid_foster.ipynb)では、解析円形断面、切欠き付き3次元導体、非線形2次元導体を扱います。Fosterの評価帯域とDC条件を別々に変更し、同じ3次元離散系に対してCLNと比較しました。高周波向けの帯域調整が有効でしたが、DC条件を外す追加効果は小さい結果です。

非線形例では、9本の独立CLN基底に残る誤差を、表面ポートを含む55本の基底で低減しました。候補数と独立本数、磁束とジュール損失を分けて示します。3次元のメッシュ間差は約3.5%あり、縮約誤差が小さいことだけで連続体の精度は保証できません。追加結果はCodex自己レビュー済みで、Claudeの独立レビューは未実施です。

[同じ実励振によるCLN・境界応答補足・POD・Foster](docs/08_3d_surface_pod_foster.ipynb)に比較を組み直しました。CLNもPODも同じ実励振から基底を作ります。違いは展開点・周波数情報・生成手順です。複数展開点CLNと、CLN2＋実励振の境界応答補足を同じ総状態数で比較します。境界応答のPODと空気領域のSteklovモード、Foster端子等価回路と場に対応する物理的CLN回路も区別します。旧仮想境界ポートの結果から、CLN＋表面モード全般への結論は出しません。

保存結果だけの点検は `python validation/surface_hybrid/validate_evidence.py`。FEM再計算の手順はノート末尾にあります。

## Reproduce

Install `ngsolve`, `numpy`, `matplotlib`, `mpmath`, `scipy`, `sympy`, `nbformat`, `nbclient` and `ipykernel` for Python; symbolic checks require `wolframscript` on PATH (or `WOLFRAMSCRIPT`). Open the executed notebooks under `docs/` and rerun their cells.

Standalone commands:

```text
wolframscript -file mathematica/derive_type1_same_circuit.wls type1_results.json
python validation/type1_ngsolve.py --output type1_fe.json
python validation/validate_3d.py --output three_dimensional.json
```

- `docs/*.ipynb`: primary derivations and executed reproduction entry points.
- `mathematica/*.wls`: symbolic and high-precision checks.
- `validation/`: public finite-element reproduction code.
- `docs/data/`: recorded second-runtime numerical evidence.

The public repository currently contains this focused documentation release. The broader MCP application is not yet included. These four-element continuum results do not prove arbitrary stage counts or general 3D mixed formulations. A finite shifted ladder matches locally near its expansion point, not at all frequencies. Positive elements in a shifted representation alone do not prove passivity in physical s.

Type2's underlying derivation received an independent Claude Opus 5.5 review. Type1 was verified by Codex using Wolfram, independent Bessel evaluation and two FE formulations; no new Opus review is claimed.

BSD-3-Clause license.
