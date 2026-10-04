# CLNを理解するためのノートブック

| 順番 | 読むもの | 分かること | 再実行に必要な環境 |
|---|---|---|---|
| 1 | [CLNとは何か](00_what_is_cln.ipynb) | 表皮効果、場と回路、エネルギー、段数と誤差 | Python + numpy + matplotlib + mpmath |
| 2 | [展開点とエネルギー](01_expansion_and_energy.ipynb) | Typeの違い、展開点、規格化、受動性の注意、保存済みFEM収束結果 | 同上 |
| 3 | [定式化・3次元・ゲージ・境界条件](02_formulations_gauge_boundary.ipynb) | A–φ/T–Ω/A–Tの復元式、ゲージ恒等式、谷本氏の3次元資料の位置づけ | Python + sympy |
| 4 | [非線形FP-CLNと表面モード](03_nonlinear_fp_surface_modes.ipynb) | 固定点補正、表面の近似、反復残差とモデル誤差の違い | Python + numpy + matplotlib |
| 5 | [終端・型変換・タイル](04_termination_conversion_tiles.ipynb) | 終端の逆算、厳密変換と近似、電圧・電流の可視化 | Python + sympy + matplotlib |
| 6 | [マルチポート](05_multiport.ipynb) | 結合行列、仕事保存、相反性・散逸の検査 | Python + numpy |
| 7 | [Type2の導出](type2_same_circuit.ipynb) | A/Tの境界条件と場の復元、同じ4要素になる証明 | Python + Wolfram |
| 8 | [Type1の導出とFEM再現](type1_same_circuit.ipynb) | 同じ4要素になる証明、別々のA/T有限要素解、再現結果 | Python + Wolfram + NGSolve |

| 9 | [3次元丸線の再現](06_3d_round_wire.ipynb) | A–φ/T–Ω/A–Tの初段、物理場と回路定数の収束、ゲージ感度と端子規格化 | 図表の再検査: Python + numpy + matplotlib、FEM再計算: NGSolve + scipy + radia |

全ノートは実行済みです。グラフと数値はノート内に保存されているため、GitHubで閲覧できます。入門2本は解析解を使い、FEMやWolframなしで再実行できます。

この公開版は導出と再現資料を中心に構成しています。MCP本体の配布ではありません。解析的な連分数の実演と、場から同じ回路を導く証明の範囲は区別して記載しています。

## 実行例の意味

追加した4本は、定式化と拡張の意味を理解するための実行済み解説です。ゲージ恒等式、説明用非線形モード系、最小の終端・等価変換、2ポートRL系を扱います。一般3次元FEM・実際のFP-CLN・一般のCLN型変換器・多導体行列CLNをすべて公開実装したという意味ではありません。研究データの再現と解説用の計算を、各ノートで区別しています。

3次元丸線ノートは実際のFEM再現です。ただし線形・s₀=0・導体内部・R₀/L₁/R₂に限定します。追加部分はCodex自己レビュー済みで、Claudeによる独立レビューは未実施です。
