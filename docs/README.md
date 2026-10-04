# CLNを理解するためのノートブック

| 順番 | 読むもの | 分かること | 再実行に必要な環境 |
|---|---|---|---|
| 1 | [CLNとは何か](00_what_is_cln.ipynb) | 表皮効果、場と回路、エネルギー、段数と誤差 | Python + numpy + matplotlib + mpmath |
| 2 | [展開点とエネルギー](01_expansion_and_energy.ipynb) | Typeの違い、展開点、規格化、受動性の注意、保存済みFEM収束結果 | 同上 |
| 3 | [Type2の導出](type2_same_circuit.ipynb) | A/Tの境界条件と場の復元、同じ4要素になる証明 | Python + Wolfram |
| 4 | [Type1の導出とFEM再現](type1_same_circuit.ipynb) | 同じ4要素になる証明、別々のA/T有限要素解、再現結果 | Python + Wolfram + NGSolve |

全ノートは実行済みです。グラフと数値はノート内に保存されているため、GitHubで閲覧できます。入門2本は解析解を使い、FEMやWolframなしで再実行できます。

この公開版は導出と再現資料を中心に構成しています。MCP本体の配布ではありません。解析的な連分数の実演と、場から同じ回路を導く証明の範囲は区別して記載しています。
