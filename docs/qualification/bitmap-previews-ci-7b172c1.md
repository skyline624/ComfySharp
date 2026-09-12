# Aperçus bitmap — CI au commit 7b172c1

Le [run 34679543429](https://github.com/skyline624/ComfySharp/actions/runs/34679543429), au commit [`7b172c1`](https://github.com/skyline624/ComfySharp/commit/7b172c122f42723683c5b05dd15a1361b444c0c7), est terminé. Windows réussit ; Linux et macOS échouent encore aux gates numériques. L'audit porte sur 49 rapports TRX.

| Contrôle | Windows x64 | Linux x64 | macOS arm64 |
|---|---:|---:|---:|
| Application : Core, Workflow, Tokenization, Desktop, Host | 1 800 PASS | 1 800 PASS | 1 800 PASS |
| Dont Desktop, avec 26 nouveaux cas bitmap/transport | 71 PASS | 71 PASS | 71 PASS |
| IMAGE dédié | 134 PASS | 134 PASS | 134 PASS |
| Références au premier accès natif | 194 PASS | 182 PASS / 12 FAIL | 194 PASS |
| Inference ordinaire hors CLIP complet | 708 PASS | Non exécutée | 708 PASS |
| CLIP complet séparé | 3 PASS | Non exécuté | 1 PASS / 2 FAIL |

Le smoke avec vraie fenêtre, PNG, navigation dans les lots, métadonnées du workflow et redémarrage du Host réussit sur les trois OS. Les 26 nouveaux tests réussissent aussi sur chaque OS, avec le backend raster Skia réel sous Avalonia Headless. Les commandes dédiées répètent certains tests ; leurs nombres ne s'additionnent pas comme des cas distincts.

Windows réunit 2 511 tests distincts réussis dans les suites ordinaires et CLIP séparée. macOS réunit 2 509 réussites et deux échecs CLIP L/G. Sous Linux, les douze échecs concernent un U-Net, quatre CFG, cinq Euler et deux pipeline ; les suites suivantes non exécutées restent non vérifiées. Une campagne Windows réussie ne résout pas les variations numériques observées dans les autres campagnes ou plateformes.

Cette preuve couvre les aperçus PNG des primitives, sans génération préentraînée, GPU ni qualification des distributions. L'import du workflow depuis le PNG a été ajouté après cette révision et dispose d'une campagne distincte.
