# Nœuds fichiers — CI au commit 8556ab2

Le [run 34678723960](https://github.com/skyline624/ComfySharp/actions/runs/34678723960), au commit [`8556ab2`](https://github.com/skyline624/ComfySharp/commit/8556ab2ed59ed64ca9df868bd3069695e690861b), est terminé en échec. L'audit de ses 47 rapports TRX confirme que les **34 nouveaux cas des nœuds fichiers et du cache d'objets passent sur les trois OS**.

| Contrôle | Windows x64 | Linux x64 | macOS arm64 |
|---|---:|---:|---:|
| Core / Workflow / Tokenization / Desktop / Host | 1 774 PASS | 1 774 PASS | 1 774 PASS |
| Dont nouveaux cas cache / workflow / template / Host | 26 PASS | 26 PASS | 26 PASS |
| IMAGE dédié, dont huit nouveaux cas SaveImage/PreviewImage | 134 PASS | 134 PASS | 134 PASS |
| Références au premier accès natif | 176 PASS / 18 FAIL | 178 PASS / 16 FAIL | 194 PASS |
| Inference ordinaire hors CLIP complet | Non exécutée | Non exécutée | 708 PASS |
| CLIP complet dans un processus séparé | Non exécuté | Non exécuté | 1 PASS / 2 FAIL |

Les exécutions dédiées et ordinaires se recoupent et ne doivent pas être additionnées comme des tests distincts. Le Host passe 300 tests sur chaque runner, y compris le test de liens symboliques de fichiers exclu du poste Windows local. Le contrôle de catalogue et le smoke Desktop/Host de cette révision passent aussi sur chaque OS ; ce smoke précède l'ajout des bitmaps.

Les 18 échecs Windows se répartissent en cinq U-Net, six CFG, cinq Euler et deux pipeline. Les 16 échecs Linux concernent deux U-Net, six CFG, sept Euler et un pipeline. macOS conserve deux échecs CLIP aux dimensions complètes. Aucun changement de référence, tolérance ou qualification GPU n'est déduit de cette campagne. Les suites non exécutées après une gate en échec restent non vérifiées.
