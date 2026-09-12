# Fondations fichiers et PNG — CI du 12 septembre 2026

Le [run 34677382144](https://github.com/skyline624/ComfySharp/actions/runs/34677382144), au commit [`10b8a1d`](https://github.com/skyline624/ComfySharp/commit/10b8a1d22035b1b49514731ade806c0a2e9c621b), est terminé en échec. Ses 47 rapports TRX téléchargés distinguent les composants fichiers/PNG réussis des gates numériques encore en échec.

| Contrôle | Windows x64 | Linux x64 | macOS arm64 |
|---|---:|---:|---:|
| Core / Workflow / Tokenization / Desktop / Host | 1 748 PASS | 1 748 PASS | 1 748 PASS |
| Dont nommage des fichiers | 25 PASS | 25 PASS | 25 PASS |
| Dont stockage, liens de fichiers et liens de dossiers | 17 PASS | 17 PASS | 17 PASS |
| Commande IMAGE dédiée, dont 19 tests PNG | 126 PASS | 126 PASS | 126 PASS |
| Premier accès natif, processus séparés | 176 PASS / 18 FAIL | 182 PASS / 12 FAIL | 194 PASS |
| Suite Inference ordinaire hors CLIP complet | Non exécutée | Non exécutée | 700 PASS |
| CLIP dimensions complètes, processus séparé | Non exécuté | Non exécuté | 1 PASS / 2 FAIL |

Les commandes séparées exécutent certains mêmes tests ; leurs résultats ne doivent pas être additionnés comme des tests distincts. Les 19 tests PNG présents dans la commande IMAGE sont également présents dans la suite Inference ordinaire macOS.

Le test de liens symboliques de fichiers, exclu de la campagne Windows locale faute de privilège, passe dans la CI des trois systèmes. La correction `10b8a1d` utilise `File.Delete` pour retirer les liens de dossiers Unix, y compris les liens devenus pendants ; la précédente campagne `3ed063e` échouait dans ce nettoyage de fixture Linux/macOS. Le test complet de stockage concerné passe désormais sur les trois OS.

Les échecs numériques Windows concernent cinq cas U-Net, six CFG, cinq Euler et deux pipeline. Linux échoue sur un U-Net, quatre CFG, cinq Euler et deux pipeline. Les deux échecs macOS concernent CLIP L/G aux dimensions complètes. Ce relevé ne démontre pas leur cause et ne modifie ni référence ni tolérance. Les suites postérieures non exécutées sous Windows/Linux ne sont pas déclarées réussies.

Cette révision ne contient pas encore les nœuds SaveImage/PreviewImage ni `/view`. Elle fournit une preuve des fondations sur les trois OS, sans qualification GPU, poids préentraînés ou interface bitmap. Le raccordement des nœuds fait l'objet d'une [campagne distincte](../IMAGE_FILE_NODES.md).
