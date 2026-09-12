# CI des fondations PNG et métadonnées

Campagne [34676041618](https://github.com/skyline624/ComfySharp/actions/runs/34676041618), commit [`c262ed87b1e8a4e3db2666558df5ebd0330aa6c0`](https://github.com/skyline624/ComfySharp/commit/c262ed87b1e8a4e3db2666558df5ebd0330aa6c0), terminée le 12 septembre 2026 avec un résultat global **failure**. L'audit local a lu les 49 TRX de 11 artefacts téléchargés.

| Contrôle | Windows x64 | Ubuntu x64 | macOS arm64 |
|---|---:|---:|---:|
| Core, Desktop, Host, Tokenization, Workflow | 1 706 PASS | 1 706 PASS | 1 706 PASS |
| Nouveaux tests des métadonnées, inclus ci-dessus | 28 PASS | 28 PASS | 28 PASS |
| Suite IMAGE dédiée existante | 107 PASS | 107 PASS | 107 PASS |
| Références au premier accès natif | 194 PASS | 182 PASS, 12 FAIL | 194 PASS |
| Inference ordinaire hors CLIP stock | 700 PASS | Non exécutée après la gate précédente | 700 PASS |
| Nouveaux tests PNG, inclus dans Inference ordinaire | 19 PASS | Non exécutés | 19 PASS |
| CLIP stock, processus séparé | 3 PASS | Non exécuté | 1 PASS, 2 FAIL |

Les 47 nouveaux tests passent sur Windows et macOS ; seuls les 28 nouveaux tests de métadonnées ont été exécutés sur Linux, soit **122 exécutions réussies** au total pour ces nouveaux cas. La suite PNG n'était pas encore incluse dans la commande IMAGE dédiée de ce commit. Son ajout à cette commande permettra de l'exécuter avant les gates SD lors de la prochaine campagne ; il ne transforme pas l'absence de résultat Linux de cette campagne en réussite.

Windows passe les 2 409 tests de la suite générale, sans échec. macOS passe 2 407 tests et échoue sur les deux comparaisons CLIP de dimensions complètes L et G. Les contrôles Linux échouent sur un cas U-Net, quatre cas CFG, cinq trajectoires Euler et deux parcours réduits SD1.5. Leurs causes restent à établir ; cette campagne ne modifie ni les tolérances ni les références numériques.

Les comptes IMAGE dédiés et de premier accès sont des exécutions supplémentaires de cas déjà présents dans la suite générale. Ils ne sont pas additionnés au total Windows de 2 409 pour gonfler la couverture. Les tests PNG utilisent le lecteur C# décrit dans [les fondations](../PNG_METADATA_FOUNDATIONS.md), sans qualification d'un affichage bitmap réel ni d'une famille de modèles.
