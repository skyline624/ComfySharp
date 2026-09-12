# ImageBatch : CI du commit 8a261a0

La [campagne 34674597695](https://github.com/skyline624/ComfySharp/actions/runs/34674597695) teste exactement `8a261a0f0e1617d23fea15bc5dd6fafc0cac3eb5`. Les **53 nouveaux tests passent sur les trois OS**, soit 159 exécutions réussies, et le parcours natif Desktop/Host avec ImageBatch termine également sur les trois configurations. La campagne globale reste en échec.

| Configuration CPU | Contrats applicatifs | Suite IMAGE dédiée | Autres résultats |
|---|---:|---:|---|
| Windows x64 | 1 678 PASS | 107 PASS | Premier accès natif : 176 PASS, 18 FAIL ; Inference ordinaire et CLIP stock non exécutés ensuite |
| Linux x64 | 1 678 PASS | 107 PASS | Premier accès natif : 178 PASS, 16 FAIL ; Inference ordinaire et CLIP stock non exécutés ensuite |
| macOS ARM64 | 1 678 PASS | 107 PASS | 194 références de premier accès natif PASS ; total ordinaire et stock : 2 360 PASS, deux FAIL CLIP |

La suite IMAGE dédiée comprend les 65 cas des quatre primitives précédentes et 42 nouveaux cas ImageBatch : 18 opérations, cinq nœuds et 19 comparaisons source. Les onze autres nouveaux cas sont sept Host, trois Workflow et un Desktop. Les suites dédiées et de premier accès recouvrent Inference ; elles ne sont pas ajoutées au total comme des tests uniques.

Les assertions du smoke compilent deux images de tailles différentes, exécutent ImageBatch dans le Host supervisé puis extraient la dernière image jusqu'à son texte PreviewAny. Sous Linux, la fenêtre native utilise Xvfb. Cette route à couleurs uniformes confirme le fonctionnement applicatif ; le corpus source distinct vérifie l'interpolation. Aucun PNG ni aperçu raster n'est ajouté par cette révision.

Par rapport à la [campagne c4dc65d](image-primitives-ci-c4dc65d.md), Windows présente 18 nouveaux échecs de premier accès natif : cinq U-Net, six CFG, cinq Euler et deux pipeline. Linux conserve les mêmes 16 noms en échec. Les deux messages CLIP stock macOS restent identiques. Les graphes, fixtures et seuils numériques SD/CLIP existants sont conservés. Cette comparaison constate des résultats différents sans attribuer leur cause au CPU, à l'allocation, au bundle natif ou aux nœuds IMAGE.

L'audit porte sur 47 TRX téléchargés depuis dix artefacts de cette révision exacte. Les compteurs ont été rapprochés des résultats et des neuf processus de référence par OS. Les 53 nouveaux noms ont des identités distinctes ; les IDs du test Avalonia peuvent varier entre plateformes. Les marqueurs des trois smokes sont présents dans leurs étapes terminées avec succès. Les suites sautées restent explicitement non exécutées.

La qualification obtenue concerne le profil IMAGE CPU/Float32 borné. CUDA/MPS, autres dtypes, codecs, modèles préentraînés et catalogue complet restent ouverts. Les contrôles numériques en échec empêchent toujours l'intégration de cette branche dans `main`.
