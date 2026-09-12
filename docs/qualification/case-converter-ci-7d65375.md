# CaseConverter — CI du 12 septembre 2026

La [campagne 34671553840](https://github.com/skyline624/ComfySharp/actions/runs/34671553840)
teste le commit `7d65375ceeec8b7e3b1cd9b395c1d7d91e6d30e7`. Les **249 nouveaux cas passent
sur les trois OS**, soit 747 exécutions. Les trois parcours réels Desktop/Host vérifient
également les quatre modes de CaseConverter. La campagne globale échoue aux contrôles
numériques des modèles déjà en investigation.

| Cible | Tests applicatifs | Références en processus séparés | Suite complète observée | Job |
|---|---:|---:|---:|---|
| Windows x64 CPU | 1 648 PASS | 194 PASS | 2 225 PASS | Réussi |
| macOS ARM64 CPU | 1 648 PASS | 194 PASS | 2 223 PASS, 2 échecs CLIP | Échec |
| Linux x64 CPU | 1 648 PASS | 182 PASS, 12 échecs | Suite Inference ordinaire et stock non exécutée ensuite | Échec |

Les applications regroupent Core 750, Desktop 42, Host 244, Tokenization 540 et Workflow 72.
Les 249 nouveaux cas sont ceux de la [qualification locale](case-converter-integration.md) :
28 comportements du nœud, 68 du helper Unicode, 85 digests source, 40 scénarios source du
nœud, quatre scénarios d'ordre des arguments, 19 Host, trois Workflow et deux Desktop.
Les références lancées dans des processus séparés recouvrent les tests de la suite complète
et ne s'ajoutent pas à son total.

Les trois étapes de démarrage graphique passent après les assertions de l'application
publiée : document compilé, Host supervisé, historique et sorties des quatre modes.
Linux utilise Xvfb. Ces contrôles CPU ne qualifient ni CUDA ni MPS, ni une famille de modèles.

Par rapport à la [campagne cf8e3c8](text-comparison-ci-cf8e3c8.md), les deux diagnostics
CLIP macOS restent identiques. Linux présente désormais un échec U-Net, quatre CFG,
cinq Euler et deux pipeline. Le scénario pipeline `two-chunks-separate` échoue ; les
scénarios Euler `near-one-null` et `near-one-present`, ainsi que les scénarios U-Net
SD1.5 `batch-distinct-time` et `odd-rectangle`, passent dans cette campagne.

Les graphes de calcul, fixtures, tolérances et workflow CI de ces contrôles n'ont pas changé
entre les deux commits. Ces variations constatées ne démontrent pas une correction et
ne permettent pas d'attribuer une cause matérielle. La branche reste en qualification.

L'audit a lu les 46 TRX de huit artefacts et les journaux des trois jobs après achèvement.
Il a vérifié les compteurs et identités uniques par processus, les 249 noms communs aux
trois OS, les étapes de smoke et 35 pins de blobs Git du commit testé. Deux tests Avalonia
possèdent des IDs distincts selon l'OS, soit 253 IDs pour 249 noms ; aucun test n'a été
doublement compté. Les digests d'archives fournis par GitHub sont distincts des hashes
recalculés sur les fichiers TRX extraits. Aucun replay source ou calcul natif supplémentaire
n'a été effectué pour cet audit.
