# Moteur et catalogue — CI 849ee96

Les nouveaux tests du moteur et du catalogue passent sur les trois OS dans le [run 34660633065](https://github.com/skyline624/ComfySharp/actions/runs/34660633065), au commit `849ee9672cfbde5ce9d2f81cb116a4d142a58732`. Windows réussit ; Linux et macOS conservent des échecs numériques Inference. Les [preuves JSON](engine-catalogue-ci-849ee96.json) épinglent les 52 TRX, les métadonnées et les fichiers testés du commit.

## Suites exécutées

Les cinq projets applicatifs passent désormais avant les portes Inference. Ils s'exécutent chacun une fois et ne sont pas rejoués dans l'étape Inference ultérieure.

| Projet ou observation | Windows x64 | Linux x64 | macOS arm64 |
|---|---:|---:|---:|
| Core | 136/136 | 136/136 | 136/136 |
| Workflow | 17/17 | 17/17 | 17/17 |
| Tokenization | 540/540 | 540/540 | 540/540 |
| Desktop headless | 32/32 | 32/32 | 32/32 |
| Host | 142/142 | 142/142 | 142/142 |
| Total des cinq projets | 867/867 | 867/867 | 867/867 |
| Inference, hors stock CLIP | 574/574 | Non exécuté | 574/574 |
| Stock CLIP, processus séparé | 3/3 | Non exécuté | 1 pass / 2 échecs |
| Suite complète, tests uniques | 1444/1444 | Incomplète | 1442 pass / 2 échecs |

Les compilations des trois OS terminent sans avertissement ni erreur. Les processus supplémentaires checkpoint (25), contrats pipeline (29) et first-use (194) ne sont pas additionnés à la suite complète. Le first-use passe 194/194 sur Windows/macOS ; Linux passe 178 cas et en échoue 16. En dédupliquant tous les processus réellement exécutés, Linux a observé 1115 tests uniques ; ce n'est pas une exécution complète des 1444 tests.

## Nouvelle preuve moteur et catalogue

Sur chacun des trois OS, les 25 tests comportementaux `ExecutionBlockerTests`, les 23 `ExecutionBlockerReferenceTests` et les 33 `NodeSchemaStaticResolutionTests` passent. Ils sont déjà inclus dans les totaux Core/Host ci-dessus.

Les [références Blocker](execution-blocker-source-fffa6b0.md) couvrent 21 projections directes du mapper source et deux projections lazy. Elles ne prétendent pas reproduire l'instrumentation d'interruption du laboratoire ni PromptExecutor complet. La résolution statique vérifie le plan des 55 occurrences et ses 26 changements de statut ; elle n'exécute pas les classes Python concernées.

L'étape Catalogue réussit sur les trois OS : 1171 lignes, 664 déclarations locales, 25 nœuds enregistrés, zéro erreur, `catalogue_complete=false` et `release_ready=false`. Les tests Host utilisent aussi de vraies opérations sigma : le déplacement des cinq projets ne constitue pas une preuve d'isolation de libtorch.

## Échecs et étapes absentes

Linux conserve 2 échecs U-Net, 6 CFG, 7 Euler et le pipeline `maximum-start`. Ce dernier s'arrête sur `finalDiffusionLatent[15]` : delta `0.00008702278137207031`, borne `0.000053899762630462646`. macOS conserve les deux échecs stock CLIP L/G. Le JSON conserve chaque nom de test et son message exact ; les investigations antérieures restent distinctes, notamment la [preuve pipeline 8264b88](sd15-pipeline-ci-8264b88.md).

Sur Linux, Inference agrégé, les diagnostics CLI suivants, le smoke Desktop/Host et le stock CLIP sont ignorés après l'échec first-use. Aucun succès ne leur est attribué. L'upload stock n'est plus tenté lorsque son producteur est ignoré ; les uploads de traces pipeline/Euler réussissent après le first-use tenté. Les règles d'échec et les bornes numériques restent inchangées.

L'audit télécharge uniquement 14 artefacts de tests parmi les 25 disponibles et vérifie leurs 52 TRX, sans traces de tenseurs ni bibliothèques natives. Il ne recalcule aucune métrique numérique et ne qualifie ni poids réels, famille de modèles, GPU ou disponibilité V1. La tranche CreateList ultérieure n'appartient pas à ce commit.
