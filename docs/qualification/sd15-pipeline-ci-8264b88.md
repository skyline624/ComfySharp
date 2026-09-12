# Pipeline SD1.5 réduit — CI 8264b88

Le [run Build and test 34658845153](https://github.com/skyline624/ComfySharp/actions/runs/34658845153), au commit `8264b8886c250dfd37ae6906113a8a531b719494`, réussit sur Windows et échoue sur Linux/macOS. **Les nouvelles références pipeline passent sur Windows et macOS ; `maximum-start` échoue sur Linux.** Les trois compilations terminent sans avertissement ni erreur. Les [preuves JSON](sd15-pipeline-ci-8264b88.json) conservent les hashes des TRX, traces et références, les métriques et les identités natives.

| Observation | Windows x64 | Linux x64 | macOS arm64 |
|---|---:|---:|---:|
| Contrats checkpoint | 25/25 | 25/25 | 25/25 |
| Contrats pipeline | 29/29 | 29/29 | 29/29 |
| Références au premier accès natif | 194/194 | 178 pass / 16 échecs | 194/194 |
| Nouvelles références pipeline | 4/4 | 3/4 | 4/4 |
| Suite complète, stock CLIP inclus une fois | 1363/1363 | Non exécutée | 1361 pass / 2 échecs |
| Tests du helper d'entrées alignées, inclus dans la suite | 8/8 | Non exécutés | 8/8 |

Les 16 échecs Linux regroupent les 15 cas historiques (2 U-Net, 6 CFG et 7 Euler) et le nouveau `maximum-start`. Les deux échecs macOS restent ceux du CLIP stock. Les tests répétés dans plusieurs processus ne sont pas additionnés à la suite complète. L'audit lit 47 TRX et 22 des 24 artefacts disponibles ; les deux artefacts binaires CLIP stock ne sont pas téléchargés pour cette analyse pipeline, mais leurs résultats TRX sont conservés. Linux présente aussi un échec d'upload des traces CLIP stock, absentes puisque l'étape correspondante n'a pas été exécutée ; ce n'est pas un test numérique supplémentaire.

## Comparaison des captures complètes

Chaque OS est comparé à sa propre référence issue de la [collecte source 6ad](https://github.com/skyline624/ComfySharp/actions/runs/34656804132). Aucun seuil ni expected n'est modifié. Le [profil prospectif](sd15-pipeline-native210-cpu-f32-v1.md) conserve `abs(actual-source) <= 3e-5 + 3e-5*abs(source)` ; les indices et les scalaires sigma/sigmaHat exigent des bits exacts.

| Traces par OS | Windows | Linux | macOS |
|---|---:|---:|---:|
| Captures distinctes bit exactes, sur 64 / 28 832 valeurs | 64 | 40 | 64 |
| Observations hors borne dans ces 64 records | 0 | 87 | 0 |
| Comparaisons finales répétées bit exactes, sur 36 / 48 000 valeurs | 36 | 0 | 36 |
| Observations finales répétées hors borne | 0 | 180 | 0 |

Les 971 paramètres par cas, les entrées empruntées bruit/sigmas, les textes et options source sont exacts avant/après sur les trois OS. Les hashes des trois répétitions off/on/off sont stables, les captures restent sans grad et l'état grad de l'appelant est rétabli. Les 36 comparaisons répètent douze sorties sémantiques et ne constituent pas 36 frontières uniques supplémentaires.

## Localisation observée de l'échec Linux

Pour `maximum-start`, les quatre sorties CLIP, le latent initial, l'état `x` du premier pas et tous les scalaires sont bit exacts. **La première capture divergente est `step-0/denoised`** : les 80 valeurs diffèrent, 9 dépassent la borne, maxAbs `0.0005903244018554688`. Cette observation ne désigne pas encore une primitive interne : elle englobe la mise à l'échelle EPS, U-Net et CFG.

| Capture Linux `maximum-start` | MaxAbs | Éléments hors borne |
|---|---:|---:|
| Premier denoised | 0.0005903244018554688 | 9 |
| x au second pas | 0.000530242919921875 | 9 |
| Second denoised | 0.0005540847778320312 | 9 |
| Latent de diffusion final | 0.0005540847778320312 | 9 |
| Latent après division par 0.18215 | 0.00304412841796875 | 10 |
| Image | 0.00007367134094238281 | 41 |

L'assertion C# s'arrête au latent final, élément 15 : delta `0.00008702278137207031`, borne `0.000053899762630462646`. Toutes les captures avaient déjà été écrites : l'audit révèle donc aussi les écarts des étapes et de l'image. Les sorties finales contiennent 60 éléments hors borne, repris trois fois dans les 180 observations répétées. Les trois autres cas Linux restent dans la borne malgré des différences F32 ; leur première différence capturée apparaît également au denoised du premier pas.

Les anciennes références U-Net/CFG/Euler échouent dans ce même job Linux. Cette coexistence suggère un périmètre à examiner, mais ne démontre ni une cause commune, ni un défaut d'alignement, ni un kernel particulier. Aucun ajustement de profil ou de politique n'a été fait.

## Identités et portée

Les traces rapportent .NET 10.0.12, TorchSharp 0.107.0.0, le package libtorch déclaré 2.10.0 et intra/inter-op 1. Les maps natives sont relevées après calcul via un cache de processus ; leurs SHA restent distincts des inventaires de package source. Le produit ne mesure pas son dispatch ATen effectif : les capacités .NET AVX2/AVX512/Arm ne le remplacent pas. Les détails par OS, y compris les informations explicitement indisponibles, sont conservés dans le JSON.

Cette preuve étend l'[observation Windows locale](sd15-pipeline-integration.md) à un run CI précis, avec un échec Linux encore ouvert. Elle ne qualifie pas des poids préentraînés, les largeurs complètes, une famille de modèles, le GPU, un nœud de génération ou la disponibilité V1. La tranche blockers ultérieure n'appartient pas à ce commit.
