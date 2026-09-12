# Autogrow / CreateList — CI du commit 3eafec3

Le [run 34662056211](https://github.com/skyline624/ComfySharp/actions/runs/34662056211), au commit `3eafec37fd8d76cb4d43c34626224da5a1d3d4f2`, est **en échec**. Les 55 nouveaux tests applicatifs Autogrow/CreateList passent sur les trois systèmes. Des références numériques SD échouent dans les processus Windows et Linux de premier accès natif ; macOS échoue sur deux références CLIP aux dimensions complètes. Ces résultats ne constituent pas une qualification globale du moteur ni des familles de modèles.

Les métadonnées, les empreintes des 50 TRX, les résultats individuels, les identités observées et les comparaisons sont consignés dans la [preuve JSON](autogrow-ci-3eafec3.json). Cet audit a uniquement lu les artefacts : aucun test, calcul source, chargement natif ou workflow supplémentaire n'a été exécuté.

## Couverture applicative effectivement exécutée

| Suite | Windows | Linux | macOS |
|---|---:|---:|---:|
| Core | 170 / 170 | 170 / 170 | 170 / 170 |
| Workflow | 31 / 31 | 31 / 31 | 31 / 31 |
| Desktop sans interface native | 35 / 35 | 35 / 35 | 35 / 35 |
| Host | 146 / 146 | 146 / 146 | 146 / 146 |
| Tokenization | 540 / 540 | 540 / 540 | 540 / 540 |

Les 55 nouveaux cas se répartissent en 19 `AutogrowInputTests`, 15 `CreateListTests`, 14 `CreateListWorkflowTests`, 4 `CreateListHostTests` et 3 `CreateListEditorTests`. Chacun est retrouvé avec un résultat `Passed` dans chaque OS. Cette couverture porte sur les contrats, la propagation des listes, les diagnostics, le parcours Host et l'éditeur testable sans fenêtre. Les 18 comparaisons avec le laboratoire source Autogrow, ajoutées après ce commit, **ne font pas partie de ce run** ; leur preuve reste séparée.

Le contrôle du catalogue passe sur les trois OS. Les processus supplémentaires de contrats checkpoint (25 cas) et pipeline réduit (29 cas) passent également. Ils recouvrent des tests de la suite ordinaire et ne doivent pas être additionnés comme une couverture indépendante supplémentaire.

## Échecs numériques conservés

| Processus de référence | Windows réussis / total | Linux réussis / total | macOS réussis / total |
|---|---:|---:|---:|
| Premier accès natif, neuf suites séparées | 176 / 194 | 179 / 194 | 194 / 194 |
| Dont U-Net réduit | 3 / 8 | 5 / 8 | 8 / 8 |
| Dont CFG | 0 / 6 | 2 / 6 | 6 / 6 |
| Dont Euler | 3 / 8 | 1 / 8 | 8 / 8 |
| Dont pipeline réduit | 2 / 4 | 3 / 4 | 4 / 4 |
| CLIP aux dimensions complètes | non exécuté | non exécuté | 1 / 3 |

PreviewAny, les schedules sigma, les petites références CLIP, le sampling et le VAE passent dans les neuf processus de premier accès. Les 18 échecs Windows et 15 Linux sont des assertions numériques, sans erreur de chargement observée à leur place. Les fenêtres Windows concernent cinq U-Net, les six CFG, cinq Euler et les pipelines `maximum-start` et `two-chunks-separate`.

Windows et Linux s'arrêtent avant la suite Inference agrégée et les étapes natives suivantes. Il serait donc incorrect de leur attribuer un passage complet des 1 499 tests. macOS exécute les 1 496 tests ordinaires avec succès, puis trois tests CLIP : 1 497 succès et 2 échecs au total pour ces deux ensembles disjoints. Ses premiers dépassements CLIP sont `L/final[27]`, erreur `4.503130912780762e-5`, et `G/final[5]`, erreur `6.508827209472656e-5`, contre la référence Windows historique. Le succès local rapporté séparément ne remplace aucun de ces résultats CI.

## Intégrité et observations disponibles

Les 23 artefacts annoncés ont été récupérés ; celui du premier accès Windows avait déjà été téléchargé par la racine. Les 50 TRX ont des compteurs cohérents avec leurs résultats, sans test ignoré ni erreur globale dissimulée. Les notifications xUnit des assertions échouées restent des erreurs de test légitimes. Les digests ZIP sont ceux annoncés par GitHub ; cet audit a rehashé les fichiers extraits, sans prétendre avoir recalculé les digests des archives ZIP.

Les 60 traces SD contiennent **972 payloads F32**, tous décodés comme de vrais F32 little-endian puis rehashés. Les 18 payloads binaires des deux captures CLIP macOS sont également vérifiés, avec dimensions, taille et finitude. Les neuf fichiers de référence SD utilisés pour comparer les sorties sont identiques aux blobs du commit testé. Le seuil de comparaison reste `3e-5 + 3e-5 × abs(référence)`.

Les 24 traces Euler attestent les 686 noms, formes et hashes de paramètres avant/après, identiques à leur référence. Les 12 traces pipeline font de même pour leurs 971 paramètres. Les entrées conservées sont inchangées, les scalaires sigma/sigmaHat comparés sont exacts, et les trois sorties off/on/off de chaque trajectoire/pipeline sont identiques entre elles. Les traces U-Net plus anciennes conservent l'identité du latent et dix frontières, avec un seul résultat capturé : elles ne fournissent pas la même preuve brute des trois répétitions ni les octets de tous les paramètres.

Le workflow n'active pas de trace de sortie CFG dédiée. Les six échecs Windows et quatre Linux sont donc établis par les assertions TRX ; aucune métrique exhaustive de leurs sorties n'est inventée à partir de ces messages.

## Identités natives et limites de localisation

Windows rapporte un **Intel Xeon Platinum 8370C** ; Linux un **AMD EPYC 9V74**. Les deux exposent AVX-512 au CPUID et aux intrinsèques .NET. Les 60 captures SD rapportent un thread intra-op et un thread inter-op, TorchSharp `0.107.0.0`, .NET `10.0.12`, et une version libtorch *déclarée* `2.10.0`. `ATEN_CPU_CAPABILITY` est absent. **Le dispatch ATen effectif n'est pas exposé par cette observation.**

Windows fournit quatre images chargées distinctes : `torch_cpu.DLL`, `c10.dll`, `LibTorchSharp.DLL` et `libiomp5md.dll`. Linux en fournit cinq : les trois bibliothèques cœur, le pont TorchSharp et OpenMP. Les ensembles basename/taille/SHA sont stables entre les familles U-Net, Euler et pipeline de chaque OS, sans doublon observé. L'inventaire est capturé après calcul puis mis en cache dans chaque processus ; ce n'est pas une vérification de chargement avant/après. Sur macOS, l'énumération retourne explicitement `unavailable` ; le succès de calcul n'est pas utilisé pour fabriquer une identité binaire.

Une lecture complémentaire des huit traces U-Net Windows du [run précédent 34660633065](https://github.com/skyline624/ComfySharp/actions/runs/34660633065) montre un **AMD EPYC 7763 sans AVX-512**. Les quatre DLL ont exactement les mêmes tailles et hashes que dans le run actuel ; OS, runtime .NET et threads correspondent aussi. Le diff Git `849ee96 → 3eafec3` ne change aucun calcul Inference, aucune fixture ni le workflow normal : dans ce périmètre, `TensorNodeTests` modifie uniquement le nombre attendu de nœuds enregistrés au catalogue, de 25 à 26.

Les huit sorties `timeEmbedding` sont déjà différentes entre ces deux runs Windows : maximum `1.1920928955078125e-7`, ou `2.384185791015625e-7` pour les cas carrés. Toutes les frontières internes conservées restent dans le seuil ; cinq sorties U-Net finales le dépassent. Le maximum final est `0.00017480552196502686` pour `sd2-reduced/batch-shared-time-odd` (53 éléments hors seuil). Le booléen d'alignement du latent diffère aussi pour un autre cas. Il n'existe ici ni contrôle de tous les inputs d'une primitive, ni contrôle du contexte d'allocation ou des assemblages managés entre hôtes : **ces observations corrélées ne prouvent pas une causalité CPU ou une primitive responsable**.

Dans les pipelines Windows, les sorties CLIP diffèrent déjà légèrement de la référence, tout en restant dans le seuil ; `maximum-start/initialDiffusionLatent` diffère aussi de `4.76837158203125e-7`. Il serait incorrect d'attribuer tout l'écart final au seul U-Net ou au VAE. Pour ce cas, le maximum final diffusion vaut `0.0003662109375`, puis `0.00201416015625` après conversion du latent VAE. Les métriques détaillées et les nombres d'éléments hors seuil figurent dans le JSON, séparément du premier échec affiché par chaque test.

Toute expérience suivante doit isoler un facteur mesurable sur le même hôte, avec entrées, allocations et identités contrôlées, puis garder les références acceptées intactes. Cet audit ne propose ni nouveau profil, ni remplacement de référence, ni adoption d'un bundle natif. La CI demeure rouge.
