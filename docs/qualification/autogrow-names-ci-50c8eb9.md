# CI TemplateNames et valeurs de prompt — 50c8eb9

Les **36 tests de contrat TemplateNames et les 28 comparaisons de phases Prompt Values passent sur Windows, Linux et macOS** : 64 cas distincts, 192 exécutions réussies. Les cinq suites applicatives passent également leurs 1037 tests sur chaque OS. Le [run 34665300013](https://github.com/skyline624/ComfySharp/actions/runs/34665300013), au commit exact [`50c8eb9ae1d37d9aa5f9bd8ea9fbdd733513068f`](https://github.com/skyline624/ComfySharp/commit/50c8eb9ae1d37d9aa5f9bd8ea9fbdd733513068f), reste globalement en échec à cause des portes numériques décrites ci-dessous.

La [preuve JSON](autogrow-names-ci-50c8eb9.json) contient les 64 noms et identifiants de tests, leurs résultats par OS, les hashes des 19 TRX téléchargés, les pins des métadonnées et logs, ainsi que les fichiers exacts du commit testé. Cet audit n'a exécuté aucun test ni relancé la CI.

| Périmètre | Windows x64 | Linux x64 | macOS arm64 |
|---|---:|---:|---:|
| Core | 224 PASS | 224 PASS | 224 PASS |
| Workflow | 49 PASS | 49 PASS | 49 PASS |
| Tokenization | 540 PASS | 540 PASS | 540 PASS |
| Desktop headless | 35 PASS | 35 PASS | 35 PASS |
| Host | 189 PASS | 189 PASS | 189 PASS |
| **Sous-total applicatif unique** | **1037 PASS** | **1037 PASS** | **1037 PASS** |
| Dont nouveaux cas Names / Prompt Values | 36 / 28 PASS | 36 / 28 PASS | 36 / 28 PASS |
| Smoke Desktop natif et Host supervisé | PASS | PASS | PASS |
| Inference ordinaire, hors CLIP stock | 574 PASS | Non exécuté | 574 PASS |
| CLIP stock, processus distinct | 3 PASS | Non exécuté | 1 PASS / 2 FAIL |
| **Total unique ordinaire + stock observé** | **1614 PASS** | **1037 PASS, partiel** | **1612 PASS / 2 FAIL** |

Les 64 cas sélectionnés sont inclus dans les totaux applicatifs. Les campagnes checkpoint, pipeline et premier accès natif répètent des cas et ne sont pas ajoutées au total unique. Les identifiants des suites ordinaires et du processus CLIP stock ont été contrôlés disjoints. Le total Linux partiel n'atteste pas la réussite d'Inference.

## Le smoke précède maintenant les portes numériques

Les builds et le contrôle de cohérence du catalogue réussissent sur les trois OS. Le smoke Desktop s'exécute en étape 8, après les suites applicatives et le catalogue, avant les contrats Inference et le premier accès natif aux références. Les trois jobs enregistrent son marqueur de réussite : fenêtre native, Host supervisé, graphes texte et sigma CPU, historique UI sensible à la casse et aperçu natif. Il s'agit d'un processus distinct ; sa réussite n'est pas déduite des tests Desktop headless.

Ce déplacement rend le résultat UI visible sur Linux malgré l'échec numérique ultérieur. Il ne modifie ni les assertions, ni la politique d'échec des références.

## Échecs et étapes non exécutées

Windows termine avec succès. Sur macOS, les deux échecs sont les références CLIP stock L et G. Les premiers éléments signalés sont respectivement `final[27]` (erreur absolue `4.503130912780762e-5`, borne `4.443028271198273e-5`) et `final[5]` (erreur `6.508827209472656e-5`, borne `5.667007863521576e-5`). Ces messages proviennent des TRX rehashés ; aucune nouvelle attribution numérique n'est proposée.

Sur Linux, les logs des neuf processus de premier accès natif annoncent **178 PASS et 16 FAIL sur 194 cas**, sans ignoré : deux U-Net, six CFG, sept Euler et le cas pipeline `maximum-start`. Le JSON conserve les 16 noms exacts. Ces comptes proviennent des logs, dont le hash est enregistré ; les TRX de cette porte n'ont pas été téléchargés pour cet audit borné. Les familles d'échecs restent distinctes de la validation des contrats applicatifs et ne sont pas réinvestiguées ici.

L'échec de cette porte empêche ensuite l'exécution de la suite Inference ordinaire, des diagnostics CLI suivants et du processus CLIP stock sur Linux. Aucun succès n'est attribué à ces étapes. Les métadonnées conservent les étapes en échec et celles non exécutées.

## Portée de cette preuve

Les tests Names de ce commit sont les 36 tests de comportement du contrat. Le comparateur de 25 cas source Names et la correction ultérieure de l'ordre des arguments sont absents de `50c8eb9` ; ils ne sont pas qualifiés par cette CI. La [preuve locale Names](autogrow-names-local.md) décrit la tranche testée ici.

Les 28 tests Prompt Values conservent leurs divergences explicites : 26 verdicts de validation comparables et deux divergences, 13 acquisitions comparables et une divergence d'enveloppe de lien, avec trois seconds passages adaptés. Leur réussite n'établit ni une parité de tous les diagnostics, ni une parité du corps des nœuds source ou du protocole HTTP complet. Voir la [preuve de comparaison des phases](prompt-values-integration.md).

Les cinq artefacts utiles ont été téléchargés une fois après achèvement. Leurs 19 TRX ont été rehashés et recomptés ; les digests des archives sont ceux rapportés par GitHub. Les traces mathématiques volumineuses n'ont pas été téléchargées ou comparées. Cette preuve ne modifie aucun profil, aucune borne, aucune fixture, ni les limites numériques antérieures.
