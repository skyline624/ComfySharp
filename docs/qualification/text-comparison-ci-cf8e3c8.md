# Comparaison textuelle : CI du commit cf8e3c8

Les **224 nouveaux cas passent sur les trois OS**, soit 672 exécutions réussies, dans la [CI Build and test 34669615538](https://github.com/skyline624/ComfySharp/actions/runs/34669615538) du commit [cf8e3c85dbd12b0210d4846bf6825a1d37c67e51](https://github.com/skyline624/ComfySharp/commit/cf8e3c85dbd12b0210d4846bf6825a1d37c67e51). Le build, le contrôle du catalogue, les cinq suites applicatives et le nouveau smoke Desktop/Host réussissent partout. **La CI globale reste en échec** à cause des portes numériques d'inférence sur Linux et macOS.

| Runner | Cinq suites applicatives | Nouveaux cas inclus | Smoke réel | Tests ordinaires + CLIP stock observés | Job |
|---|---:|---:|---|---:|---|
| Windows x64 | 1399 PASS | 224 PASS | PASS | 1976 PASS | Succès |
| Linux x64 | 1399 PASS | 224 PASS | PASS | 1399 PASS ; Inference ordinaire et CLIP stock non exécutés | Échec |
| macOS ARM64 | 1399 PASS | 224 PASS | PASS | 1974 PASS, 2 FAIL | Échec |

Les cinq suites représentent Core 525, Desktop 40, Host 225, Tokenization 540 et Workflow 69 par OS. Les 224 nouveaux cas sont déjà inclus : 35 comportements des nœuds, 48 tests du helper Unicode, 68 plans de digests, 40 cas source des nœuds, une régression V3 COMBO, 24 Host, six Workflow et deux Desktop. Les résultats sont recomptés dans **19 TRX issus de cinq artefacts**. Les identités ordinaires et stock sont disjointes dans chaque OS ; les répétitions ciblées et les premiers accès natifs ne sont pas additionnés à ces totaux.

Les deux tests Avalonia possèdent des testId différents selon l'OS : 228 IDs bruts correspondent aux 224 mêmes noms complets, arguments compris. Le rapprochement n'altère ni nom ni argument. Les empreintes des listes de noms et d'IDs par groupe/OS, des TRX, des métadonnées et des fichiers du commit figurent dans la [preuve JSON](text-comparison-ci-cf8e3c8.json).

## Ce que les nouveaux tests vérifient

La [preuve locale](text-comparison-integration.md) décrit les frontières reprises en CI. Les 36 cas source comparables comprennent 39 booléens après mapping et un blocker sans appel de corps. Les deux erreurs source typées sont distinguées du chemin moteur C# qui normalise les littéraux STRING. Les deux modes invalides retournent zéro sortie dans la collecte sans validation globale ; le moteur local les refuse à l'admission COMBO. Il ne s'agit donc pas de 40 succès de prédicat identiques.

Les 68 plans comparent les empreintes d'entrée et de sortie du vrai helper aux références CPython 3.12.10 : 17 plans Unicode et quatre contextes fixes. La source a effectué trois passages ; chaque test C# effectue un passage. La même collecte figée est utilisée sur les trois OS. Ces tests ne deviennent ni trois nouveaux oracles source ni une preuve de toutes les chaînes Unicode possibles.

Le smoke s'exécute avant les portes d'inférence. Le code Desktop épinglé crée les vrais nœuds, connecte PrimitiveBoolean pour imposer `case_sensitive=false`, compile les workflows et les soumet au Host supervisé. Il exige l'aperçu **True** pour Contains sur `İ` / `i` + point combinant et **False** pour Ends With sur `ΟΣ` / `Σ`, puis applique les résultats aux previews. Son marqueur de succès est présent dans les trois logs seulement après ces assertions ; ce contrôle dépasse les tests d'éditeur sans fenêtre.

## Portes numériques toujours ouvertes

Les logs consignent neuf processus de premier accès natif, avec 194 résultats dans cette série ciblée. Windows et macOS passent les 194. Linux a **179 PASS et 15 FAIL** : trois U-Net, quatre CFG, sept Euler et un pipeline SD1.5 (`maximum-start`). Cette porte empêche ensuite l'Inference ordinaire et les diagnostics CLI/natifs tardifs ; leur absence n'est pas un succès.

Par rapport à la [CI StringFormat précédente](string-format-ci-b62f628.md), Linux présente un échec supplémentaire U-Net `sd15-reduced/batch-distinct-time`, tandis que les deux politiques CFG `guidance-2-10` ne sont plus en échec. Le total passe ainsi de 178/16 à 179/15. Ce constat porte sur les résultats observés, sans attribution causale ni modification de tolérance ou de fixture.

macOS passe l'Inference ordinaire puis échoue sur les deux dimensions CLIP stock. Les messages restent exactement ceux de la CI précédente : G `final[5]` a une erreur absolue `6.508827209472656e-5` pour une borne `5.667007863521576e-5` ; L `final[27]` a une erreur `4.503130912780762e-5` pour une borne `4.443028271198273e-5`. Windows passe également les trois tests CLIP stock.

Cet audit lit les métadonnées finales, les logs et les TRX, sans replay .NET, source ou natif local, et sans nouvelle analyse des traces mathématiques volumineuses. Les 35 pins de la preuve locale correspondent aux octets du commit publié ; le workflow et la preuve locale sont également épinglés. Les empreintes de fichiers ne prétendent pas attester toutes les bibliothèques chargées. Ces résultats bornés ne qualifient ni tous les workflows, ni une plateforme entière, ni de nouveaux profils numériques d'inférence.
