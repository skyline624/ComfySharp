# CI StringFormat — b62f628

Les **109 nouveaux cas StringFormat passent sur les trois OS**, soit 327 exécutions réussies. Les cinq suites applicatives passent leurs **1175 tests par OS**, et le vrai smoke Desktop avec Host supervisé réussit également partout. Le [run 34667426207](https://github.com/skyline624/ComfySharp/actions/runs/34667426207), au commit exact [`b62f6286efdd8fa265da8bb798a24f293cb394cf`](https://github.com/skyline624/ComfySharp/commit/b62f6286efdd8fa265da8bb798a24f293cb394cf), reste globalement en échec à cause des portes numériques existantes.

La [preuve JSON](string-format-ci-b62f628.json) conserve les cas, résultats, identifiants par plateforme, 19 hashes de TRX, métadonnées, logs et 20 pins de fichiers au commit testé. L'audit a téléchargé une fois les cinq artefacts utiles après achèvement ; il n'a relancé ni CI, ni .NET, ni calcul source ou natif.

| Groupe nouveau | Windows x64 | Linux x64 | macOS arm64 |
|---|---:|---:|---:|
| Comportement du profil | 36 PASS | 36 PASS | 36 PASS |
| Comparaisons aux observations source | 44 PASS | 44 PASS | 44 PASS |
| Host réel | 12 PASS | 12 PASS | 12 PASS |
| Workflow | 14 PASS | 14 PASS | 14 PASS |
| Desktop headless | 3 PASS | 3 PASS | 3 PASS |
| **Total** | **109 PASS** | **109 PASS** | **109 PASS** |

Les trois faits Avalonia ont des identifiants TRX différents selon l'OS. La matrice les rapproche par leur nom complet exact, sans normaliser noms ou arguments, et conserve chaque identifiant brut. Les 115 identifiants bruts présents sur les trois plateformes représentent bien 109 cas, sans test absent.

## Suites et smoke réellement exécutés

Chaque OS passe Core 333, Desktop 38, Host 201, Workflow 63 et Tokenization 540, soit 1175 cas applicatifs uniques. Les 109 cas StringFormat sont inclus dans ces comptes. Builds et contrôle de cohérence du catalogue réussissent sur les trois jobs.

Le smoke s'exécute avant les portes Inference. Les trois logs contiennent son marqueur de réussite couvrant la fenêtre native, le Host supervisé, les graphes texte/sigma CPU, l'historique sensible à la casse, **StringFormat Host preview** et l'aperçu natif. Le résultat StringFormat `****xy` est exigé par le code de smoke épinglé. Cette preuve de processus n'est pas déduite des tests headless.

| Campagnes ordinaires et CLIP stock, sans doublons | Windows | Linux | macOS |
|---|---:|---:|---:|
| Suites applicatives | 1175 PASS | 1175 PASS | 1175 PASS |
| Inference ordinaire hors CLIP stock | 574 PASS | Non exécuté | 574 PASS |
| CLIP stock en processus distinct | 3 PASS | Non exécuté | 1 PASS / 2 FAIL |
| **Total unique observé** | **1752 PASS** | **1175 PASS, partiel** | **1750 PASS / 2 FAIL** |

Les identifiants ont été dédupliqués dans chaque OS, avec contrôle que le processus CLIP stock n'ajoute pas des cas déjà comptés. Les campagnes checkpoint, pipeline et premier accès natif ne sont pas additionnées à ces totaux. Le total Linux partiel n'atteste pas la réussite d'Inference.

## Échecs numériques conservés

Les neuf processus de premier accès natif passent leurs 194 cas sous Windows et macOS. Sur Linux, les logs annoncent **178 PASS et 16 FAIL sur 194** : deux U-Net, six CFG, sept Euler et le cas pipeline `maximum-start`. Les 16 noms exacts sont conservés dans le JSON et correspondent à ceux du [run précédent](autogrow-names-ci-50c8eb9.md). Ce contrôle de noms ne constitue pas une nouvelle comparaison mathématique. Les comptes de cette porte proviennent des logs ; ses TRX et traces volumineuses n'ont pas été téléchargés pour cet audit borné.

Après cet échec Linux, la suite Inference ordinaire, les diagnostics CLI suivants et CLIP stock ne s'exécutent pas. Aucun succès ne leur est attribué. Sur macOS, les deux échecs CLIP stock L/G restent présents dans les TRX : premiers éléments `final[27]` et `final[5]`, avec les mêmes messages chiffrés que le run précédent. Windows termine avec succès. Aucune cause native ou précision nouvelle n'est déduite ici.

## Portée de la comparaison StringFormat

Les 44 cas source restent répartis en **24 comparaisons de sorties exactes (26 chaînes), 12 frontières d'erreurs et huit refus locaux de fonctionnalités hors profil**. Ils ne deviennent pas 44 succès de formatage. La source demeure l'unique collecte figée issue du laboratoire publié `52fe346c74c2db14b6c73aec804c215674550f4d` ; cette CI ne crée pas trois nouveaux oracles Python.

Les détails des types, de l'ordre d'évaluation, des captures et de la différence descriptive du schéma sont dans la [preuve d'intégration locale](string-format-integration.md). Le port reste `partial`. Cette campagne ne qualifie ni tout Python `str.format`, ni le frontend dynamique, ni tous les workflows HTTP, ni les composants numériques SD/CLIP. Les digests d'archives sont rapportés par GitHub ; les octets des TRX et fichiers publiés ont été rehashés indépendamment.
