# Intégration locale de StringFormat

La suite complète Windows Release passe **1752 tests uniques, zéro échec et zéro ignoré** avec le vrai nœud StringFormat et son profil limité `python-format-text-v1`. Le smoke Desktop et son Host supervisé passent également, avec un aperçu StringFormat vérifié à `****xy`. Cette preuve concerne une implémentation **partial**, pas toutes les fonctionnalités de Python `str.format`.

La [preuve JSON](string-format-integration.json) contient les empreintes des fichiers, des dix TRX et des trois fichiers du smoke. Les résultats ont été recomptés depuis les artefacts persistés, sans relancer .NET, Python source ou le natif. Le build solution a été rapporté par root avec code 0, zéro avertissement et zéro erreur.

## Comparaison à une source indépendante

Le laboratoire a été publié au commit [`52fe346c74c2db14b6c73aec804c215674550f4d`](https://github.com/skyline624/ComfySharp/tree/52fe346c74c2db14b6c73aec804c215674550f4d/labs/string-format-source) avant la première collecte. Il exécute le vrai AST StringFormat du backend `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`, avec le binder et le mapper source, sous Python 3.12.10. La [preuve source](string-format-source-52fe346.md) décrit ses dix blobs, 73 sélections AST et contrôles de provenance.

La collecte de 408527 octets est épinglée par SHA-256 `a6c1aa98c333851195d0bff97f3e1305835634a1cb922dcc102dabe5e07871e0`. Le protocole de 47482 octets porte le SHA-256 `2eae9a7d3719a225110e032e4b12233300c7ae168a6659d140d87778a34aaefd`. Les deux ressources embarquées sont des copies exactes ; leurs tailles et hashes sont vérifiés à chaque lecture par le comparateur.

| Partition prospective | Assertion C# | Résultat local |
|---|---|---:|
| 24 cas comparables | Toutes les sorties égales aux chaînes source | 24 PASS, 26 chaînes exactes |
| 12 erreurs source | Frontière et type Python attestés, catégorie locale distincte | 12 PASS |
| 8 cas hors profil | Refus local explicite malgré le succès Python enregistré | 8 PASS |
| **Total** | **44 cas ordinaires sans skip** | **44 PASS** |

Les 44 PASS ne sont donc pas 44 succès de formatage. Les huit formes hors profil restent refusées. Les diagnostics Python et C# ne sont pas assimilés ; les priorités entre champ absent, conversion et syntaxe sont vérifiées explicitement. Aucun expected n'est fabriqué par un formateur C#.

Les tests utilisent le StringFormatNode réellement enregistré. Un décorateur photographie les arguments managés, puis délègue directement au vrai corps. Ses 46 observations d'entrée sont comparées aux enregistrements source correspondants, avec les ordres des clés. Il n'existe qu'une frontière instrumentée côté C#, comparée séparément au retour du binder et à l'entrée du corps source. Ce nombre décrit les assertions réussies ; il ne désigne pas 46 fichiers de traces C# conservés.

Les producteurs de test matérialisent uniquement les valeurs du cache déclaré par le protocole. Les scalaires JSON et leurs lexèmes numériques sont conservés, sans conversion de float en entier, ni formatage de secours. La mémoïsation des producteurs est contrôlée localement ; elle n'est pas présentée comme une API C# identique aux lectures du cache source. Les prompts, ressources et spécifications restent inchangés et les résultats détenus sont libérés.

Le schéma `object_info`, ses partitions et ordres sont comparés. Seule la description diffère volontairement : la source annonce toutes les fonctionnalités Python, tandis que le port annonce son profil partiel. Ce delta est affirmé, pas masqué comme une égalité complète des métadonnées.

## Exécutions ciblées et campagne complète

La première campagne Core a produit **332 PASS et 1 FAIL sur 333**, sans ignoré. Les 44 comparaisons source et les 36 tests de comportement StringFormat passaient déjà. L'unique échec concernait l'ancien compteur du registre BuiltInNodes : 14 attendus contre 15 nœuds réels après enregistrement. Root a corrigé uniquement ce compteur de test ; le parser, les assertions de formatage et les ressources source sont restés identiques.

Les campagnes ciblées Workflow (63), Desktop (38) et Host (201) passent. Leurs nouveaux cas couvrent respectivement 14 tests de compilation, trois tests d'éditeur et 12 parcours Host réels : schéma, défaut persisté, constante, connexion remplaçant le widget, noms non consécutifs dans le template, mapping avec CreateList et refus explicites. Le smoke s'exécute dans un vrai processus Desktop avec Host supervisé ; il n'est pas déduit des tests headless.

| Projet | Suite complète : PASS |
|---|---:|
| Core | 333 |
| Desktop | 38 |
| Host | 201 |
| Inference | 577 |
| Tokenization | 540 |
| Workflow | 63 |
| **Total unique** | **1752** |

Les six TRX de la suite complète contiennent 1752 identifiants distincts. Les campagnes ciblées sont des sous-ensembles répétés et ne sont pas ajoutées à ce total. Les 109 nouveaux cas comprennent 36 tests du profil, 44 comparaisons source, 12 Host, 14 Workflow et trois Desktop. Le smoke n'est pas compté comme un test xUnit supplémentaire.

## Identités et limites

Dix-neuf fichiers ont été photographiés pendant la validation puis rehashés : 17 fichiers de code, tests et ressources restent identiques ; `docs/STRING_FORMAT.md` et le manifeste reçoivent ensuite leurs annotations de résultat. Cette photographie n'est ni un snapshot avant compilation ni une attestation des binaires chargés. La campagne est décrite par ces pins locaux, sans l'attribuer à un commit antérieur aux changements.

Le manifeste porte désormais `unitTests=true` sur StringFormat `partial`. `realWorkflow`, les six plateformes et les preuves de plateforme restent faux ou vides. Ces parcours ciblés ne qualifient pas tous les workflows source, le HTTP complet, le frontend dynamique ou une plateforme entière. Les limites fonctionnelles et les refus du [profil StringFormat](../STRING_FORMAT.md) restent applicables. Aucune nouvelle qualification GPU, numérique des composants SD/CLIP, ni parité Python générale n'est déduite de cette campagne locale.
