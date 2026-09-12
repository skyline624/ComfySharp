# Comparaison textuelle : intégration locale

La campagne locale Windows termine avec **1 976 tests réussis, zéro échec et zéro test ignoré**, dans six TRX. Le build solution Release et la commande globale ont retourné 0 selon le runner root, avec zéro avertissement et zéro erreur de compilation ; les compteurs, résultats uniques et empreintes des six TRX ont été vérifiés séparément en lecture seule. Les [preuves structurées](text-comparison-integration.json) conservent les pins et les campagnes antérieures.

| Projet | Tests réussis |
|---|---:|
| Core | 525 |
| Desktop | 40 |
| Host | 225 |
| Inference | 577 |
| Tokenization | 540 |
| Workflow | 69 |

Les **224 nouveaux cas** se répartissent en 35 comportements des nœuds, 48 tests du helper Unicode, un contrôle de projection V3 COMBO, 68 comparaisons des empreintes lowercase, 40 cas du corpus des nœuds, 24 tests Host, six Workflow et deux Desktop. Les campagnes ciblées déjà réussies sont des sous-ensembles ou des répétitions ; elles ne s'ajoutent pas au total.

`StringContains` et `StringCompare` sont réellement enregistrés, compilables dans un workflow et disponibles dans l'éditeur. Les noms, l'ordre des entrées, les trois choix Starts With / Ends With / Equal, les sorties BOOLEAN et le défaut `case_sensitive=true` suivent les classes figées. La projection V3 conserve `["COMBO", {"multiselect": false, "options": [...]}]`, tandis que le test protège aussi la projection legacy de StringTrim et l'immuabilité des options.

## Référence indépendante des calculs C#

Le [laboratoire source publié](text-comparison-source-5466f97.md) au commit `5466f97312d8dbdbe5c656a6c31061895afc9af3` précède la collecte. Ses 10 blobs et 75 déclarations AST viennent de ComfyUI `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`. Le corpus CPython 3.12.10 / Unicode 15.0.0, SHA256 `12935c33d4632a79e499a7c24b052d88036aee8323e45ea92d923443b2b9e04f`, est embarqué sans modification avec son protocole. Aucun résultat attendu n'est calculé à partir du helper C# ou de sa table.

Les 68 tests invoquent réellement `PythonUnicodeLower.Lower` pour les 1 112 064 scalaires valides dans quatre constructions fixes : le scalaire seul, suivi de Σ, précédé de A puis suivi de Σ, et placé entre AΣ et A. Cela représente **4 448 256 entrées par passage**. Chaque entrée et sortie est cadrée par le codepoint uint32 little endian, la longueur UTF-8 stricte uint32 little endian, puis les octets UTF-8. Compteurs, longueurs et empreintes d'entrées et de sorties concordent avec les trois passages source, soit 13 344 768 appels builtin lors de la collecte. Le comparateur C# fait un passage par test, pas trois. Ces quatre contextes ne couvrent pas toutes les chaînes possibles.

Les 40 tests des nœuds passent par les véritables nœuds enregistrés et le moteur. **36 cas sont comparables** : 35 cas booléens produisent 39 valeurs en raison du mapping de listes ; le dernier propage un blocker sans appeler le corps. Métadonnées, arguments observés, ordre, sorties et signalement du blocker sont comparés au corpus.

Les quatre autres cas gardent leurs différences explicites. Pour null et un nombre, la source atteint le corps et lève TypeError ou AttributeError. Le moteur local normalise ces valeurs en chaînes `None` et `7` et renvoie false dans ces deux exemples ; une invocation typée directe locale les refuse avec ArgumentException. Pour les deux modes hors COMBO, la source observée sans validation globale retourne zéro sortie ; l'admission locale refuse le mode avant le corps. Aucun de ces cas ne revendique une égalité d'exceptions ou une sortie booléenne source inexistante.

## Parcours Desktop et historique des corrections

Le smoke réel ouvre la fenêtre, utilise le Host supervisé et compile les nœuds ajoutés à l'éditeur. Un lien PrimitiveBoolean impose `case_sensitive=false`. `StringContains` compare U+0130 à `i` + U+0307 et affiche **True** ; `StringCompare` teste Ends With sur `ΟΣ` et `Σ` et affiche **False**. Les deux résultats sont lus dans l'historique terminé du Host puis appliqués aux previews. Le marqueur stdout, le résultat exit0 et stderr vide sont épinglés ; cette vérification ne se réduit pas à construire des contrôles UI.

Les premiers échecs restent conservés. Une campagne préalable de 29 assertions de nœuds était rouge avant implémentation. Le premier Core avait 406 Passed / 3 Failed sur 409 résultats et une alerte de découverte pour ID dupliqué : les surrogates invalides placés dans InlineData avaient été normalisés dans les métadonnées. Leur construction depuis des entiers au runtime a corrigé les tests, sans changer le helper ni la table ; Core a ensuite passé 416 cas. Le contrôle isolé V3 COMBO a d'abord échoué puis motivé la correction de projection. Le premier Host avait 224 Passed / 1 Failed sur 225 : l'attente de schéma StringCompare utilisait la forme legacy ; elle a été alignée sur l'object_info V3 collecté, puis les 225 cas ont passé. Les 68 comparaisons lowercase et 40 comparaisons des nœuds ont ensuite passé avant la campagne complète.

## Portée et provenance

Le contrat traite les chaînes composées de scalaires Unicode valides, sans NFC, casefold ni dépendance à la culture courante. La table est extraite statiquement des données C de CPython 3.12.10 et le contexte sigma est calculé sur les arguments complets ; le builtin collecté constitue une vérification distincte. Les surrogates isolés, la coercition du moteur, les diagnostics d'admission, l'annulation et la propriété des valeurs restent des contrats locaux explicitement testés.

Cet audit d'intégration vérifie les artefacts, les routes et le comparateur Lower en lecture seule. Aucun test .NET, exécution source ou calcul lowercase n'a été relancé par l'auditeur. Les pins des fichiers code/tests/ressources ont été capturés pendant la campagne à `2026-09-12T02:54:09.279272+00:00`, avant son verdict final, puis revérifiés inchangés. Les annotations documentaires après campagne sont identifiées séparément. Ces hashes ne constituent ni une attestation des DLL chargées ni une attribution globale à un commit déjà publié. Cette campagne locale ne qualifie pas d'autres OS, l'ensemble des nœuds de texte, toutes les entrées HTTP ni de nouveaux résultats numériques d'inférence.
