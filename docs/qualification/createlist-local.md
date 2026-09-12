# CreateList — validation locale bornée

La campagne locale Windows réussit **1499 tests sur 1499**, sans échec ni test ignoré. Elle porte sur les octets des **16 fichiers** identifiés dans [la preuve JSON](createlist-local.json), encore non committés lors de l’audit. Le commit de base n’est pas présenté comme le commit testé. Les SHA bruts concordent avec les octets LF canoniques vérifiés par Git pour ces 16 fichiers.

| Projet | Réussis | Nouveaux cas |
|---|---:|---:|
| Core | 170 | 34 |
| Workflow | 31 | 14 |
| Desktop | 35 | 3 |
| Host | 146 | 4 |
| Tokenization | 540 | 0 |
| Inference | 577 | 0 |
| **Total** | **1499** | **55** |

Les six TRX ont été relus indépendamment avec la bibliothèque standard Python : compteurs, résultats individuels, identités d’exécution et classes des nouveaux tests concordent. Le coordinateur rapporte une compilation sans avertissement ni erreur. L’audit n’a lancé aucun build, test .NET ou calcul natif.

La première campagne avait 1494 réussites et cinq échecs. Trois tests Host sélectionnaient CreateList directement : le Host exigeant un nœud de sortie, ils recevaient `prompt_no_outputs` avant le diagnostic recherché. Ils ciblent désormais un vrai `PreviewAny` relié à CreateList, tout en conservant les assertions HTTP 400, `unsupported_dynamic_input` et file vide. Deux tests existants attendaient encore 13 et 25 inscriptions ; ces attentes ont été corrigées à 14 et 26. Ces ajustements concernent les tests et une précision documentaire, sans correction produit entre les campagnes. La première campagne ne s’ajoute pas au total.

Les deux captures réelles du Host confirment **25 anciennes entrées `/object_info` identiques**, valeurs et ordre des propriétés compris, avec exactement `CreateList` ajouté. Les hashes des captures et les inventaires d’assemblies sont conservés dans le JSON. Le coordinateur rapporte aussi le passage de `Catalog --node-evidence` : 1171 lignes, 664 déclarations locales de nœuds inventoriées, 26 nœuds inscrits, zéro erreur ; `catalogueComplete` et `releaseReady` restent faux.

Le périmètre reste **partiel** : `TemplatePrefix` de premier niveau, prototypes AnyType ou MatchType wildcard, concaténation d’un niveau de listes d’exécution, conservation des valeurs et des blockers, et dix ports fixes dans l’éditeur. Les options non qualifiées, les imbrications dynamiques et le mélange Autogrow/lazy sont refusés. Les noms dynamiques supplémentaires sont rejetés en amont ; cette règle n’est pas annoncée équivalente au comportement source pour les extras. La croissance automatique et la propagation commune de MatchType dans l’interface ne sont pas implémentées. Les tableaux JSON littéraux utilisent l’enveloppe locale `__value__` : leur représentation brute HTTP n’est pas qualifiée équivalente à la source.

Les déclarations amont sont épinglées au commit `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`, avec leurs SHA dans le JSON. **Cette campagne locale précède la collecte indépendante et n’inclut aucune comparaison à la source.** Son verdict ne qualifie donc ni parité Python, ni qualification multiplateforme, ni compatibilité de famille. Aucune fixture de sortie n’a été produite depuis C# pour servir d’oracle.

Une précision documentaire postérieure à la campagne décrit cette chronologie dans le contrat CreateList. Son empreinte a été actualisée dans le JSON ; les 15 fichiers produit et tests restent identiques à ceux audités.

Voir [le contrat et les limites CreateList](../CREATE_LIST.md).
