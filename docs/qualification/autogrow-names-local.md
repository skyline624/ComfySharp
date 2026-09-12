# TemplateNames : validation locale du contrat moteur

La campagne locale Windows Release a réussi **1614 tests uniques dans six projets**, sans échec, test ignoré ou doublon d'ID. L'intégrateur rapporte un build terminé avec zéro avertissement et zéro erreur. Le recomptage indépendant des six TRX concorde avec son audit.

| Projet | Tests réussis |
|---|---:|
| Core | 224 |
| Desktop | 35 |
| Host | 189 |
| Inference | 577 |
| Tokenization | 540 |
| Workflow | 49 |
| **Total unique** | **1614** |

Ce total inclut **36 nouveaux cas AutogrowNames**, **28 cas Prompt Values** et les **18 références Autogrow Prefix existantes**. Ces sous-groupes ne sont pas ajoutés une seconde fois au total. La [preuve Prompt Values ciblée](prompt-values-integration.md) décrit séparément ses phases, ses différences connues et son premier TRX Host ; la suite complète présente réexécute ces mêmes 28 cas.

Les 36 tests Names exercent les instantanés des noms/options, la troncature aux 100 premiers noms, l'ordre ordinal, les noms vides ou invalides, le minimum, les prototypes optionnels, object_info, les collisions, les extras, la distinction omission/null, les deux modes de mapping, repeat-last, les blockers et la propriété des ressources. Les cas de fan-out, dernière rétention, échec partiel et annulation passent par le vrai moteur et son binder. Les nœuds auxiliaires sont des sondes de contrat explicites, pas des implémentations source ajoutées au registre produit.

**Aucune collecte source TemplateNames n'est encore qualifiée dans cette preuve.** Ces tests ne sont donc pas des comparaisons à un oracle Names. Le comportement structurel relu provient du backend figé `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a` ; les restrictions locales et limites sont précisées dans le [contrat TemplateNames](../AUTOGROW_NAMES.md). Les 18 anciens cas Prefix ont, eux, réussi contre leur fixture source inchangée, SHA256 `04f2f2d1cea2a66460a832d471f751fcf3effd45ad4bc36a6508d540a02d2924`.

Le catalogue garde ses **26 nœuds implémentés ou partiels** et son manifeste inchangé. Ce nombre ne désigne pas la taille de l'inventaire upstream. Aucun nœud StringFormat, contrôle Desktop dynamique ou support complet du frontend n'est ajouté par ce contrat. Les groupes restent requis, au premier niveau, sans mélange lazy ; les prototypes sont limités à AnyType ou MatchType wildcard. Les noms effectifs doivent être simples et uniques, restriction locale distincte du constructeur upstream.

La propriété `InputSchema.Autogrow` utilise désormais la base `AutogrowTemplate`. Les consommateurs 0.x doivent être recompilés et les constructeurs Prefix auparavant inférés doivent être nommés explicitement. Aucune compatibilité binaire avec l'ancienne signature n'est promise. La projection Prefix, le binder partagé et le contrôle des blockers avant regroupement sont conservés.

Le code testé est l'état de travail **non committé identifié par ses empreintes**, comprenant les changements Names et les appels Prefix explicites. Il n'est pas attribué à un ancien commit prétendument inchangé. Le [relevé machine](autogrow-names-local.json) conserve les sept pins du gel Names, les quatre pins du comparateur Prompt Values et de ses ressources, les SHA/octet des six TRX et les noms exacts des 82 cas des trois sous-groupes.

| TRX | SHA256 |
|---|---|
| Core | `2227065755f6be4050d17905951e396c857e3f0e4c7869c5705a2751b75f3d0f` |
| Desktop | `2d1f519b69dde025b783b5f6dabc0543f1170f1d15aba8a213edb202cbd4f79b` |
| Host | `a4f432c187dc99876e6f9e4445afb6d679b8c5bea85b1dff92bf5546b39086e6` |
| Inference | `6f4f6ec09c40a311273d1626ea274c5005f034e8ee042b2dbf16fdf8ad2abd99` |
| Tokenization | `f27d25d9b379bfaabb0104d5d5afac5bec93eab832ef386409da1e6282c702f7` |
| Workflow | `a8d7fe144092f3d3ab7c1ec7dc3a6d0106431c5ab76f0f6885352725d6451e91` |

La prochaine CI reste **en attente** à la date de cette preuve. Son smoke Desktop existant a seulement été déplacé, avec son script inchangé, après le catalogue et avant les barrières Inference ; il utilise un processus distinct des références de première utilisation native. Aucun nouveau résultat de smoke graphique ou de CI trois OS n'est déduit des six TRX locaux. Cette campagne ne clôt pas les anciens écarts numériques interplateformes ni les qualifications de familles de modèles.

L'audit de cette preuve a uniquement lu les fichiers et recompté les résultats avec la bibliothèque standard Python ; il n'a relancé ni .NET, ni natif, ni déclarations source.
