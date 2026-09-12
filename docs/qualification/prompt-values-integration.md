# Prompt Values : comparaison ciblée des phases

La campagne locale Windows a réussi **189 tests Host uniques, dont les 28 cas Prompt Values**, sans échec ni test ignoré. Le build Release est rapporté par l'intégrateur avec zéro avertissement et zéro erreur. Cette preuve porte sur cette suite Host ; elle ne constitue pas une nouvelle campagne de solution complète ni une validation sur trois OS.

Le laboratoire publié au commit [`8dad64f0cf9b04548ee722b7533776b25d3b4c46`](https://github.com/skyline624/ComfySharp/tree/8dad64f0cf9b04548ee722b7533776b25d3b4c46/labs/prompt-values-source) a produit la [collecte source auditée](prompt-values-source-8dad64f.md) en exécutant les fonctions de validation et d'acquisition du backend figé `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`. Cette collecte n'exécute **aucun corps de nœud source**, y compris PreviewAny. Il n'existe donc pas ici de golden de sorties de corps source à comparer aux sorties C#.

| Phase | Résultat contrôlé |
|---|---|
| Première validation, 28 cas | 26 verdicts alignés ; deux divergences explicites. Source : 16 acceptations et 12 refus. C# : 14 acceptations et 14 refus. |
| Arguments de première exécution C# | 13 cas alignés avec l'acquisition source après validation ; une divergence explicite pour une enveloppe contenant une liste en forme de lien. |
| Second passage adapté, trois cas | Deux acceptations avec arguments alignés ; un refus aligné. Les prompts sont de nouveaux objets issus du snapshot source après sa première validation. |
| Répétition du prompt C# original | Même verdict et JSON inchangé ; elle n'est pas confondue avec le second passage source sur son prompt muté. |

Les divergences sont des assertions réussies sur des différences connues, pas des tests ignorés ou des équivalences fabriquées. Les slots `false` et `-1` sont acceptés par la source de ce corpus et refusés par C#. Pour `wrapped-link-shaped`, la validation source enlève l'enveloppe puis son acquisition suit le lien ; C# conserve la liste littérale `['id', 0]` et n'exécute pas le producteur. Le test vérifie cette différence sans modifier le moteur ou les attendus.

Le comparateur utilise les vrais schémas et les vrais corps des quatre primitives et de CreateList. Une enveloppe d'observation copie immédiatement les arguments empruntés puis délègue avec le même contexte et token. Elle ne produit aucun résultat de remplacement. Les valeurs du groupe CreateList sont projetées vers les noms plats ; les arguments scalaires des primitives reçoivent seulement la dimension de liste d'exécution présente dans l'acquisition source. Les types bool/int/Float64, les bits finis Float64, les listes, les nulls et l'ordre des dictionnaires sont comparés. Seul le champ descriptif Python `repr` des flottants est exclu ; le slot original `0.0` garde son lexème.

PreviewAny sert à la validation du prompt et n'est pas exécuté par ce comparateur. Les appels réels C# ciblent le subject et ses producteurs atteints. Les corps doivent terminer, mais leur sortie n'est pas déclarée équivalente à un corps source non exécuté. Les captures d'arguments sont vérifiées par les assertions réussies ; aucun artefact distinct de captures C# n'est revendiqué. L'emplacement dans Host.Tests ne signifie pas que ces 28 tests passent par HTTP.

Les phases source de reconnaissance, d'acquisition directe, de cache et d'erreurs restent des preuves source. Le test ne crée pas de fausse API C# get_input_data et ne transforme pas les tuples d'erreur Python en diagnostics C# identiques. Les trois seconds passages adaptés conservent la distinction entre entrée originale et entrée déjà mutée par la validation source ; ils ne masquent pas la divergence du lien enveloppé.

Le [relevé machine](prompt-values-integration.json) contient les 28 IDs, les partitions par phase, les sept empreintes des fichiers pertinents compilés, les deux ressources et le TRX. L'inventaire comprend les nouveaux contrats/expansion TemplateNames et la syntaxe Prefix explicite présents au moment du build : il n'affirme pas que tout le produit est identique à un ancien commit. Les tests TemplateNames et leur future qualification source relèvent d'une preuve séparée.

| Preuve brute | Octets | SHA256 |
|---|---:|---|
| TRX Host ciblé | 419420 | `432cb0bbf4c50f3e8f37106e62a66f056a8337bb313202b78cfe833ccd77e996` |
| Fixture source, copie exacte de la collecte | 677389 | `04af1fcf3c83c91d70cfd7a1d0b97f51a6eca5bdcf9c612ac9ea29fa420bee3f` |
| Protocole, copie exacte du blob Git publié | 17911 | `dbd4038623e3313005cbb29393da978c0d8c9eb49a96860fb765ff217b6fe2ee` |
| Comparateur C# exécuté | 19286 | `b94a617966bbfeb2896bc8d89cac2cf2c72a727dbd1d033d4dfdcd1aa14ca767` |

Un audit indépendant stdlib a recompté les IDs et résultats du TRX, contrôlé les empreintes compilées, les copies brutes et le protocole publié. Il n'a relancé ni .NET ni la source. Cette comparaison bornée ne qualifie pas la validation HTTP complète, le frontend, toutes les coercitions scalaires ou la parité des sorties des nœuds.
