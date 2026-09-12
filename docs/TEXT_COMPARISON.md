# Comparaison de texte : StringContains et StringCompare

`StringContains` (`Contains Text`) et `StringCompare` (`Compare Text`), catégorie `text`, utilisent les vrais nœuds C# et le moteur existant. Leur statut reste **partial** : le domaine retenu est le texte formé de scalaires Unicode valides, avec une conversion en minuscules fondée sur les tables CPython 3.12.10. Aucun runtime Python n'est chargé par l'application.

Les schémas et les opérations proviennent de [StringContains](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_extras/nodes_string.py#L197) et [StringCompare](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_extras/nodes_string.py#L225) dans le backend figé.

## Schémas et prompts

| Nœud | Entrées requises, dans l'ordre | Sortie |
|---|---|---|
| StringContains | `string` STRING, `substring` STRING, `case_sensitive` BOOLEAN | BOOLEAN, nom `contains` |
| StringCompare | `string_a` STRING, `string_b` STRING, `mode` COMBO, `case_sensitive` BOOLEAN | BOOLEAN, nom `BOOLEAN` |

Les widgets STRING sont multilignes. `case_sensitive` porte les options `default=true` et `advanced=true`. Ce défaut initialise le widget ; il ne remplace pas une entrée requise absente d'un prompt HTTP. La projection V3 de `mode` conserve le type `COMBO`, `multiselect=false` et le tableau `options` dans l'ordre `Starts With`, `Ends With`, `Equal`. Les choix sont sensibles à la casse. La projection historique des anciens nœuds est conservée.

Les projections `/object_info` conservent les noms V3, les alias de recherche et le module `comfy_extras.nodes_string`. Les sorties sont scalaires, les entrées ne sont pas `InputIsList` et ces nœuds ne sont pas des nœuds de sortie.

```json
{
  "compare": {"class_type": "StringCompare", "inputs": {
    "string_a": "Bonjour", "string_b": "bon", "mode": "Starts With", "case_sensitive": false
  }},
  "preview": {"class_type": "PreviewAny", "inputs": {"source": ["compare", 0]}}
}
```

Le compilateur sauvegarde les widgets dans cet ordre. Une connexion BOOLEAN sur `case_sensitive` remplace son littéral sauvegardé. Le moteur garde ses règles d'admission et de coercition existantes ; cette tranche ne définit pas une nouvelle conversion générale des valeurs JSON en texte ou booléen.

## Opérations et limites Unicode

Avec `case_sensitive=true`, la recherche, le préfixe, le suffixe et l'égalité utilisent une comparaison ordinale. Avec `false`, chaque chaîne complète est d'abord convertie séparément en minuscules, puis la même opération ordinale est appliquée. La conversion utilise les tables Unicode 15 de CPython 3.12.10 et la règle contextuelle du sigma final, décrites dans le [laboratoire des tables](../labs/python-unicode-tables/README.md). Elle ne dépend pas de la culture du processus ni des tables Unicode du système d'exploitation.

Cette opération n'est ni une normalisation Unicode ni un casefold : `İ` devient `i` suivi du point combinant, `Straße` n'est pas égal à `STRASSE`, et `é` n'est pas égal à `e` suivi de l'accent combinant. Le contexte est celui de chaque chaîne entière : la minuscule de `ΟΣ` se termine par `ς`, tandis que celle de `Σ` seul est `σ`. Leur comparaison de suffixe insensible à la casse est donc fausse.

Les paires UTF-16 valides, y compris les caractères supplémentaires, sont admises. Les surrogates isolés sont refusés par le corps C#, même en comparaison sensible à la casse. Ce domaine est plus étroit que toutes les chaînes représentables en Python ; il ne constitue pas une promesse de traitement identique de toute entrée HTTP mal formée ou transformée par le sérialiseur.

Les entrées restent eager. Le mapping existant produit un appel par ligne et répète le dernier élément d'une colonne plus courte. Deux sorties CreateList peuvent ainsi fournir les deux textes ; sa sortie wildcard peut aussi alimenter le COMBO pour mapper les modes. Une liaison typée STRING vers COMBO n'est pas admise par le moteur. Les résultats booléens peuvent être affichés par le vrai PreviewAny.

## État de validation

La [preuve d'intégration locale](qualification/text-comparison-integration.md) consigne **1976 tests uniques PASS** sous Windows, sans échec ni test ignoré, après un build Release sans avertissement ni erreur. Les 224 nouveaux cas comprennent 35 tests Core des nœuds, 48 tests de l'aide Unicode, 68 plans de digests source, 40 cas de comparaison des nœuds, 24 Host, six Workflow, deux Desktop et une régression de projection V3 COMBO. Les campagnes ciblées et leurs répétitions ne s'ajoutent pas au total unique de la suite.

Les 40 cas ne sont pas 40 succès de prédicat identiques. Les 36 comparables comprennent 39 booléens exacts après mapping et un blocker sans appel de corps. Deux erreurs source sont comparées à un appel direct C# avec les mêmes valeurs typées ; le même prompt soumis au moteur C# suit séparément ses coercitions STRING observées. Deux modes invalides ont produit zéro sortie côté source, tandis que l'admission COMBO locale les refuse avant le corps. Ces différences de frontières ne sont pas masquées par une normalisation des attendus.

Les 68 plans de digests source couvrent les 17 plans Unicode dans quatre contextes chacun, avec trois répétitions source et un passage C# par plan dans la campagne complète. Ils ne prouvent pas à eux seuls toutes les chaînes et tous les contextes multi-caractères possibles. Les propriétés contextuelles ciblées complètent ce contrôle sans élargir la portée aux surrogates isolés ou à la normalisation Unicode. Les attendus proviennent du laboratoire publié avant sa collecte, puis comparé au C# ; ils n'ont pas été calculés par le produit.

Les 24 cas Host couvrent les deux schémas, la compilation suivie de POST `/prompt` et de PreviewAny, les trois modes, les propriétés Unicode ci-dessus, les connexions BOOLEAN, deux mappings CreateList et les refus d'admission. Ils utilisent les nœuds réellement enregistrés. Le smoke Desktop et son Host ont également compilé et exécuté StringContains et StringCompare, avec aperçus `True` et `False`.

Le manifeste porte `unitTests=true` pour les deux implémentations `partial`. `realWorkflow` et les six indicateurs de plateforme restent faux, avec `evidence` vide. Ces parcours ciblés ne qualifient ni tous les workflows source ni une plateforme entière. Les pins et résultats détaillés figurent dans la preuve d'intégration ; les annotations documentaires ont été mises à jour après la campagne.
