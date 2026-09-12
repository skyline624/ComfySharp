# Autogrow.TemplateNames : contrat moteur borné

`AutogrowNamesTemplate` permet à un nœud C# de déclarer un groupe ordonné de noms explicites. Le moteur valide les entrées API plates, découpe les listes d'exécution, puis remet au corps du nœud un `RuntimeValue` de type Map. Ce lot ajoute un contrat réutilisable ; il n'enregistre aucun nœud et n'ajoute aucun contrôle Desktop. StringFormat et son langage Python de formatage restent hors périmètre.

Le comportement structurel est tiré du backend figé [`1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_api/latest/_io.py#L1087), notamment `_AutogrowTemplate`, `TemplateNames`, l'expansion Autogrow et [`build_nested_inputs`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_api/latest/_io.py#L1935). SHA-256 du blob `_io.py` : `55391c45bee9cfd821c7c86a23ab0f9e12ce9b5b40d77d48308c1d3698ed4080`. Aux tests de comportement initiaux s'ajoutent les [comparaisons avec 25 cas source prospectifs](qualification/autogrow-names-integration.md), dont une frontière d'erreur explicitement distinguée des retours. Aucune parité générale de nœud, du frontend ou de l'API HTTP n'est annoncée.

## Déclaration et instantanés

```csharp
var template = new AutogrowNamesTemplate(
    new InputSchema("value", "*"), ["z", "a", "b"], min: 1);
var group = new InputSchema("values", "COMFY_AUTOGROW_V3", Autogrow: template);
```

Les entrées plates sont `values.z`, `values.a`, `values.b`. `values.z` est requise ; les deux suivantes sont optionnelles. Un prompt contenant `values.b` puis `values.z` produit une Map dont les membres sont `z`, puis `b`. Le nom absent `a` n'est pas ajouté. Un membre présent avec la valeur JSON null reste présent.

La liste est copiée et tronquée aux 100 premiers éléments **avant** la validation des noms effectifs. L'ordre est préservé ; aucun tri, changement de casse ou dédoublonnage n'est effectué. Une liste vide est valide. `min` doit être positif ou nul et peut dépasser le nombre de noms ; tous les noms disponibles sont alors requis si le prototype est requis. Un prototype optionnel rend toutes les feuilles optionnelles, même si `min` est supérieur au nombre de noms. La propriété `Min` garde la valeur déclarée.

`Names` et `MemberNames` sont des vues en lecture seule sur l'instantané détenu par le template. Les options du prototype sont copiées au constructeur, puis à chaque lecture de `Input`. L'expansion et object_info produisent également leurs propres copies des options. Modifier la liste ou le JSON d'origine, ou une projection déjà reçue, ne modifie pas le template.

Le champ `InputSchema.Autogrow` a maintenant pour type `AutogrowTemplate`, base abstraite dont les deux implémentations disponibles sont les classes scellées `AutogrowPrefixTemplate` et `AutogrowNamesTemplate`. Les constructeurs explicites Prefix gardent leur signature et leur contrat. Les appels qui utilisaient `Autogrow: new(...)` doivent désormais préciser `new AutogrowPrefixTemplate(...)`. Cette évolution d'API 0.x exige une recompilation des consommateurs ; aucune compatibilité binaire avec l'ancien accesseur ou constructeur `InputSchema` n'est promise. Les preuves historiques et leurs empreintes restent attachées aux fichiers qu'elles ont réellement vérifiés.

## Limites explicites

Les deux templates acceptent seulement un prototype AnyType (`*`) ou MatchType avec un template explicite `{ "template_id": "…", "allowed_types": "*" }`. Les options de présentation `display_name`, `tooltip` et `advanced` restent permises et typées. Les options sémantiques non implémentées, les widgets, les prototypes dynamiques imbriqués et les prototypes lazy sont refusés.

Les groupes doivent être requis, non lazy et au premier niveau, avec un nom non vide sans point. Mélanger un groupe Autogrow avec une autre entrée lazy est également refusé. Les noms de membres effectifs doivent être non vides et composés uniquement de lettres ASCII, chiffres ASCII ou `_`, avec unicité ordinale. Ainsi `a` et `A` sont distincts, mais deux occurrences de `a`, `a.b` ou une chaîne vide déclenchent une `ArgumentException` explicite au constructeur. Les éléments après le centième sont ignorés, y compris s'ils seraient invalides dans la tranche effective. Ces restrictions de noms sont locales : le constructeur source n'effectue pas cette validation et ses cas ambigus ne sont pas présentés comme équivalents.

À l'expansion, un groupe ou une option non pris en charge reçoit `unsupported_dynamic_template`. Une collision de feuilles ou de groupes reçoit `invalid_dynamic_template`. Tout nom plat supplémentaire, alias ou nom de groupe utilisé comme entrée reçoit `unsupported_dynamic_input` avant l'exécution des producteurs. Les extras source ne sont donc pas silencieusement acceptés ou qualifiés. La validation des feuilles requises demeure indépendante : une connexion à `values.b` ne satisfait pas l'absence de `values.z` si cette dernière est requise.

## Exécution et propriété

Pour `InputIsList=false`, le moteur applique son mapping existant avant le regroupement. Chaque membre est la valeur de la ligne courante ; les listes plus courtes répètent leur dernière valeur. Une liste d'exécution vide combinée à une liste non vide garde l'erreur existante. Quand toutes les listes reçues sont vides, l'invocation unique ne reçoit pas de valeur pour ces chemins ; les chemins connus sont regroupés avec null. Sans chemin présent, y compris avec zéro nom déclaré, le groupe devient une Map vide.

Pour `InputIsList=true`, le corps est appelé une fois et chaque membre contient sa liste d'exécution entière. Une liste JSON littérale à l'intérieur d'une liste d'exécution reste un seul élément. Le contrat de liens et de l'enveloppe `__value__` n'est pas modifié par ce lot.

Le choix du premier blocker examine toujours les entrées **plates dans l'ordre du prompt**, avant la création des Maps. Un blocker direct peut empêcher l'appel même si le corps n'aurait pas utilisé ce membre. Les blockers contenus dans une Map ou une valeur de liste imbriquée ne sont pas recherchés récursivement. Pour les nœuds Autogrow, le moteur présente ensuite les arguments plats au binder dans l'ordre du prompt : les arguments ordinaires conservent cet ordre avant les groupes ajoutés. Les groupes suivent l'ordre de déclaration et leurs membres l'ordre du template. Cette présentation ne change pas l'ordre de résolution des producteurs.

Les valeurs d'entrée et les enfants des Maps sont empruntés. `BindArguments` alloue ses Maps dans le `RuntimeNodeContext` de l'invocation et retient leurs enfants. Les sorties passent par la rétention normale du moteur. Pour conserver une valeur après l'invocation, utiliser `Retain()` et libérer cette nouvelle référence. Les chemins de défaillance partielle et d'annulation doivent libérer les acquisitions intermédiaires sans voler les références de l'appelant.

## Projection et vérifications

object_info décrit le template original, pas la liste des ports finalisés. Le template Names contient exactement les propriétés `input`, `names`, `min`, dans cet ordre ; aucun `max` dérivé ni `prefix` n'est ajouté. Le format Prefix conserve `input`, `prefix`, `min`, `max`.

`AutogrowNamesTests` couvre les instantanés, les bornes et restrictions, les partitions requises/optionnelles, object_info, les collisions, les noms supplémentaires, les omissions et nulls, le mapping scalaire avec repeat-last et arguments statiques, les listes entières, les listes vides, les blockers, le fan-out, le dernier propriétaire, l'échec partiel du binder et l'annulation. Les nœuds de test sont des sondes de contrat explicitement non enregistrées dans le catalogue produit. L'état d'exécution de ces tests relève du rapport de validation de la campagne qui les lance ; ce document ne remplace pas ce résultat.

`AutogrowNamesSourceReferenceTests` conserve les 36 observations source immuables et compare les 25 cas délimités avant collecte. Les métadonnées, partitions, ordres et arguments transportés sont comparés exactement pour les cas retournés ; le cas d'erreur compare la frontière d'échec et le diagnostic local séparément de l'exception Python. La sonde transporte les arguments et ne porte aucun algorithme de nœud source. `AutogrowArgumentOrderTests` vérifie aussi les deux templates et les deux modes de listes, avec résolution des producteurs inchangée.
