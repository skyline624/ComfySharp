# CreateList et entrées Autogrow préfixées

Cette tranche ajoute le nœud local `CreateList` du backend ComfyUI
`1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`. Il concatène des listes d'exécution
sans modèle ni calcul de tenseur. Son port reste partiel : les inputs supplémentaires
sont rejetés plus tôt que dans certains chemins Python, et l'éditeur utilise dix
ports fixes sans propagation complète du type MatchType. Aucune compatibilité de
famille de modèles ou de frontend complet n'est établie.

Le [source du nœud](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_extras/nodes_toolkit.py#L5)
déclare un input Autogrow `inputs`, un prototype MatchType `input` de template
`type`, et une sortie liste nommée `list`. Les bornes du nœud sont exactement
minimum 1, maximum 10. And/Or ne font pas partie de cette tranche ; leur
truthiness Python exige un contrat séparé.

## Du contrat public à l'exécution

`/object_info/CreateList` expose le template original `COMFY_AUTOGROW_V3`, avec
le prototype `COMFY_MATCHTYPE_V3`, allowed_types `*`, prefix `input`, min 1 et
max 10. Les champs `is_input_list` et `output_is_list[0]` valent true ;
`output_matchtypes[0]` vaut `type`. Cette projection V3 explicite conserve les
métadonnées du nouveau nœud et ne modifie pas la forme des 25 anciennes entrées.

Le prompt API transmet des noms **plats**. Par exemple :

```json
{
  "a": {"class_type": "PrimitiveString", "inputs": {"value": "alpha"}},
  "b": {"class_type": "PrimitiveString", "inputs": {"value": "🌍"}},
  "list": {
    "class_type": "CreateList",
    "inputs": {"inputs.input0": ["a", 0], "inputs.input2": ["b", 0]}
  },
  "length": {"class_type": "StringLength", "inputs": {"string": ["list", 0]}}
}
```

`inputs.input0` est obligatoire. Les ports `inputs.input1` à `inputs.input9`
sont optionnels ; les trous sont autorisés. Un port optionnel connecté ne remplace
pas le port zéro. L'API directe Engine permet de sélectionner explicitement une
cible scalaire telle que `length`. Le Host HTTP exige un nœud de sortie : relier
`length` à PreviewAny et sélectionner ce PreviewAny dans `partial_execution_targets`.

Le moteur valide et résout les feuilles plates, conserve les listes d'exécution,
inspecte les blockers, puis regroupe les feuilles dans une Map `inputs` avant
d'appeler le nœud. La concaténation suit l'ordre du template : input0 avant input2,
même si l'ordre des propriétés du prompt est inversé. Le choix du premier blocker
conserve au contraire l'ordre du prompt, conformément au chemin d'exécution source.
Le regroupement ne cache donc pas un blocker dans une Map avant son contrôle.

Deux listes de longueurs 2 et 1 produisent trois items, sans zip, répétition du
dernier élément ou dédoublonnage. Une liste d'exécution vide apporte zéro item.
Un tableau JSON littéral reste un item entier : il n'est pas aplati récursivement.
Comme pour les autres nœuds du Host, un tableau littéral du prompt doit être
enveloppé dans `{"__value__": [...]}` pour ne pas être interprété comme un lien.

## Périmètre du template géré

`AutogrowPrefixTemplate` capture son prototype et ses options par copie. Le
template exposé est préfixé, de premier niveau, avec prototype AnyType (`*`) ou
MatchType wildcard explicite. Il accepte les bornes source min >= 0 et
1 <= max <= 100. La source autorise min > max : dans ce cas tous les noms générés
sont requis. Un prototype optionnel rend ses feuilles optionnelles quel que soit
min ; sans feuille vivante, le regroupement produit une Map vide.

Cette API ne prend pas en charge les prototypes widgets, templates dynamiques
imbriqués, TemplateNames, DynamicCombo/Slot, groupes externes optionnels,
rawLink, ni le mélange d'Autogrow avec une quelconque entrée lazy. Ces cas sont
refusés explicitement. Les préfixes sont limités aux lettres/chiffres ASCII et
underscore ; les noms de groupes ne sont pas des chemins imbriqués. Les options
acceptées sont les textes `display_name`/`tooltip`, le booléen `advanced`, et le
template MatchType wildcard. Les autres options ne sont pas ignorées.

Pour un nœud Autogrow, tout nom d'input hors du contrat développé produit le
diagnostic anticipé `unsupported_dynamic_input`, avant l'exécution des producteurs.
Cela couvre notamment `inputs.input10`, `inputs.input01` et un input directement
nommé `inputs`. Python peut ignorer certains littéraux supplémentaires ou échouer
plus tard pour un lien supplémentaire ; ce rejet strict est une limite documentée
du port, pas une affirmation de parité sur ces prompts hors contrat.

Les valeurs opaques peuvent être transportées et retenues comme items, sans être
inspectées ni projetées en JSON. Le nœud n'applique aucune règle de truthiness ou
de conversion numérique. Les allocations de conteneurs appartiennent au contexte
d'invocation ; une erreur ou annulation libère les leases temporaires sans voler
ceux du caller. Les outputs restent détenus jusqu'au dernier propriétaire.

## Éditeur et workflows

Le Desktop présente dix ports wildcard fixes lors de l'ajout de CreateList, sans
widget persistant ; le premier port doit être connecté. Les commandes existantes
Connect/Disconnect permettent de relier plusieurs producteurs. Il n'y a ni
croissance automatique de ports ni propagation commune du type MatchType.

L'import conserve l'ordre, les indices, métadonnées et absence éventuelle de
widgets des documents existants. Le compilateur émet les propriétés plates dans
l'ordre des slots importés, refuse les noms non canoniques ou dupliqués, l'absence
du premier lien et les liens/target_slot incohérents. Il ne réécrit pas silencieusement
un document importé pour lui ajouter dix ports.

## Vérification et provenance

Les tests gérés couvrent la projection, les copies d'options, les bornes et refus,
la concaténation et le mapping réel, l'ordre des blockers, le transport de
conteneurs, le fan-out, les leases avec ressources factices, l'annulation et les
diagnostics HTTP. Ils ne chargent pas de modèle et ne constituent pas un oracle
source généré par le produit.

La [campagne locale initiale](qualification/createlist-local.md) passe 1499 tests
avant la collecte indépendante. Le [laboratoire source](qualification/autogrow-source-ae241a0.md)
utilise ensuite les constructeurs et fonctions AST exacts et conserve 36 cas.
Les [18 comparaisons du port](qualification/autogrow-integration.md) passent avec
les 188 tests Core : 12 cas CreateList et six templates génériques. Les schémas,
l'ordre, le regroupement, les appels réels, les sorties et les champs communs des
blockers correspondent exactement. Les autres observations restent conservées
sans être transformées en attentes de succès C#.

Le cas source du tableau littéral vide observe `get_input_data`, avant la
validation complète du prompt. Cette validation amont reconnaît aussi
`__value__` ; la comparaison utilise cette enveloppe pour obtenir la même valeur
d'exécution. Elle ne prétend pas établir l'acceptation HTTP d'un tableau brut.
Ces preuves ne qualifient pas la totalité de V3, le frontend complet, les
plateformes matérielles ni les familles de modèles.
