# CaseConverter : conversion de casse Unicode

`CaseConverter` (`Convert Text Case`, catégorie `text`) propose les quatre modes du [nœud ComfyUI figé](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_extras/nodes_string.py#L106). L'implémentation reste **partial**, limitée aux chaînes composées de scalaires Unicode valides. L'application utilise les helpers C# et les tables CPython 3.12.10 ; elle ne charge pas de runtime Python.

## Schéma et exécution

Les entrées requises sont `string` STRING multiligne, puis `mode` COMBO. La projection V3 conserve `multiselect=false` et `options` dans l'ordre **UPPERCASE, lowercase, Capitalize, Title Case**. Aucun défaut de mode n'est déclaré dans le schéma source : une sélection initiale de l'éditeur ne dispense pas un prompt HTTP de fournir cette entrée. La sortie est une STRING scalaire nommée `STRING`.

Les alias sont `case converter`, `text case`, `uppercase`, `lowercase` et `capitalize`. Les options, noms et ordre sont exposés par `/object_info/CaseConverter` ; ce nœud n'est ni un nœud de sortie ni `InputIsList`.

```json
{
  "convert": {"class_type": "CaseConverter", "inputs": {
    "string": "Bonjour", "mode": "UPPERCASE"
  }},
  "preview": {"class_type": "PreviewAny", "inputs": {"source": ["convert", 0]}}
}
```

Le compilateur sauvegarde les widgets `string`, puis `mode`. Une liaison STRING remplace le littéral du widget texte. Le mapping scalaire utilise les listes du moteur et répète le dernier élément d'une colonne plus courte. La sortie wildcard de CreateList peut alimenter le COMBO pour mapper les quatre modes ; une liaison typée STRING vers COMBO est refusée à l'admission.

Les entrées sont eager. Un mode inconnu est refusé par l'admission COMBO. Le corps source, appelé directement sans cette validation, retourne pourtant la chaîne inchangée ; le corps C# conserve ce comportement pour du texte Unicode valide. Ces deux frontières ne constituent pas une même politique d'admission HTTP. Les coercitions STRING du moteur restent celles déjà existantes.

## Opérations et domaine retenu

`UPPERCASE` applique la conversion complète en majuscules, y compris les expansions ; `lowercase` utilise la conversion complète en minuscules et le contexte du sigma final. `Capitalize` applique la casse de titre au premier scalaire puis les minuscules au reste. `Title Case` suit les frontières de caractères avec casse de CPython, sans découpage local arbitraire en mots. Le calcul des contextes se fonde sur la chaîne originale.

Le résultat peut contenir plus de scalaires que l'entrée : `ß` devient `SS` en majuscules et `İ` devient `i` suivi du point combinant en minuscules. La casse de titre d'un digraphe peut différer de sa majuscule. Ni NFC, ni casefold, ni transformation dépendant de la culture courante ne sont ajoutés. Les séquences combinantes restent telles quelles hormis les conversions de casse applicables.

Les surrogates UTF-16 isolés sont refusés dans le corps C#. Cette restriction est distincte des transformations ou refus du sérialiseur HTTP ; elle ne promet pas une parité pour toutes les chaînes Python ou les entrées mal formées. La provenance statique est décrite dans le [laboratoire des tables lower](../labs/python-unicode-tables/README.md) et celui des [tables upper/title](../labs/python-unicode-case-tables/README.md).

## Validation locale

La compilation complète Release s'est terminée sans avertissement ni erreur, puis les six suites ont passé **2 225 tests, sans échec ni test ignoré** : Core 750, Desktop 42, Host 244, Inference 577, Tokenization 540 et Workflow 72. Les 249 nouveaux cas se répartissent entre 28 tests du nœud, 68 du helper Unicode, 85 plans de digests source, 40 comparaisons du nœud à la source, quatre régressions d'ordre des arguments du moteur, 19 tests Host, trois Workflow et deux Desktop.

Les 85 plans couvrent cinq constructions sur tous les scalaires Unicode valides, avec trois répétitions source et un passage C# par plan. Ils ne démontrent pas tous les contextes de chaînes possibles. Les 40 comparaisons conservent trois frontières distinctes : 28 cas comparables, huit erreurs sur des arguments directement typés séparées des coercitions STRING du moteur, et quatre modes inconnus retournant l'identité au niveau du corps alors que l'admission COMBO les refuse.

Le premier passage des 125 comparaisons source avait produit 124 succès et un échec sur l'ordre des arguments `mode`, puis `string`. Le moteur a été corrigé ; les fixtures et les attentes source sont restées inchangées. Les 125 comparaisons passent dans la campagne complète après cette correction.

Les 19 tests Host exercent le schéma V3, douze parcours compilés, la priorité de la connexion STRING, le mapping CreateList et quatre refus. Le compilateur, les nœuds enregistrés et PreviewAny sont réellement utilisés. Le smoke réel après correction du moteur a également réussi pour les quatre modes.

Les résultats et leurs limites sont détaillés dans la [preuve d'intégration](qualification/case-converter-integration.md). Le manifeste conserve l'implémentation `partial`, avec `unitTests=true`, `realWorkflow=false`, les six plateformes fausses et aucune preuve de plateforme. Cette campagne locale ne qualifie ni toutes les chaînes Python, ni un workflow complet, ni une matrice de plateformes.
