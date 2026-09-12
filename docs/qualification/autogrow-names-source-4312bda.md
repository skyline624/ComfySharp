# Autogrow.TemplateNames — collecte source 4312bda

Le laboratoire et son protocole ont été [publiés au commit 4312bdafabe2906b74bc51cdca678aa26aeec5ee](https://github.com/skyline624/ComfySharp/tree/4312bdafabe2906b74bc51cdca678aa26aeec5ee/labs/autogrow-names-source) avant la première exécution source effectuée par root. La collecte sous Python 3.12.10, Windows, contient **36 cas : 33 retours et trois erreurs prescrites**. Le fichier brut mesure 296 326 octets, SHA-256 `f3f6772bdf05aa464cc831edb8e929ff179f73df3d01ea2f4546616eafcbe4e3`.

Cette preuve est un **audit des artefacts par l’auteur du collecteur**, pas une revue par une autre personne. Elle ne qualifie ni les résultats du port C# Names ni un nœud StringFormat C#. Aucun calcul source supplémentaire, import du collecteur, test .NET ou calcul natif n’a été exécuté pendant l’audit.

## Provenance et observations vérifiées

Les 610 contrôles de lecture vérifient le protocole prospectif, les trois fichiers du laboratoire tels que publiés, les trois fichiers du helper Autogrow antérieur au commit `ae241a0`, ainsi que les **dix blobs Git et 73 déclarations AST** du backend figé `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`. Chaque déclaration est contrôlée par sa ligne, son AST et son segment source ; chaque blob par sa taille et son SHA-256. Les identités détaillées figurent dans la [preuve JSON](autogrow-names-source-4312bda.json).

La collecte utilise les vrais TemplateNames, Schema, MatchType, ComfyNode et NodeOutput, avec finalisation, acquisition des entrées, mapper, binder et fusion des résultats issus des AST source inchangés. Le helper publié conserve sa fermeture source antérieure ; la déclaration supplémentaire est le vrai [StringFormat](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_extras/nodes_string.py#L9). Aucun faux constructeur Schema ou Tensor ne remplace ces classes.

| Observation | Nombre vérifié |
|---|---:|
| Cas et empreintes centrales recalculées | 36 |
| Empreintes off/on/off enregistrées | 108 |
| Métadonnées/finalisations obtenues | 34 |
| Retours observés de `build_nested_inputs` | 33 |
| Appels du corps-écho de fixture | 30 |
| Appels du vrai StringFormat | 3 |
| Rapports du callback de blocage source | 3 |

Chaque résultat central, hors champs d’observation et de répétition, a été réencodé exactement et rehashé. Son empreinte correspond aux trois empreintes enregistrées ; les snapshots des entrées et caches restent identiques au protocole pour les 36 cas. Les résultats bruts des deux passages sans observation ne sont pas conservés : leur égalité provient du contrôle effectué par le collecteur, tandis que l’audit recalcule le passage central conservé.

Les événements `sys.setprofile` observent les vrais retours du binder et les arguments du corps. Les ordres explicites des partitions, chemins, entrées plates et dictionnaires regroupés ont été vérifiés. Pour le corps-écho, arguments et retour du binder sont identiques, ordre compris. Pour StringFormat, les arguments nommés proviennent des variables locales du frame et sont enregistrés dans l’ordre de la signature ; l’ordre réel du dictionnaire construit par le binder demeure enregistré séparément. Le champ `python_module` distingue la fixture `laboratory.autogrow_names` de la vraie classe `comfy_extras.nodes_string` ; il ne remplace pas le champ module initial du schéma.

## Résultats structurels et limites de comparaison

La partition prospective reste inchangée : **25 comparables, six unsupported-template, un constructor-error, deux source-only et deux unsupported-input**. « Comparable » délimite les futures comparaisons structurelles ; ce mot ne constitue pas un résultat de test C#.

Les noms explicites conservent leur ordre, distinct de celui du prompt. Les listes de tailles différentes sont mappées avec répétition du dernier élément ; InputIsList reçoit les listes entières. Deux groupes et des entrées ordinaires rendent visibles les ordres des arguments racines et des membres. La liste vide de noms est acceptée, y compris avec minimum 1, et un minimum supérieur au nombre de noms reste conservé. La troncature aux 100 premiers noms précède l’examen des noms effectifs : le 101e nom ambigu n’apparaît pas dans la structure. La mutation de la liste privée fournie au constructeur ne modifie pas la copie source. Aucune clé `max` ou `prefix` n’est ajoutée aux métadonnées Names.

Les trois erreurs correspondent exactement aux cas admis avant calcul : `IndexError` du mapper pour une liste d’exécution vide en mapping ordinaire, puis `AssertionError` au constructeur pour un prototype dynamique et pour un minimum négatif. Dans le premier cas, aucun callback de blocage ni corps n’est appelé : le slicing échoue avant la recherche du blocker. Le blocker silencieux rencontré en premier supprime le message suivant ; les messages vide et non vide restent distingués.

Les observations hors périmètre sont conservées : un doublon peut apparaître dans les partitions required et optional ; les points produisent une structure imbriquée ; un nom vide devient une clé vide. Un littéral inconnu est ignoré à l’acquisition, tandis qu’une connexion inconnue peut parvenir au corps-écho comme argument racine non regroupé. Ce corps accepte délibérément `**kwargs` pour exposer les arguments : son succès ne prouve pas qu’une méthode de nœud à signature stricte accepterait cet argument.

Pour 34 cas, `FixtureNames.execute(**kwargs)` retourne simplement le vrai `NodeOutput(kwargs)`. Ses sorties sont un transport transparent des arguments, **pas un algorithme de nœud source**. Les deux cas StringFormat exécutent réellement le formateur source et produisent respectivement `Z1|A`, `Z2|A` et `no fields` ; ces sorties restent source-only et ne qualifient aucun port du langage de formatage.

Enfin, la frontière est celle de la finalisation et de `get_input_data`, pas celle de PromptExecutor ou de la validation HTTP complète. L’absence d’une feuille required, le traitement d’un littéral ou le wrapper `__value__` ne doivent donc pas être extrapolés en contrat HTTP à partir de cette collecte. Lazy, interruption, frontend, propriétés de durée de vie et parité complète des nœuds demeurent hors de cette preuve.
