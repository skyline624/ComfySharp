# Réconciliation des inscriptions et contrats de nœuds

L'analyse porte sur les sources publiques ComfyUI au commit [`1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`](https://github.com/comfy-org/ComfyUI/tree/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a). Elle produit des candidats et des contrats de source. Aucun import ComfyUI, modèle, téléchargement ou calcul d'inférence n'a été exécuté. `catalogueComplete`, `registrationComplete` et `schemaComplete` restent `false`.

## Résultat et fichiers

| Mesure | Nombre | Interprétation |
| --- | ---: | --- |
| Modules du chargeur intégré | 178 | `nodes.py`, 136 extras ordonnés, 41 modules API triés |
| Candidats locaux intégrés | 652 | Identifiants résolus depuis les inscriptions, sous réserve d'import et validation |
| Candidats API distants | 275 | Périmètre exclu conservé explicitement |
| Déclarations locales absentes des listes d'inscription | 10 | Classes de base, exemples ou classes non exposées |
| Déclarations distantes absentes des listes d'inscription | 2 | Conservées avec leur statut de déclaration |
| Référence custom livrée avec l'amont | 1 | `SaveImageWebsocket`, périmètre `reference-test` conservé |
| Identités/module dans le catalogue de nœuds | 940 | 927 candidats intégrés et 13 autres déclarations |
| Lignes de nœuds du manifeste | 942 | Deux identités possèdent chacune une déclaration de mapping et une déclaration de schéma |
| Entrées totales du manifeste | 1 160 | 1 106 entrées initiales et 54 nouveaux identifiants |
| Contrats de source locaux et de référence | 663 | Toutes les 662 identités locales et la référence custom |
| Contrats dont les arguments explicites sont résolus statiquement | 533 | Valeurs et structures déclarées, sans finalisation ou validation d'exécution |
| Contrats partiellement résolus | 130 | 281 expressions non résolues, chacune associée à un chemin et une source |

Le [catalogue d'inscriptions](../capabilities/node-registration.json) donne l'ordre des modules, le mécanisme de chargement, la classe réelle, les conditions, les imports du module, les sources et l'état de chaque identité. Le [catalogue des schémas](../capabilities/node-schemas.json) donne les entrées, sorties, métadonnées, héritages, sources par paramètre et expressions non résolues. Le [manifeste](../capabilities/manifest.json) relie chaque ligne à ces preuves sans modifier ses statuts de réalisation ou de qualification.

## Chemins de chargement rapprochés

Le chargeur donne priorité à `NODE_CLASS_MAPPINGS` lorsqu'il est présent. Sinon il appelle `comfy_entrypoint`, `on_load`, `get_node_list`, puis `GET_SCHEMA` pour chaque classe. Les exceptions deviennent des échecs d'import. Une inscription ultérieure remplace un identifiant déjà présent, sauf lorsqu'il appartient à `ignore`. Ces règles sont documentées depuis le [chargeur épinglé](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/nodes.py#L2294).

La liste des extras est rapprochée dans son ordre exact. Les modules API viennent du glob `nodes_*.py` trié, activé par `init_api_nodes`. La découverte des modules custom passe ensuite par les chemins configurés, les règles de désactivation, la whitelist et la politique du manager. L'ensemble des noms intégrés protège ces noms des remplacements custom. Les répertoires privés et tiers n'ont pas été parcourus.

Un contrôle AST indépendant a retrouvé **168 retours directs de listes d'extension et dix modules de mapping**. Neuf de ces mappings sont des dictionnaires littéraux ; `nodes_hooks.py` remplit son dictionnaire dans une boucle de vingt classes. Le total indépendant est de 927 candidats, sans doublon d'identifiant actif. Aucun autre module suivi de `comfy_extras` déclarant un mapping ou une entrée d'extension n'est absent de la liste du chargeur. Les opérations de mapping et leurs conditions sont conservées dans `modules[].mappingOperations`, y compris les affectations de la boucle.

`nodes_replacements.py` expose une liste vide mais son `on_load` enregistre des remplacements. Ses **huit règles** sont conservées dans `replacementRules`, avec les identifiants anciens/nouveaux et les transformations d'entrées/sorties. Ces remplacements ne créent pas de nouvelles entrées `NODE_CLASS_MAPPINGS`.

## Écarts avec les seules déclarations littérales

| Source | Identifiants supplémentaires | Cause |
| --- | ---: | --- |
| `comfy_extras/nodes_hooks.py` | 20 | Clés `node.NodeId` affectées depuis `node_list` |
| `comfy_extras/nodes_dataset.py` | 19 | `define_schema` hérité et identifiants définis comme attributs de classe |
| `comfy_extras/nodes_context_windows.py` | 2 | Schéma parent puis mutation de `schema.node_id` |
| `comfy_extras/nodes_hunyuan.py` | 1 | Variante `EmptyHunyuanVideo15Latent` dérivée du schéma parent |
| `comfy_api_nodes/nodes_bfl.py` | 4 | Variantes dérivées avec identifiants modifiés |
| `comfy_api_nodes/nodes_comfy_cloud.py` | 8 | Constructeurs de schéma partagés et attributs de classe |

Les 54 identifiants sont énumérés dans `literalInventoryReconciliation.addedCandidates`. Les doublons de déclaration initiaux de `ModelAttentionBackend` et `SUPIRApply` restent deux lignes chacun dans le manifeste ; chaque paire pointe vers une seule identité de catalogue. Les identifiants, sources, périmètres, paramètres de qualification, implémentation, tests, workflows, plateformes, preuves et scénarios des 1 106 entrées originales sont inchangés.

Les dix déclarations locales non inscrites sont `AutogrowNamesTestNode`, `AutogrowPrefixTestNode`, `BatchImagesMasksLatentsNode`, `ComboOptionTestNode`, `ComfySoftSwitchNode`, `ConvertStringToComboNode`, `DA3GeometryToPointCloud`, `DCTestNode`, `InvertBooleanNode` et `MultiGPU_Options`. Les deux déclarations distantes non inscrites sont `MinimaxSubjectToVideoNode` et `RecraftStyleV3VectorIllustrationNode`. Leur état `declaration-not-registered` interdit de les confondre avec des candidats intégrés. Le périmètre historique de chaque ligne demeure conservé.

## Format du contrat de source

Chaque enregistrement utilise le couple `(nodeId, module)`. `sourceClass`, `classModule`, `classSource`, `schemaSources` et `registrationSources` identifient la provenance, même lorsqu'un schéma est hérité. `parameterSources` ajoute les références des entrées/sorties et leur chemin dans le document.

Pour une classe legacy, `inputs` conserve les groupes `required`, `optional` et `hidden` de `INPUT_TYPES`. Les tuples deviennent des tableaux JSON. `outputs` conserve notamment `RETURN_TYPES`, `RETURN_NAMES` et `OUTPUT_IS_LIST` lorsqu'ils sont déclarés ou hérités. Les boucles bornées de `ModelMergeSD1` et des autres variantes produisent les noms réels des paramètres : les 32 entrées requises SD1 comprennent ainsi `input_blocks.0.` à `input_blocks.11.`.

Pour une classe V3, chaque constructeur conserve `constructor`, `arguments`, `fields` et `source`. Les options imbriquées de `DynamicCombo`, les modèles `Autogrow.TemplatePrefix`/`TemplateNames`, les champs optionnels, valeurs par défaut, bornes, pas, tooltips et drapeaux déclarés restent structurés. Les schémas de `TextGenerate` et `TextGenerateLTX2Prompt` partagent leurs neuf entrées ; l'option de sampling `on` conserve ses sept sous-entrées. Les variantes Wan/LTX de fenêtres de contexte gardent chacune leurs dix entrées modifiées. `WanContextWindowsManual` conserve bien la dernière affectation du nom d'affichage, `Wan Context Windows`.

Une expression non résolue est un objet avec `resolution: "unresolved"`, l'expression amont et sa source épinglée. `unresolvedFields` donne son chemin. Les branches conditionnelles peuvent inclure `alternatives` ; elles ne deviennent pas une valeur choisie arbitrairement. Par exemple, la liste du backend d'attention distingue la présence de Comfy Kitchen de son absence.

`sharedContract` contient 132 définitions de l'API IO : types déclarés par `comfytype`, signatures de constructeurs, valeurs par défaut, enums et champs de `Schema`. Les valeurs par défaut partagées restent séparées des arguments explicitement fournis par chaque nœud. La finalisation ajoute notamment des entrées cachées et des identifiants de sortie ; ses [règles amont](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_api/latest/_io.py#L1794) sont référencées. Le document n'est pas une capture de `/object_info` et n'affirme pas avoir exécuté `GET_SCHEMA`, `validate`, la conversion V1 ou les valeurs par défaut de tous les comportements de widgets.

## Limites et preuves encore requises

Les 281 expressions restantes incluent des fournisseurs de fichiers, modèles et jeux de données, des listes de périphériques, des helpers de schéma, de la réflexion sur les modes de traitement et certaines références de types/enums. Elles restent consultables dans `unresolvedCases` et, avec leur détail, dans `records[].unresolvedFields`. Les résoudre ou implémenter leurs fournisseurs natifs constitue du travail restant ; l'absence d'un import optionnel ne supprime aucun candidat.

Les trois familles de limites d'inscription sont explicites dans le catalogue : disponibilité des imports et extensions, découverte custom hors périmètre public suivi, puis finalisation/validation effective du schéma. Les identités candidates des chemins intégrés sont rapprochées à ce commit, mais leur inscription exécutable sur une machine n'est pas prouvée.

Le catalogue global reste incomplet : variantes de modèles, encodeurs, VAE, widgets, formats historiques, dépendances natives et scénarios de référence/hardware exigent encore une qualification dédiée. Aucune nouvelle entrée n'acquiert `unitTests`, `realWorkflow` ou une plateforme validée à partir de cette extraction.

## Vérification de l'artefact

Le laboratoire isolé utilise uniquement l'AST et des opérations déterministes sur littéraux/listes/dictionnaires ; il est extérieur au dépôt produit. Aucun source Python, interpréteur ou dépendance Python de produit/test/outillage n'est ajouté à ComfySharp. Les entrées du laboratoire sont les fichiers publics suivis de l'amont épinglé. Les sorties distribuées sont JSON et Markdown avec liens publics ou relatifs.

Les contrôles ont vérifié l'ordre des 178 modules, les 927 candidats par un second parcours indépendant, la couverture des 663 schémas, les 1 106 lignes initiales inchangées, les cas ciblés d'héritage/mutation/loop/options conditionnelles, l'absence de chemins locaux dans les artefacts et 8 711 références de source épinglées. Les sorties JSON ont été régénérées pour vérifier leur déterminisme. Ces contrôles vérifient l'inventaire documentaire ; ils ne qualifient pas les implémentations C# ou l'inférence.
