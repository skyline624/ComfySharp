# Modes d'exécution du document

L'éditeur propose **Enable**, **Mute** et **Bypass** pour le nœud sélectionné. Le mode apparaît sur le nœud, reste dans le workflow sauvegardé et participe à undo/redo. Un changement de mode invalide les aperçus et les résultats des soumissions antérieures comme les autres éditions.

La compilation construit un prompt distinct sans modifier le document :

- **Mute / mode 2** omet le nœud et ne fournit aucune connexion aux consommateurs. Une valeur de widget déjà présente chez le destinataire demeure disponible ; une entrée sans valeur reste absente et devra satisfaire la validation du Host.
- **Bypass / mode 4** omet le nœud et suit une entrée sélectionnée d'après le slot et les types. Il ne calcule pas ce nœud et n'utilise pas son widget comme résultat.
- Les modes 0, 1 et 3 sont sérialisés comme les nœuds actifs dans le prompt API, comme dans le frontend. Cela n'ajoute pas de boucle d'événements LiteGraph au moteur. Les modes inconnus reçoivent un diagnostic.

Un type non porté peut rester muet ou être traversé en bypass à partir de ses ports conservés, sans exécuter d'extension. Un type actif non porté reste une erreur de compilation.

## Sélection d'une entrée de bypass

Le comportement de référence est celui de [`ExecutableNodeDTO`](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/src/lib/litegraph/src/subgraph/ExecutableNodeDTO.ts#L235). Pour une destination `*` ou vide, le slot opposé est choisi s'il existe, sinon le premier. Pour les autres types, la priorité est : slot opposé compatible avec sortie et destination, première correspondance exacte à la destination, puis première entrée compatible avec les deux.

La sélection précède l'examen de la connexion : une entrée choisie non connectée ne provoque pas une recherche parmi les autres entrées. Chaque maillon utilise ensuite le type de son entrée choisie pour continuer la résolution. Les règles de compatibilité proviennent de [`LiteGraphGlobal.isValidConnection`](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/src/lib/litegraph/src/LiteGraphGlobal.ts#L688), avec types génériques, casse et listes séparées par des virgules. Les noms de types ASCII usuels sont qualifiés ici ; l'équivalence des conversions JavaScript pour tous les types Unicode ou valeurs atypiques reste à établir.

Une absence de type correspondant produit un avertissement `bypass_no_match` et laisse la connexion omise. Les avertissements restent visibles dans le résultat de compilation et les messages d'export/soumission. Les cycles parcourus en résolvant un bypass et les slots absents sont des erreurs. La recherche de cycles du graphe exécutable utilise les connexions résolues : une boucle entièrement inactive ne bloque plus une sortie indépendante. Le document conserve toujours les liens physiques, et leur validation structurelle existante reste plus stricte que certaines entrées malformées tolérées par le frontend.

La sérialisation des widgets et l'omission des modes suivent [`graphToPrompt`](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/src/utils/executionUtil.ts). Les sous-graphes, applications de nœuds virtuels, Set/Get, callbacks d'extensions et reroutes complets restent à porter ; cette tranche ne les simule pas. Elle n'établit pas de compatibilité avec l'intégralité du compilateur frontend.

## Vérification

La [campagne locale](qualification/workflow-modes.json) couvre les formats 0.4/1, les priorités de slots/types, les chaînes de bypass, entrées déconnectées, widgets de repli, cycles actifs/inactifs, nœuds inconnus et modes persistés. Les tests Desktop actionnent les boutons, vérifient les badges, les liens conservés et undo/redo. Le smoke natif exécute un même graphe StringReplace → PreviewAny dans les modes 0, 4 et 2 avec un Host séparé et contrôle ses trois résultats distincts.

Les tests de contrats sont construits en C# après lecture des sources et tests amont figés. Aucun frontend JavaScript ni corpus différentiel exécuté n'est inclus dans cette tranche. Les tolérances numériques, références de modèles et dépendances natives restent inchangées ; aucune famille de génération n'est qualifiée par ces résultats.
