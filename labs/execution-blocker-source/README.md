# ExecutionBlocker — laboratoire source V1

Ce laboratoire produit un corpus de comparaison depuis six déclarations AST
inchangées de ComfyUI, au commit
`1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`. Il utilise uniquement la bibliothèque
standard Python et Git. Aucun module ComfyUI, Torch ou modèle n'est importé.
Il reste séparé du produit et des tests .NET distribués.

Les 24 cas sont définis avant exécution dans [protocol.json](protocol.json) :
messages silencieux, vides et non vides, ordre des inputs, répétition du dernier
élément, listes, sorties partielles, lazy, async, UI, erreur et interruption.
Les résultats attendus ne sont pas reconstruits depuis l'implémentation C#.

## Déclarations exécutées

- `ExecutionBlocker` depuis `comfy_execution/graph_utils.py`.
- `_async_map_node_over_list`, `resolve_map_node_over_list_results`,
  `merge_result_data` et `get_output_from_returns` depuis `execution.py`.
- La fonction imbriquée `execution_block_cb`, compilée sans modifier son AST ;
  les bindings de sa fermeture sont fournis explicitement par le laboratoire.

Chaque blob Git est vérifié par SHA-256 avant compilation. Le résultat conserve
les hashes des AST, segments source, protocole et fichiers du laboratoire. La
commande d'exécution exige que ces trois fichiers soient déjà publiés dans le
commit local admis avant le calcul et identiques à leurs blobs Git. Cette
révision et ces fichiers sont revérifiés avant écriture du résultat.

## Adaptations explicites

Les nœuds V1 de fixture renvoient des valeurs entières et des contrôles. Un
contexte enregistre les appels ; un serveur enregistre les événements sans
réseau ; le contrôle d'interruption lève une exception dédiée à l'indice prévu.
Deux classes sentinelles représentent les types V3 pour les tests de type ;
aucun nœud ne les instancie et aucune branche V3 n'est exercée.

Les chaînes `fixture-prompt`, `fixture-client` et `fixture-upstream` remplacent
les identités d'infrastructure. Le callback source reçoit une séquence ordonnée
d'exécutions antérieures. Les retours async sont résolus par la déclaration
source correspondante. Les retours du hook lazy sont enregistrés directement ;
le laboratoire n'exécute pas le réordonnancement du graphe.

L'objet à clé unique `$blocker` sert uniquement à encoder les données de ce
corpus. Ce n'est ni une valeur littérale spéciale du prompt ComfySharp, ni un
format d'export des valeurs runtime. Aucun résultat de ce laboratoire ne
qualifie `PromptExecutor`, les sous-graphes, les caches persistants, la couche
WebSocket complète ou la compatibilité V1 du moteur.

## Utilisation

Depuis un clone indépendant de ComfySharp, avec un clone source accessible :

```sh
python labs/execution-blocker-source/reference.py --source /path/to/ComfyUI
python labs/execution-blocker-source/reference.py --source /path/to/ComfyUI --execute --output /new/absolute/output
```

La première commande vérifie les sources et compile les déclarations sans les
exécuter. La seconde refuse un dossier de sortie existant ou situé dans l'un
des deux dépôts, même ignoré par Git. Un résultat complet
est écrit dans `reference.json` ; une interruption du processus peut laisser un
dossier incomplet, qui ne constitue pas une collecte réussie.

Les captures incluent l'ordre des invocations, arguments, événements, contrôles
d'interruption, retours avant fusion et sorties/UI fusionnées. L'identité des
entrées après calcul est vérifiée ; les erreurs de fixture attendues sont
enregistrées par type. Toute autre exception fait échouer la collecte.
