# Groupes du document et du canvas

L'éditeur affiche les cadres de groupes des workflows 0.4 et 1. **Group selected nodes** crée un groupe autour de la sélection. Le panneau permet de choisir un groupe, modifier son titre, sa couleur et sa taille, l'épingler ou retirer son cadre. Retirer un cadre conserve les nœuds et les connexions.

Glisser son en-tête déplace le groupe et son contenu. Maintenir **Shift au début du geste** déplace uniquement le cadre. Les groupes épinglés restent immobiles. Le déplacement se prévisualise sans écrire dans le document ; relâcher la souris crée une seule édition annulable. Une perte de capture, une reconstruction du canvas ou une modification concurrente du document annule ce déplacement non validé.

## Géométrie et données

La référence est [`LGraphGroup`](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/src/lib/litegraph/src/LGraphGroup.ts) au commit frontend figé. Le rectangle `bounding`, le titre, la couleur, les flags et tous les champs inconnus restent dans le document. Les groupes anciens sans identifiant sont affichés sans réécriture à l'import. Les nouveaux groupes obtiennent un identifiant numérique distinct des identifiants de nœuds ; la version 1 conserve `state.lastGroupId`.

`WorkflowGroups.cs` fournit une projection indépendante d'Avalonia. L'éditeur lui transmet les rectangles mesurés des nœuds lorsque ceux-ci sont disponibles. Sinon, la projection utilise les dimensions sérialisées et une hauteur de titre de 30 unités, conformément au cas standard de [`LGraphNode.measure`](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/src/lib/litegraph/src/LGraphNode.ts#L2186). Sans mesure, des dimensions absentes utilisent le défaut du port, 220×100 ; un nœud réduit exige une mesure explicite. Les géométries personnalisées d'extensions et la reproduction exacte des dispositions frontend restent à qualifier.

La création ajoute une marge de 10 unités et l'espace du titre. Les commandes de taille respectent les minima amont de 140×80. Le contenu est déterminé par le centre des rectangles des nœuds. Les groupes entièrement contenus et les positions de reroutes sérialisées à l'intérieur du cadre suivent le déplacement. Les nœuds épinglés et les cadres imbriqués épinglés ne sont pas déplacés. Chaque nœud et chaque cadre est translaté une seule fois, y compris en présence de groupes imbriqués ; tous les cas récursifs historiques LiteGraph/Vue ne sont pas déclarés équivalents.

Les positions de reroutes sont lues dans `extra.reroutes` en 0.4 et `reroutes` en version 1. Cette translation des données ne constitue pas une qualification de leur affichage ou de leur édition. Les structures non interprétables provoquent un diagnostic et un retour à l'état initial. Un groupe au rectangle invalide reste sauvegardable dans le document ; le panneau signale pourquoi les groupes ne peuvent pas être affichés.

Les groupes n'ajoutent aucun nœud au prompt exécutable. Les liens et les valeurs des nœuds restent inchangés pendant les commandes de groupe. Les tableaux ou objets de coordonnées conservent leurs champs supplémentaires. Le lecteur de coordonnées accepte aussi les valeurs JSON numériques créées directement par les modèles C#, dont les dimensions entières.

## Vérification

La [campagne locale Windows](qualification/workflow-groups.json) passe **261 tests Workflow et 112 tests Desktop**, dont **27 nouveaux**, sans échec ni test ignoré. Le build Release complet passe sans avertissement. Les cas couvrent les deux formats, les champs inconnus, les groupes sans ID, les compteurs, les groupes imbriqués, le contenu géométrique, les pins, les reroutes sérialisées, les erreurs atomiques et undo/redo.

Avalonia Headless réalise des événements souris réels sur les en-têtes, avec zoom 1 et 1,5, déplacement du cadre seul, capture interrompue, reconstruction du canvas et édition concurrente. Le parcours natif crée un groupe autour d'un graphe à trois nœuds, le déplace, vérifie undo/redo et fait exécuter le même prompt par le Host séparé avec le résultat attendu.

Copier/couper/dupliquer les groupes, les sélections rectangulaires mixtes, les poignées de redimensionnement, le déplacement automatique de la vue pendant le glissement, les styles et typographies exhaustifs, les callbacks de géométrie et la qualification matérielle multiplateforme restent ouverts. Les autres projets de tests n'ont pas été répétés ; aucune famille de modèles ni plateforme GPU n'est qualifiée par ces résultats.
