# Supprimer et couper une sélection

**Delete selection** supprime les nœuds sélectionnés en une seule édition annulable. Les touches **Suppr** et **Retour arrière** déclenchent la même commande lorsque le canvas reçoit le clavier. **Cut nodes** copie la sélection dans le presse-papiers, puis la supprime seulement lorsque l'écriture a réussi. Les champs de texte conservent leurs propres commandes d'édition.

Les commandes Copier, Couper et Coller du canvas utilisent les gestes fournis par la plateforme Avalonia, comme les champs de texte. Cela suit les raccourcis natifs et leur configuration, sans les déduire uniquement de l'OS.

## Reconnexion lors de la suppression

La source [`LGraphCanvas.deleteSelected`](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/src/lib/litegraph/src/LGraphCanvas.ts#L4875) appelle [`LGraphNode.connectInputToOutput`](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/src/lib/litegraph/src/LGraphNode.ts#L3958) avant de retirer chaque nœud. La commande C# reproduit ce passage pour les graphes statiques :

1. Une entrée connectée peut remplacer les connexions de la sortie au même index, si les types entrée/sortie sont compatibles.
2. Chaque consommateur est reconnecté à la source effective seulement si ses ports sont compatibles. Une connexion à soi-même est omise.
3. Avec `flags.keepAllLinksOnBypass: true`, un second passage recherche la première sortie compatible pour chaque entrée. Le réglage par défaut reste faux.
4. Les liens encore incidents au nœud supprimé sont déconnectés ; leurs références sont nettoyées des ports restants.

L'ordre de sélection est conservé. Une chaîne de plusieurs nœuds sélectionnés peut donc être reconnectée progressivement. Les nouveaux liens reçoivent des identifiants supérieurs au compteur et aux liens existants, dans la limite des entiers exactement représentables en JavaScript. Les compteurs sont désormais aussi respectés par la commande Connect. Les champs opaques des nœuds survivants ne sont pas modifiés. Les anciens liens supprimés sont remplacés par de nouveaux liens standards, comme lors d'une connexion.

Les booléens `block_delete: true`, `ignore_remove: true` et `removable: false` protègent un nœud. Ils sont contrôlés avant toute reconnexion. Ce choix évite la modification intermédiaire possible dans la source lorsque le canvas reconnecte puis que [`LGraph.remove`](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/src/lib/litegraph/src/LGraph.ts#L1051) refuse le retrait. Ces protections et les limites de coercition ne sont pas une déclaration d'équivalence de tous les cas JavaScript atypiques.

La méthode historique `WorkflowDocument.Delete(NodeId)` demeure une suppression directe sans reconnexion ; l'interface utilise `DeleteNodes` pour la sémantique de sélection décrite ici.

## Couper sans perdre le document

`PrepareCut` valide la copie et simule la suppression sur une copie indépendante avant toute écriture du presse-papiers. Les nœuds protégés restent dans le document et ne sont pas inclus dans le fragment. Un nœud supprimable mais `clonable: false` annule Couper avec un diagnostic.

Après l'écriture, l'éditeur vérifie qu'il est encore ouvert et que le document n'a pas été modifié. `CommitCut` vérifie aussi le propriétaire et l'état du document avant d'appliquer le résultat en une seule édition. La sélection capturée au déclenchement est utilisée, même si la sélection visuelle change pendant l'attente. Un échec du presse-papiers ne change ni le document, ni ses aperçus, ni son historique d'édition. Si le document change après une écriture réussie, la copie peut déjà être présente dans le presse-papiers, mais aucun nœud n'est retiré.

Les liens incohérents, les sous-graphes reconnus, les liens incidents avec reroutes et les liens flottants incidents sont refusés atomiquement en attendant leur portage. Les callbacks et cycles de vie d'extensions ne sont pas exécutés. La compatibilité des types réutilise le [profil statique existant](WORKFLOW_MODES.md), dont les limites restent applicables.

## Vérification

La [campagne Windows locale](qualification/workflow-delete.json) passe **244 tests Workflow et 102 tests Desktop**, dont **28 nouveaux**, sans échec ni test ignoré. Le build Release complet passe sans avertissement. Les tests couvrent les formats 0.4/1, les chaînes dans les deux ordres, les consommateurs multiples, les index/types, les protections, les échecs atomiques, undo/redo, les raccourcis et les écritures de presse-papiers retardées ou défaillantes.

Le parcours natif supprime un StringReplace puis fait exécuter le graphe reconnecté par le Host séparé. Il vérifie ensuite Couper, Coller dans un nouvel onglet, l'exécution attendue et la restauration exacte par Undo. Le transfert utilise un fragment en mémoire ; le presse-papiers du poste n'est pas remplacé. Les adaptateurs du presse-papiers et les raccourcis sont vérifiés avec Avalonia Headless.

La [CI précédente du presse-papiers](https://github.com/skyline624/ComfySharp/actions/runs/34683919217/job/103527424683) avait révélé un échec macOS : le test attendait une copie de texte avec le modificateur déduit de l'OS. Le canvas et les tests utilisent maintenant la configuration de gestes d'Avalonia. Un cas configurant explicitement Ctrl+Shift+C passe localement. La [preuve macOS au commit af29a1d](qualification/workflow-delete-macos.json) confirme **244 tests Workflow et 102 tests Desktop réussis**, sans échec ni test ignoré, dont le cas de régression et les 28 nouveaux cas. Le parcours natif est également confirmé par le journal du job. Le [job macOS](https://github.com/skyline624/ComfySharp/actions/runs/34684628480/job/103529355691) reste en échec sur les comparaisons CLIP aux dimensions complètes ; aucune tolérance numérique n'est modifiée.

Cette tranche ne qualifie pas encore les groupes, sous-graphes, reroutes, callbacks d'extensions, la persistance native du presse-papiers ou une famille de modèles. Les autres projets de tests n'ont pas été répétés dans la campagne locale.
