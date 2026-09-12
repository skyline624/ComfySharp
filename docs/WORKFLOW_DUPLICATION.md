# Duplication de nœuds

Dans l'éditeur, sélectionner un ou plusieurs nœuds puis utiliser **Duplicate selection** crée des copies décalées de 40 unités sur chaque axe. Les connexions entre les nœuds copiés sont recréées ; les connexions vers les destinations non sélectionnées restent attachées aux originaux. **Duplicate with external inputs** reconnecte aussi les entrées provenant de sources non copiées. Les copies deviennent la sélection courante.

L'opération est une seule édition annulable et rétablissable. Elle invalide les aperçus précédents comme les autres modifications du document. Une erreur de structure ou d'allocation d'identifiant annule toute la duplication, y compris les changements de compteurs et de connexions. Un champ `clonable: false` exclut le nœud de la copie.

## Contrat du document

`WorkflowDocument.DuplicateNodes` fonctionne sans Avalonia pour les workflows 0.4 et 1. Il retourne la correspondance entre les identifiants des originaux et ceux des copies. Les nouveaux identifiants numériques dépassent les identifiants numériques et les compteurs existants, dans la limite des entiers exactement représentables en JavaScript. Les identifiants source textuels, y compris `01`, restent inchangés.

Les widgets, modes, métadonnées d'import API et champs inconnus sont copiés profondément. Les identifiants de nœuds, extrémités de liens et références de ports connues sont réécrits ; les autres données ne sont pas interprétées. Les propriétés supplémentaires des positions et des liens sont conservées. Les copies peuvent être éditées et compilées indépendamment des originaux.

La base fonctionnelle est la copie/sérialisation puis la reconstruction des liens dans [`LGraphCanvas.ts`](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/src/lib/litegraph/src/LGraphCanvas.ts#L4050), ainsi que la remise à zéro des références dans [`LGraphNode.clone`](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/src/lib/litegraph/src/LGraphNode.ts#L1108). Le port conserve aussi les nœuds inconnus, conformément au contrat de migration, sans exécuter leurs extensions.

## Limites explicites

Cette tranche couvre la duplication de nœuds dans un même document. Le presse-papiers, les groupes, les reroutes et la duplication des définitions de sous-graphes restent à porter. Une instance reconnue dans `definitions.subgraphs` ou un lien objet copié contenant `parentId` produit un diagnostic sans modification du document. Les références cachées dans les données opaques d'une extension sont conservées telles quelles ; leur remappage requiert le portage de cette extension. Les callbacks JavaScript ne sont pas exécutés.

Les compteurs sont lus sous leur forme entière habituelle. Cela ne qualifie pas toutes les représentations numériques atypiques ni tous les workflows malformés acceptés par le frontend. L'allocation existante des autres commandes d'édition n'est pas modifiée par cette tranche.

## Vérification

La [campagne Windows locale](qualification/workflow-duplication.json) passe **210 tests Workflow et 89 tests Desktop**, dont **20 nouveaux**, sans échec ni test ignoré. Le build Release complet passe sans avertissement. Les cas couvrent les deux versions de document, la conservation des données, les liens internes/externes, les nœuds inconnus, les erreurs atomiques, undo/redo, l'indépendance des widgets et la sélection du canvas.

Le parcours natif ouvre la fenêtre avec un Host séparé, duplique trois nœuds connectés, modifie le texte et le mode d'une copie puis exécute les deux branches dans un même prompt. Il vérifie les deux résultats distincts et l'intégrité des originaux. Les autres projets de tests n'ont pas été répétés. Aucun corpus JavaScript différentiel exécuté, modèle préentraîné, GPU ou nouvelle plateforme n'est qualifié ici.
