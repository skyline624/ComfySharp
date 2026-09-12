# Copier et coller une sélection

**Copy nodes** copie les nœuds sélectionnés et leurs liens internes dans le presse-papiers système. **Paste nodes** les insère dans le document actif, avec de nouveaux identifiants, à proximité du coin supérieur gauche de la vue. Les copies deviennent la sélection courante. Le canvas accepte aussi **Ctrl+C / Ctrl+V**, ou **Cmd+C / Cmd+V** sur macOS ; les champs de texte gardent leurs raccourcis d'édition.

Le document source reste inchangé. Les connexions vers des nœuds non copiés sont retirées du fragment ; elles ne peuvent pas se raccorder par hasard à un identifiant identique dans le document destinataire. La [duplication dans un même document](WORKFLOW_DUPLICATION.md) conserve son option distincte de reconnexion des entrées externes.

Chaque collage est une seule modification annulable et rétablissable. Une erreur de contenu, de conversion ou d'allocation d'identifiant rétablit le document complet et conserve sa pile undo/redo. Une lecture asynchrone du presse-papiers ne peut plus modifier un éditeur fermé ou un document modifié pendant l'attente : un nouveau collage est alors nécessaire.

## Format et conservation

Le texte utilise l'enveloppe `{"format":"ComfySharp.nodes","version":1,"workflow":{...}}`. Le fragment contient uniquement la sélection, ses liens internes et les informations de structure nécessaires. Il ne copie pas les réglages du document entier. Ses identifiants sont locaux au fragment, puis remappés dans le document destinataire.

`WorkflowDocument.CopyNodes` et `PasteNodes` sont indépendants d'Avalonia. Les modes, widgets, propriétés, données d'import API et champs inconnus des nœuds sont clonés profondément. Une entrée `clonable: false` reste exclue. Le coin supérieur gauche des nœuds est placé à la position demandée, en conservant leurs écarts relatifs. La taille du texte est limitée à 16 Mi caractères UTF-16 à la copie et au collage.

Les quatre transferts 0.4→0.4, 0.4→1, 1→0.4 et 1→1 sont couverts pour les liens standards. Les liens deviennent des tableaux pour la version 0.4 et des objets pour la version 1. Les champs supplémentaires des liens sont conservés lors d'un transfert de même version. Lors d'un changement de version, leur présence produit un diagnostic et annule le collage : la conversion sans perte des extensions reste à porter. Les références cachées dans des champs opaques ne sont pas interprétées.

La source fonctionnelle est [`LGraphCanvas._serializeItems` et `_deserializeItems`](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/src/lib/litegraph/src/LGraphCanvas.ts#L4063), avec la remise à zéro des connexions de [`LGraphNode.clone`](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/src/lib/litegraph/src/LGraphNode.ts#L1108). Le frontend utilise `localStorage` ; ce port utilise un texte versionné propre à ComfySharp sur le presse-papiers natif. Il ne revendique pas la compatibilité binaire avec le presse-papiers interne du navigateur.

## Vérification et travail restant

La [campagne Windows locale](qualification/workflow-clipboard.json) passe **224 tests Workflow et 94 tests Desktop**, dont **19 nouveaux**, sans échec ni test ignoré. Le build Release complet passe sans avertissement. Les tests couvrent collisions d'identifiants, liens, positions, champs inconnus, indépendance des documents, erreurs atomiques, undo/redo, lectures retardées et fermeture. Avalonia Headless vérifie l'adaptateur du presse-papiers, le changement d'éditeur et les raccourcis clavier.

Le parcours natif copie un fragment depuis une sélection d'un document 0.4, le colle dans un nouvel onglet version 1 et fait exécuter ce graphe par le Host séparé. Il vérifie la sortie attendue, l'intégrité du document source et undo/redo. Il utilise les API de fragment, sans lire ni remplacer le presse-papiers système du poste. La propriété du presse-papiers natif après fermeture, sa persistance et les interactions réelles Linux/macOS restent à qualifier.

[Couper et la suppression de sélection](WORKFLOW_DELETE_CUT.md) sont livrés dans une tranche distincte, qui utilise aussi les gestes configurés par Avalonia pour les raccourcis. Groupes, reroutes, sous-graphes, remappage des références d'extensions et reconnexion optionnelle d'entrées externes par collage restent ouverts. Les autres projets de tests n'ont pas été répétés dans cette campagne ; ces preuves ne valident aucune famille de modèles ni plateforme GPU.
