# Import des prompts API

La [CI des trois OS](qualification/import-ci-20260912.md) confirme les 33 nouveaux cas et le parcours natif au commit `5980cbe`. Les contrôles numériques globaux restent ouverts.

**Open** accepte un prompt API JSON et le champ `prompt` d'un PNG dépourvu de workflow graphique. L'import ouvre un nouveau document éditable : positions initiales, ports, connexions et widgets nommés. Une sauvegarde produit un workflow JSON ; le chemin du prompt ou de l'image d'origine n'est jamais utilisé automatiquement comme destination.

Le parcours natif importe un prompt texte, modifie sa valeur dans le document, le compile et l'exécute avec le Host séparé. Il vérifie la sortie modifiée de PreviewAny. Il n'exécute aucune extension ni constructeur de nœud amont pendant l'import.

## Données et édition

- Les clés de nœuds restent des chaînes exactes : `1`, `01`, `PREVIEW` et les identifiants composés restent distincts.
- Une connexion devient un lien du document et peut être déconnectée ou remplacée. Les tableaux littéraux restent des valeurs ; à la compilation ils reçoivent l'enveloppe `__value__` déjà utilisée par le moteur. Les objets, y compris les enveloppes explicites et leurs champs supplémentaires, sont conservés sans déballage destructif. Un `null` explicite n'est pas une entrée omise.
- Les valeurs occupent `widgets_values` sous leur nom. Les ports associés déclarent `widget.name`, contrat explicite de sérialisation accepté par le compilateur. Une connexion prend priorité sur la valeur du widget. L'inspecteur JSON existant permet l'édition ; les widgets spécialisés restent à porter.
- Les champs API extérieurs à `class_type` et `inputs`, dont `_meta`, sont conservés dans `properties["comfysharp.api_import"]`, version 1. La compilation les recopie, puis reconstruit le type et les entrées depuis le document courant. Renommer un nœud met à jour `_meta.title` sans perdre les autres propriétés d'un objet `_meta`. Une version inconnue de cette enveloppe reçoit `invalid_api_import`.
- Les nœuds non portés restent visibles et modifiables. Le compilateur conserve ses diagnostics `unsupported_node` et `unavailable_node`. Un import réussi ne prouve pas qu'un graphe soit exécutable.

L'application fournit les ports typés de ses templates connus. Les sorties absentes du prompt ne sont pas inventées comme capacités : pour un type inconnu, seules les positions nécessaires aux liens présents sont représentées, avec le type `*`. Leur validité d'exécution reste à établir après portage. Les templates sont copiés ; leurs valeurs par défaut ne remplacent pas des entrées absentes du prompt. Les champs inconnus ne servent jamais de copie cachée des entrées à rejouer.

Les liens vers un nœud absent, les cycles et les slots négatifs restent dans le document et reçoivent des diagnostics de compilation. Les slots fractionnaires ou hors Int32 ne sont pas représentables par le contrat GraphLink et font échouer l'import explicitement. L'inférence de sorties sans template est limitée à 4 096 slots par nœud pour éviter une allocation commandée par un index arbitraire. Cette borne n'est pas un plafond global de mémoire. Les [tokens non finis du JSON Python](IMPORT_JSON.md) sont convertis en null avec un avertissement à la frontière d’import.

## Référence et différences explicites

La reconstruction s'appuie sur [`isApiJson` et `loadApiJson`](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/src/scripts/app.ts#L2206) du frontend figé. Le port utilise des widgets nommés, préserve les champs supplémentaires et garde les IDs chaîne afin d'éviter les collisions de conversion numérique. Sa disposition initiale est une grille stable ; elle ne reproduit pas encore l'arrangement LiteGraph ni les callbacks de widgets et les remplacements de nœuds du frontend.

La distinction lien/littéral suit [`graph_utils.is_link`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_execution/graph_utils.py#L1) : tableau de deux éléments avec ID chaîne et slot numérique, y compris le comportement des booléens Python. Les slots numériques intégraux sont normalisés en entiers. Cette distinction préserve les tableaux qui ne sont pas des liens, alors que le frontend figé tente de traiter tous les tableaux comme des connexions. Les enveloppes littérales suivent le contrat de [`execution.py`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L983).

Un workflow graphique PNG non vide conserve la priorité sur `prompt`. Un workflow malformé déclenche maintenant un [repli signalé vers le prompt](IMPORT_JSON.md). A1111, le glisser-déposer, les autres médias, les callbacks et le compilateur frontend complet restent ouverts. Aucun modèle, service distant ou interpréteur n'est requis pour cet import.

## Preuves

La [campagne locale](qualification/api-prompt-import.json) vérifie les projets Workflow et Desktop, le build Release complet et le parcours natif décrit plus haut. Elle identifie les fichiers testés, les hashes des rapports et les limites de la qualification. Les tests .NET utilisent des prompts de contrats construits pour les cas décrits ; aucun programme JavaScript de comparaison ni corpus exhaustif du frontend n'a été exécuté. Les contrôles numériques SD/CLIP et la qualification des familles restent ouverts.
