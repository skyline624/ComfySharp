# SaveImage et PreviewImage : fichiers PNG locaux

Le Host enregistre désormais `SaveImage` et `PreviewImage`, soit **37 identifiants**. Ces deux nœuds exécutent un parcours réel : tenseur IMAGE → fichiers PNG → descripteurs dans les événements et l'historique → lecture HTTP. Leurs sorties IMAGE peuvent alimenter d'autres nœuds. Le profil reste partiel et ne valide aucune famille de modèles.

## Contrat disponible

La référence est le backend figé : [`SaveImage`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/nodes.py#L1660), [`PreviewImage`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/nodes.py#L1721) et le [nommage des fichiers](IMAGE_FILES.md).

| Aspect | Comportement |
|---|---|
| Entrées | `images`; `filename_prefix` pour SaveImage, valeur de widget initiale `ComfyUI` |
| Entrées cachées | `prompt` reçoit le prompt complet figé ; `extra_pnginfo` vient de `extra_data.extra_pnginfo` |
| Sortie | Slot IMAGE nommé `images`, conservant le tenseur reçu avec une propriété partagée gérée par le moteur |
| UI | `images: [{ filename, subfolder, type }]` dans les événements `executed` et l'historique |
| Sauvegarde | Un PNG par image dans `output`, compression zlib 4 |
| Aperçu | Un PNG par image dans `temp`, compression zlib 1, préfixe `ComfyUI_temp_` et cinq caractères |
| Métadonnées | Texte JSON du prompt, puis entrées ordonnées de `extra_pnginfo`; mots-clés dupliqués conservés |

Le [profil de l'encodeur](PNG_METADATA_FOUNDATIONS.md) accepte les tenseurs CPU Float32 denses, finis, de forme NHWC positive, RGB/RGBA et sans gradients. Les données non contiguës sont prises en charge. Les limites par frame sont de 16 777 216 pixels, 1 MiB de texte et 128 MiB de PNG ; elles ne constituent pas une limite globale de mémoire ou de taille du batch. Chaque PNG est encodé avant l'écriture asynchrone. Les ressources tensoriellement temporaires sont libérées avant cette écriture.

Le Host accepte `--disable-metadata true` pour omettre explicitement les textes PNG. Sinon, `extra_pnginfo` doit être un objet ou null. Les métadonnées sont comparées comme du JSON : les espaces et échappements produits par `System.Text.Json` ne sont pas annoncés identiques aux octets de `json.dumps`. Les octets PNG compressés ne sont pas annoncés identiques à Pillow.

Une erreur ou annulation peut laisser les fichiers déjà écrits et, en cas d'interruption pendant une écriture, un fichier partiel. Aucun succès ni descripteur final n'est fabriqué pour un nœud qui échoue. Les résultats natifs libérables gardent leurs propres leases ; le cache d'objets décrit ci-dessous ne possède pas ces sorties.

## Durée de vie des objets de nœuds

`IRuntimeNodeFactory` permet à une définition enregistrée de créer un objet pour chaque couple `(node_id, class_type)`. Le moteur conserve cet objet entre les prompts où ce couple reste présent, même si les entrées changent ou si le nœud n'est pas sélectionné. Cela maintient le suffixe de PreviewImage entre deux exécutions. Le suffixe utilise l'alphabet amont, y compris son caractère répété ; il n'est pas une reproduction du générateur aléatoire global Python.

La clé suit [`CacheKeySetID`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_execution/caching.py#L67). L'instance n'est créée qu'à sa première utilisation, avant la résolution lazy. Une définition sans cette interface reste partagée selon son contrat existant. Le moteur libère les instances `IDisposable` retirées ; les définitions restent la propriété de leur créateur.

Les prompts invalides ne modifient pas ce cache. Lorsque plusieurs appels au moteur se chevauchent, la destruction attend leur fin et utilise le dernier graphe admis pour le nettoyage. Ce mécanisme protège leur durée de vie, sans sérialiser leurs appels sur une même instance. Le Host conserve sa file à un seul travail actif. La fermeture du moteur refuse de nouveaux travaux et reporte la destruction des objets encore utilisés. Ce cache de graphe statique ne réalise ni le cache de résultats inter-jobs, ni LRU, ni l'expansion dynamique du lot 3.

## API et documents

`GET` et `HEAD` sur `/view` et `/api/view` servent le PNG original. Les paramètres sont `filename`, `subfolder` (vide par défaut) et `type` (`output` par défaut, ou `temp`). La réponse utilise `image/png`, `Cache-Control: no-store` et les plages d'octets HTTP. Les règles de [confinement des fichiers](IMAGE_FILES.md) s'appliquent à l'ouverture.

Les options non portées sont refusées explicitement : autres formats, conversions, canal alpha isolé, répertoire `input` et chemins d'assets annotés. Aucun support complet du `/view` amont n'est revendiqué.

Les workflows 0.4 et 1 peuvent compiler `EmptyImage → SaveImage → PreviewImage`. Les modèles de nœuds dans Desktop exposent les entrées et sorties correspondantes, avec sauvegarde et undo/redo. **Desktop n'affiche pas encore les PNG comme bitmaps** et ne transmet pas encore automatiquement son document dans `extra_pnginfo.workflow`. L'appel API peut déjà transmettre ce document. Les substitutions frontend de noms restent à porter.

## Preuves et suites

La [campagne locale](qualification/image-file-nodes.json) sépare les tests des objets, des fichiers tensoriels, du Host, des documents et des templates Avalonia. Les tests vérifient des pixels explicites avec un lecteur PNG C# distinct, les métadonnées, les fichiers réels, les événements, l'historique, HEAD et les plages d'octets. Les tests de durée de vie couvrent réutilisation, lazy/listes, suppression, changement de type, chevauchement, fermeture, erreur et annulation.

Les validations réelles avec poids, le corpus exécuté contre SaveImage/Pillow, les autres dtypes/devices, l'interface bitmap et les plateformes requises restent ouverts. Les tolérances SD/CLIP et leurs échecs CI antérieurs ne sont pas modifiés. Les deux entrées du manifeste restent `partial`, sans déclaration de workflow préentraîné ni de plateforme complète.
