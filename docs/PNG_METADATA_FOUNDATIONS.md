# Fondations PNG et métadonnées d'exécution

Cette documentation décrit la tranche initiale de l’encodeur PNG C# et de la transmission des métadonnées. Son registre comptait alors 35 identifiants. Les [nœuds SaveImage/PreviewImage et leur API](IMAGE_FILE_NODES.md), puis les [aperçus bitmap natifs](BITMAP_PREVIEWS.md), ont été raccordés dans les tranches suivantes. Aucune capacité de génération de modèle n’est promue.

## Entrées cachées

`NodeSchema.HiddenInputs` décrit une liste ordonnée de noms et de types. Le registre expose ces déclarations dans `input.hidden` et `input_order.hidden`. Elles ne deviennent ni des widgets ni des connexions requises dans le prompt.

Le moteur prend en charge les entrées cachées historiques suivantes :

| Type | Valeur fournie |
|---|---|
| `PROMPT` | Copie du prompt original complet, avec les nœuds non sélectionnés et leurs métadonnées |
| `EXTRA_PNGINFO` | Valeur de `extra_data.extra_pnginfo`, ou JSON null si absente |
| `UNIQUE_ID` | Identifiant du nœud exécuté, conservé comme chaîne, y compris les identifiants contenant `:` |

Le prompt et les données supplémentaires sont copiés avant le premier `await`. Chaque invocation reçoit des valeurs JSON isolées via le contrat de propriété du moteur. Les trois méthodes d'exécution acceptent `extraData`, et la file du Host transmet les données déjà figées lors de l'insertion du travail. Les autres champs de `extra_data` restent dans l'historique sans devenir des arguments cachés du nœud.

Ces métadonnées participent à la sémantique des listes : chaque champ caché est une liste d'exécution contenant une valeur. Le moteur répète cette valeur lorsque les entrées ordinaires contiennent plusieurs éléments ; `InputIsList` reçoit cette liste singleton. Une liste ordinaire vide ne peut pas fournir de dernier élément pour une invocation mappée où un champ caché est présent. La sélection des entrées lazy voit aussi les champs cachés.

Les arguments cachés présents dans le prompt ne peuvent pas remplacer les métadonnées du travail. Leur position est conservée dans l'ordre des arguments ; les champs absents sont ajoutés dans l'ordre du schéma. Les noms vides, doublons et collisions avec une entrée visible sont refusés à l'enregistrement.

La référence est le traitement historique de [`get_input_data`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L159), avec la projection [`node_info`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/server.py#L751). Les tests ajoutés sont des tests de comportement C# établis après lecture de la source ; ils ne constituent pas un corpus exécuté dans le laboratoire Python.

`DYNPROMPT`, les secrets des services distants et le contexte caché V3 ne sont pas implémentés par ce contrat. Leur déclaration est refusée explicitement. Le véritable objet de graphe dynamique et le contexte V3 restent à porter ; un objet JSON n'est pas utilisé pour simuler leur disponibilité.

## Encodeur PNG

`ImagePngEncoder.EncodeFrame` encode une frame d'un tenseur dense CPU/Float32 NHWC RGB ou RGBA, avec dimensions positives et sans gradient. Il emprunte le tenseur, respecte les coordonnées d'une vue non contiguë et restitue un tableau d'octets indépendant. Il effectue une multiplication Float32 par 255, une saturation puis une troncature vers un octet. L'alpha reste non prémultiplié. Les valeurs non finies sont actuellement refusées explicitement ; les autres dtypes et dispositifs restent à intégrer.

Le PNG contient un `IHDR` 8 bits RGB/RGBA, des chunks texte ordonnés, un flux zlib réparti en chunks `IDAT`, puis `IEND`. Chaque ligne utilise le filtre None et chaque chunk contient un CRC. Les choix de filtres et les octets compressés ne sont pas annoncés identiques à Pillow. Le format suit la [spécification PNG](https://www.w3.org/TR/2025/REC-png-3-20250624/) ; les niveaux de compression 1 et 4 sont exprimés directement via [`ZLibCompressionOptions.CompressionLevel` de .NET 10](https://learn.microsoft.com/en-us/dotnet/api/system.io.compression.zlibcompressionoptions.compressionlevel?view=net-10.0).

Les mots-clés répétés sont conservés. Le texte Latin-1 utilise `tEXt` ; le texte nécessitant Unicode utilise `iTXt` UTF-8 non compressé. Les mots-clés suivent les contraintes PNG de longueur et d'espaces. Les textes contenant un caractère nul ou une séquence UTF-16 invalide sont refusés.

Les plafonds par défaut sont 16 777 216 pixels par frame, 1 Mio de chunks texte et 128 Mio de PNG encodé. Les deux plafonds de frame et de sortie sont configurables par appel. Ils limitent des allocations distinctes ; ils ne garantissent pas un plafond global de RAM. Les ressources temporaires TorchSharp sont libérées de façon déterministe, y compris sur erreur et annulation.

La quantification et les niveaux de compression ont pour référence [`SaveImage` et `PreviewImage`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/nodes.py#L1660). Aucune dépendance Python ou Pillow n'est ajoutée au produit ou aux tests .NET.

## Validation et travail restant

La [preuve locale Windows CPU](qualification/png-metadata-foundations.json) compte **2 409 tests réussis, aucun échec ni test ignoré**, dont **47 nouveaux** : 23 pour le moteur, 5 pour le Host et 19 pour l'encodeur. La solution compile sans avertissement ni erreur. Les 13 fichiers source, tests et configurations relevés avant compilation sont identiques après la campagne. Ce contrôle ne constitue pas une attestation des DLL natives chargées. Les échecs numériques des campagnes CI précédentes restent ouverts.

Les nouveaux tests couvrent la projection des schémas, les trois frontières d'exécution, les données absentes, les entrées lazy, les listes vides et non vides, la modification du document après suspension, deux travaux simultanés et les soumissions HTTP avec les deux alias. Les tests PNG utilisent un lecteur de test distinct avec un calcul CRC bit par bit, le décodage zlib et les cinq filtres PNG pour vérifier les pixels attendus, l'alpha, les métadonnées, les chunks multiples, les plafonds et la durée de vie des wrappers natifs.

Ce lecteur de test ne remplace pas la qualification par un décodeur natif ni les comparaisons de PNG produits par la source. Les suites prévues à cette étape étaient les suivantes ; le raccordement des fichiers, des nœuds et du bitmap est désormais couvert par les documents liés ci-dessus, tandis que les comparaisons de source et la qualification complète restent ouvertes :

1. Sauvegarder dans les répertoires propres à ComfySharp, avec les règles amont de préfixes, compteurs, noms de fichiers et répertoires.
2. Enregistrer les véritables nœuds `SaveImage` et `PreviewImage`, retourner leur IMAGE empruntée et les descripteurs `ui.images`, puis vérifier les métadonnées incorporées.
3. Servir les fichiers par l'API locale et gérer les aperçus bitmap dans Desktop, avec libération et refus des résultats périmés.
4. Constituer les comparaisons avec la source figée, vérifier les décodeurs natifs et exécuter les parcours complets sur les plateformes requises.

La sérialisation JSON des métadonnées, les formats de pixels supplémentaires et les opérations de fichiers doivent encore être qualifiés. Cette étape ne valide pas le lot médias ni la V1.
