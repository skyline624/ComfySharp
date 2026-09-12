# Ouvrir un workflow incorporé à un PNG

La [CI des trois OS](qualification/import-ci-20260912.md) confirme les 22 nouveaux cas et le parcours natif au commit `8f5c647`. Cette preuve porte sur l'import graphique et conserve les échecs numériques globaux visibles.

Le bouton **Open** accepte maintenant les workflows JSON et les images PNG contenant un workflow graphique. Le PNG ouvre un nouvel onglet ; le document peut être édité puis sauvegardé en JSON. Le chemin de l'image n'est jamais utilisé comme destination de cette sauvegarde. Les onglets existants restent intacts si l'import échoue.

## Lecture des métadonnées

Le comportement pris comme référence est celui de [`metadata/png.ts`](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/src/scripts/metadata/png.ts) : lecture des chunks `tEXt`, `comf` et `iTXt`, texte UTF-8 et dernière occurrence d'une clé retenue. Les mots-clés restent sensibles à la casse. Un `iTXt` compressé utilise zlib ; une compression non reconnue ou corrompue produit un avertissement et ne remplace pas une précédente valeur valide. `zTXt` n'est pas interprété par ce lecteur amont et reste ignoré ici.

Le workflow graphique est choisi avant les autres métadonnées, conformément à [`handleFile`](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/src/scripts/app.ts#L1971). Son import conserve les champs inconnus et les nœuds indisponibles. Si le champ `version` est absent, la frontière d'import ajoute explicitement `0.4`, notamment pour la fixture officielle historique. Les versions explicites prises en charge restent 0.4 et 1. Cette normalisation ne change pas le fichier source.

La lecture des métadonnées ne décode aucun pixel et ne dépend ni d'Avalonia ni de TorchSharp. Le flux appartient à l'appelant et reste ouvert. Les signatures, longueurs et CRC des chunks sont vérifiés ; les PNG tronqués ou corrompus sont refusés explicitement. Le lecteur ne constitue pas une validation complète du contenu image. Il s'arrête à `IEND`.

Le profil limite le fichier parcouru à 128 MiB, les données cumulées des chunks texte reconnus à 16 MiB et leur texte décompressé cumulé à 16 MiB. Une limite dépassée interrompt l'import ; elle n'est pas transformée en simple avertissement de décompression. Au plus cent avertissements détaillés sont conservés, puis un message indique leur omission. L'annulation est propagée pendant la lecture et la décompression.

## Périmètre restant

Le champ `prompt` seul est maintenant [reconstruit comme document API éditable](API_PROMPT_IMPORT.md). Les [nombres non finis du JSON Python](IMPORT_JSON.md) sont convertis avec avertissement. Un workflow graphique non vide conserve la priorité ; s'il est malformé, un avertissement explique le repli vers le prompt API. L'import des paramètres A1111 et les autres médias restent à porter. Les PNG sans métadonnées ne créent pas encore automatiquement un nœud LoadImage. L'import par glisser-déposer reste également ouvert.

Les contrôles de corruption et les limites sont plus stricts que ceux du lecteur frontend figé. Les cas malformés qu'il tolère ne sont pas déclarés compatibles. Les noms ou données contenus dans les métadonnées n'autorisent aucun accès à des fichiers, téléchargement de modèle ou code d'extension.

## Preuves

La [campagne locale](qualification/png-workflow-import.json) passe 99 tests Workflow et 74 tests Desktop, dont 22 nouveaux. Elle couvre les variantes textuelles, les métadonnées concurrentes, Unicode, corruption, compression, plafonds, annulation, conservation des champs et destination de sauvegarde. Le smoke avec vraie fenêtre et Host séparé crée un PNG puis réimporte son workflow, identique au document soumis.

La fixture [`with_metadata.png`](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/src/scripts/metadata/__fixtures__/with_metadata.png), 223 octets, est reproduite exactement en base64 dans les tests sous GPL-3.0-only. Son SHA-256 est `393da603c10b778089e4c48cae80c12469f03da18fbc87aa76879256e88ae1a9`. Le test charge son unique KSampler comme document ; il n'exécute pas de génération. Les autres fixtures sont construites en C# indépendamment du lecteur.
