# Aperçus PNG dans l'éditeur natif

Desktop affiche les fichiers retournés par `SaveImage` et `PreviewImage` dans leurs nœuds. Chaque lot présente une image, son nom et sa position, avec navigation précédente/suivante. Le chargement se fait à la demande ; les pixels calculés ne deviennent ni des widgets ni des données persistées du workflow.

## Parcours et durée de vie

Lors de la soumission, Desktop capture le prompt, les cibles et le document complet, y compris ses champs inconnus. Il transmet ce dernier dans `extra_data.extra_pnginfo.workflow`, que les [nœuds fichiers](IMAGE_FILE_NODES.md) incorporent aux PNG lorsque les métadonnées sont activées. La sauvegarde du document pendant une exécution ne change pas le snapshot soumis.

À la fin du travail, les descripteurs `filename/subfolder/type` proviennent de l'historique du Host. Le lecteur est lié à l'adresse et à la génération du processus qui a accepté le travail. Il consulte uniquement `/view` sur le Host local, sans suivre de redirection. Une ancienne génération ne peut pas récupérer les fichiers d'un nouveau Host sous le même descripteur.

Un objet appartenant au document possède la bitmap affichée et la requête en cours pour chaque nœud. La navigation annule l'ancienne requête et libère la bitmap remplacée. Une réponse tardive est rejetée même si le transport ignore l'annulation. Une modification du document, une nouvelle soumission, la fermeture de la fenêtre ou le redémarrage/arrêt du Host invalident les aperçus et libèrent leurs ressources. Le rechargement des vues et le passage entre onglets conservent l'aperçu valide ; le document reste son propriétaire. Aucune exécution n'est rejouée automatiquement.

Une erreur de lecture ou de décodage est affichée dans le nœud avec `Preview unavailable`. Le travail peut avoir correctement créé ses fichiers malgré une erreur ultérieure de prévisualisation. Les descripteurs non pris en charge sont refusés explicitement.

## Limites actuelles

Les aperçus utilisent les PNG originaux des répertoires `output` et `temp`. Le transfert est borné à 128 MiB par fichier, avec vérification du flux même en l'absence de `Content-Length` et une échéance de 30 secondes. Le header PNG est contrôlé avant le décodage natif ; les dimensions sont limitées à 16 777 216 pixels. La bitmap affichée conserve le rapport d'aspect et ne dépasse pas 512 pixels sur son grand côté. Une petite image n'est pas agrandie au décodage.

Ces limites portent sur chaque fichier ou aperçu, pas sur toute la mémoire du processus : un document peut contenir plusieurs nœuds avec leur propre bitmap et les bibliothèques natives peuvent allouer des buffers temporaires. L'affichage ne constitue pas un visualiseur pleine résolution. Les fonctions avancées de galerie, comparaison, masque, painter, vidéos/audio/3D, les conversions `/view` et les substitutions frontend de noms restent à porter. Aucun accès tensoriel ni libtorch n'est ajouté à Desktop.

## Vérification

La [campagne Windows locale](qualification/bitmap-previews.json) vérifie le décodage réel Skia sous Avalonia Headless, avec des pixels RGBA explicitement attendus. Les bitmaps remplacées sont effectivement inutilisables après libération. Les tests couvrent navigation tardive, invalidation du document, fermeture des onglets avec la fenêtre, snapshots de soumission, chemins HTTP encodés, refus d'adresses et réponses non prises en charge, taille du flux sans header et annulation de sa lecture.

Le smoke utilise une vraie fenêtre Avalonia et son processus Host séparé, dans un répertoire temporaire propre. Un graphe IMAGE de trois frames traverse SaveImage puis PreviewImage, produit six PNG, décode les deux aperçus, navigue jusqu'à la troisième frame et vérifie le workflow transmis dans l'historique. Il redémarre ensuite le Host et vérifie l'invalidation de l'ancien lecteur et la destruction des aperçus. Les fichiers temporaires sont supprimés après l'arrêt du processus.

Ces tests de primitives ne valident pas de génération avec poids préentraînés. La [CI du commit 7b172c1](qualification/bitmap-previews-ci-7b172c1.md) confirme cette interface sur les trois OS ; les distributions autonomes restent à qualifier. Le [réimport du workflow PNG](PNG_WORKFLOW_IMPORT.md) est ajouté dans la tranche suivante. La [CI précédente](qualification/image-file-nodes-ci-8556ab2.md) couvre les nœuds fichiers sur les trois systèmes, sans cette nouvelle interface ; ses gates numériques SD/CLIP restent ouvertes.
