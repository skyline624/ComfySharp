# Backlog de migration

Les critères complets sont dans [MIGRATION.md](MIGRATION.md). Aucun lot n'est clos par la seule présence d'un projet ou d'une classe.

| Lot | Livré actuellement | Travail nécessaire avant clôture |
|---|---|---|
| 0 | Validé : dépôt public indépendant, clone propre, Host/Desktop, tests et probe CPU en CI trois OS, fenêtres natives communiquant avec leur Host, archives autonomes construites | Critères du socle satisfaits ; installations et distributions finales suivies au lot 12 |
| 1 | Manifeste de 1 166 entrées, inscriptions intégrées rapprochées, 663 contrats de source, profils textuels et encodeurs CLIP, contrôles .NET | Finalisation des schémas/fournisseurs dynamiques, variantes de modèles/encodeurs/VAE/widgets, corpus, poids/hash, tolérances |
| 2 | Lecteur sûr initial, probe CPU et RTX3090 CUDA, gradients | Primitives complètes, SD1.5 réel, Linux/CUDA, macOS/MPS, médias et longs tests de ressources |
| 3 | Valeurs natives possédées, UI distincte et snapshots, collecte d'ancêtres, résultats libérables, validation, sorties indépendantes, listes, lazy, async, annulation | Expansion/sous-graphes, bloqueurs, hooks, caches/LRU/pression, concurrence des branches |
| 4 | JSON préservé, compilation conservatrice, éditions et canvas initial | Compilation complète, sous-graphes, bypass, reroutes, copier/coller, onglets/brouillons et fidélité exhaustive |
| 5 | API/file/historique UI/meta, ciblage OUTPUT_NODE et sessions WS, SQLite propre/réglages/prune | Contrats amont exacts, négociation/codecs complets, fichiers/profils/userdata, assets/références/imports complets |
| 6 | Dix dtypes safetensors CPU, inspection/hash, bruit natif, étape Euler, cinq générateurs et six traitements SIGMAS intégrés au Host ; tokenisation CLIP et encodeurs CPU/F32 expérimentaux avec chargement sûr et conditioning SD1/SDXL | Qualification numérique des encodeurs et vrais poids, intégration denoiser/VAE ; schedulers dépendant du modèle, samplers, formats historiques sûrs, détection, textual inversion, autres encodeurs/tokeniseurs, adapters et GPU |
| 7 | Rien annoncé compatible | Chaque famille/variante image avec vrais poids |
| 8 | Rien annoncé compatible | Vidéo, audio, 3D, codecs et traitements associés |
| 9 | Gradient et mise à jour SGD élémentaires | Nœuds d'entraînement, optimisateurs/losses complets, adapters, sauvegarde et réutilisation |
| 10 | Aucune quantification annoncée | Recettes quantifiées, kernels, offload, attention et mémoire avancée |
| 11 | Base native C#/XAML et PreviewAny texte sans persistance des sorties | PreviewAny Markdown et représentations restantes, widgets/outils locaux avancés, SDK et qualification UI |
| 12 | Aucune V1 publiée | Toutes preuves, installations propres, distributions, notices/SBOM/checksums |

Ordre suivant : rendre l'inventaire exhaustif et les corpus reproductibles ; porter la détection, la tokenisation, les schedulers/samplers et les composants SD1.5 avec vrais poids ; compléter le moteur et le document en parallèle ; avancer par famille dans la matrice. Les preuves Linux/CUDA et Mac/MPS attendent les machines prévues ; les tests CPU et le démarrage graphique en CI ont réussi et les autres travaux continuent.

Suivis du lot 2 après revue du bootstrap macOS : l'inspection en processus neuf sans libtorch et les points d'entrée tensoriels initiaux passent en CI ; remplacer les valeurs mémoire macOS non exploitables ; qualifier ou reconstruire le bridge avec une cible macOS 14 explicite avant publication des distributions natives.

Le calendrier complet sera établi à partir du lot 1 et des qualifications du lot 2. Aucune durée artificielle n'est déduite du nombre de fichiers ou des seuls nœuds déjà portés.
