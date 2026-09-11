# Backlog de migration

Les critères complets sont dans [MIGRATION.md](MIGRATION.md). Aucun lot n'est clos par la seule présence d'un projet ou d'une classe.

| Lot | Livré dans le socle initial | Travail nécessaire avant clôture |
|---|---|---|
| 0 | Dépôt indépendant, solution, Host et Desktop, tests, CI | Résultats CI trois OS et clone indépendant |
| 1 | Manifeste statique et vérificateur .NET | Exhaustivité dynamique/conditionnelle, schémas, corpus, poids/hash, tolérances |
| 2 | Lecteur sûr initial, probe CPU et RTX3090 CUDA, gradients | Primitives complètes, SD1.5 réel, Linux/CUDA, macOS/MPS, médias et longs tests de ressources |
| 3 | Validation, sorties indépendantes, listes, lazy initial, async, annulation | Contrat natif non JSON, expansion/sous-graphes, bloqueurs, hooks, caches/LRU/pression, concurrence des branches |
| 4 | JSON préservé, compilation conservatrice, éditions et canvas initial | Compilation complète, sous-graphes, bypass, reroutes, copier/coller, onglets/brouillons et fidélité exhaustive |
| 5 | API/file/historique/WS de base, SQLite propre/réglages/prune | Contrats amont exacts, négociation/codecs complets, fichiers/profils/userdata, assets/références/imports complets |
| 6 | Safetensors F32, bruit natif, étape Euler | Dtypes, formats historiques sûrs, détection, tokeniseurs, encodeurs, sampling et adapters complets |
| 7 | Rien annoncé compatible | Chaque famille/variante image avec vrais poids |
| 8 | Rien annoncé compatible | Vidéo, audio, 3D, codecs et traitements associés |
| 9 | Gradient et mise à jour SGD élémentaires | Nœuds d'entraînement, optimisateurs/losses complets, adapters, sauvegarde et réutilisation |
| 10 | Aucune quantification annoncée | Recettes quantifiées, kernels, offload, attention et mémoire avancée |
| 11 | Base native C#/XAML | Widgets/outils locaux avancés, SDK et qualification UI |
| 12 | Aucune V1 publiée | Toutes preuves, installations propres, distributions, notices/SBOM/checksums |

Ordre suivant : stabiliser le socle intégré ; rendre l'inventaire exhaustif et les corpus reproductibles ; étendre le contrat d'exécution aux valeurs natives propriétaires ; porter le chargement et les composants SD1.5 avec vrais poids ; compléter le moteur et le document en parallèle ; avancer par famille dans la matrice. Les preuves Linux et Mac attendent les machines prévues, les autres travaux continuent.

Le calendrier complet sera établi à partir du lot 1 et des qualifications du lot 2. Aucune durée artificielle n'est déduite du nombre de fichiers ou des seuls nœuds déjà portés.
