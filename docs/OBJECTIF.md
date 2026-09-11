# Objectif global : publier une V1 complète de ComfySharp

Livrer une application native C#/XAML autonome qui reproduit tout le catalogue **local intégré** et les fonctions locales de l'éditeur des deux références figées dans [MIGRATION.md](MIGRATION.md), puis publier ses sources et ses trois distributions dans `skyline624/ComfySharp`.

Le travail déjà commencé continue ; ni le dépôt initial, ni l'éditeur minimal, ni un prototype SD1.5 ne constituent l'achèvement de cet objectif.

## Résultats à obtenir

1. Un dépôt indépendant, compilable depuis un clone propre, sous GPLv3, avec dépendances verrouillées, provenance, notices et CI.
2. Un manifeste exhaustif des fonctions locales, y compris variantes et inscriptions conditionnelles, relié au backlog et aux preuves de validation.
3. Un moteur C# fidèle : validation, listes, lazy, expansion, caches, erreurs, ordonnanceur, annulation ciblée et ressources natives maîtrisées.
4. Une interface native complète : documents 0.4/1 conservés sans perte, édition/compilation des graphes, widgets, outils image/audio/vidéo/3D et SDK .NET.
5. Une API HTTP/WebSocket locale compatible et un stockage ComfySharp distinct, sans modifier les données ComfyUI originales.
6. Les modèles et traitements image, vidéo, audio et 3D, le sampling, les adapters, les formats de quantification et l'entraînement intégré réellement exécutables.
7. Des distributions autonomes `win-x64`, `linux-x64`, `osx-arm64`, sans Python, pip, Node ni frontend JavaScript, avec guides, inventaire des composants et checksums.

## Preuves obligatoires

- Chaque capacité obligatoire est implémentée et testée ; chaque famille possède un workflow réel avec poids et ressources identifiés par SHA-256.
- Les résultats respectent les tolérances numériques verrouillées avant leur acceptation ; les entraînements vérifient gradients, mise à jour, sauvegarde et réutilisation.
- Les fonctions sont qualifiées sur Windows 11 CPU/CUDA, Ubuntu 24.04 CPU/CUDA et macOS 14+ Apple Silicon CPU/MPS, sur matériel réel lorsque nécessaire.
- Erreurs, annulations et changements de modèles ne causent pas de fuite persistante ; les installations propres démarrent et exécutent leurs parcours annoncés.
- Aucun test ignoré, matériel absent, nœud factice ou exemple uniquement synthétique ne remplace une preuve de compatibilité.

## Contraintes de conduite

Exécuter les lots 0 à 12, maintenir l'avancement et publier des jalons 0.x avec leur couverture réelle. Poursuivre les parties indépendantes pendant l'attente des machines Linux/NVIDIA et Mac fournies ultérieurement. Aucun cloud payant, achat de matériel ou signature, téléchargement automatique de modèle, télémétrie ou publication de données personnelles. Services payants distants et ComfyUI-Manager exclus ; extensions tierces conservées et diagnostiquées, exécutées après portage.

**Terminé signifie :** tous les critères V1 du plan sont prouvés, les trois distributions fonctionnent et la V1 documentée est publiée. Le détail contractuel reste [MIGRATION.md](MIGRATION.md).
