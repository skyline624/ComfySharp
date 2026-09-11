# État du port — 0.1.0-dev

État initial au 11 septembre 2026. **La migration complète n'est pas terminée et aucune famille de modèles n'est annoncée compatible.**

| Composant | Disponible | Limites |
|---|---|---|
| Dépôt/socle | Dépôt public indépendant, clone propre vérifié, .NET 10, verrous, GPLv3 ; base à 177 tests sur les trois OS, nouvelle tranche à 250 tests Windows | Nouvelle campagne CI, démarrage graphique réel Linux/macOS et installations propres à qualifier |
| Moteur | Valeurs JSON/natives/listes/maps, propriété partagée déterministe, résultats libérables, validation, cibles indépendantes, listes/repeat-last, async, lazy, annulation et mémoïsation par job | Expansion, bloqueurs, sous-graphes, caches persistants et offload à porter |
| Nœuds | 5 Primitive*, 6 fonctions String/JSON, ComfyNotNode, ComfySwitchNode | 13 identifiants sur le catalogue ; aucun nœud modèle |
| Documents | Import/export préservant champs inconnus, édition/undo, compilation conservatrice | Cas non pris en charge refusés explicitement ; pas de compilateur frontend complet |
| Desktop | Avalonia/Nodify, ouvertures/sauvegardes, canvas initial et Host séparé supervisé | Fidélité UI avancée, widgets et outils médias non terminés |
| API locale | Health/catalogue/prompt/file/historique/jobs/annulation, événements WS de base | Contrats et erreurs pas encore entièrement compatibles, négociation/aperçus avancés à intégrer |
| Stockage | Base distincte, migrations up/down, réglages, enregistrement d'assets et prune par marquage | Références/tags/imports/userdata/profils complets manquants ; API assets initiale propre au port |
| Poids | Validation safetensors, inspection sans libtorch, SHA-256 sur le même fichier ouvert, chargement CPU F32/F16/BF16/F64/I8/U8/I16/I32/I64/BOOL | U16/U32/U64 en métadonnées seulement ; quantification, formats historiques, détection et architectures à porter |
| Calcul | Bruit natif par générateur, primitive Euler, opérations et gradients CPU/CUDA réels | Pas de parcours SD1.5 ; aucun modèle entraîné ou génératif exécuté |
| Manifeste | 1 160 entrées ; 652 candidats locaux, 275 distants exclus ; schémas et sources réconciliés, contrôles .NET | 281 expressions de schéma indéterminées ; catalogue global, corpus et tolérances encore incomplets |

Preuves initiales : tests unitaires du moteur, des documents, du Host/stockage et des primitives natives ; probe Windows CPU puis RTX 3090 CUDA 12.8 avec gradient, mise à jour SGD, convolution, attention, matmul, transfert de bruit et durée de vie des vues. Les rapports de commandes sont tenus dans le suivi local puis synthétisés dans la documentation de qualification. Les répétitions courtes ne prouvent pas l'absence de fuite et ne valident aucune famille.

Linux/NVIDIA et Apple Silicon/MPS ne sont pas validés. Le contrôle `ComfySharp.Catalog ... --release` doit échouer tant que les lignes obligatoires restent ouvertes. Le [plan](MIGRATION.md) et [l'objectif complet](OBJECTIF.md) demeurent inchangés.
