# État du port — 0.1.0-dev

État initial au 11 septembre 2026. **La migration complète n'est pas terminée et aucune famille de modèles n'est annoncée compatible.**

| Composant | Disponible | Limites |
|---|---|---|
| Dépôt/socle | Dépôt public indépendant, clone propre vérifié, .NET 10, verrous, GPLv3 ; 177 tests et probe CPU réussis sur les trois OS | Démarrage graphique réel Linux/macOS et installations propres à qualifier |
| Moteur | Validation/diagnostics, cibles indépendantes, listes/repeat-last, async, lazy initial, annulation et mémoïsation isolée par job | Valeurs actuellement JSON ; expansion, bloqueurs, sous-graphes et caches persistants à porter |
| Nœuds | 5 Primitive*, 6 fonctions String/JSON, ComfyNotNode, ComfySwitchNode | 13 identifiants sur le catalogue ; aucun nœud modèle |
| Documents | Import/export préservant champs inconnus, édition/undo, compilation conservatrice | Cas non pris en charge refusés explicitement ; pas de compilateur frontend complet |
| Desktop | Avalonia/Nodify, ouvertures/sauvegardes, canvas initial et Host séparé supervisé | Fidélité UI avancée, widgets et outils médias non terminés |
| API locale | Health/catalogue/prompt/file/historique/jobs/annulation, événements WS de base | Contrats et erreurs pas encore entièrement compatibles, négociation/aperçus avancés à intégrer |
| Stockage | Base distincte, migrations up/down, réglages, enregistrement d'assets et prune par marquage | Références/tags/imports/userdata/profils complets manquants ; API assets initiale propre au port |
| Poids | Validation safetensors, chargement F32 sûr | Autres dtypes/chargement historique/détection/architectures à porter |
| Calcul | Bruit natif par générateur, primitive Euler, opérations et gradients CPU/CUDA réels | Pas de parcours SD1.5 ; aucun modèle entraîné ou génératif exécuté |
| Manifeste | 1 106 entrées statiques, contrôles .NET | Exhaustivité non établie ; corpus et tolérances à verrouiller |

Preuves initiales : tests unitaires du moteur, des documents, du Host/stockage et des primitives natives ; probe Windows CPU puis RTX 3090 CUDA 12.8 avec gradient, mise à jour SGD, convolution, attention, matmul, transfert de bruit et durée de vie des vues. Les rapports de commandes sont tenus dans le suivi local puis synthétisés dans la documentation de qualification. Les répétitions courtes ne prouvent pas l'absence de fuite et ne valident aucune famille.

Linux/NVIDIA et Apple Silicon/MPS ne sont pas validés. Le contrôle `ComfySharp.Catalog ... --release` doit échouer tant que les lignes obligatoires restent ouvertes. Le [plan](MIGRATION.md) et [l'objectif complet](OBJECTIF.md) demeurent inchangés.
