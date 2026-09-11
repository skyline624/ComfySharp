# ComfySharp — Plan approuvé de migration de ComfyUI vers C#

Plan approuvé le 11 septembre 2026. Ce document décrit la cible obligatoire, pas les fonctions déjà disponibles. L'avancement réel est publié dans [STATUS.md](STATUS.md) et la [matrice](capabilities/README.md). Les versions intermédiaires restent en 0.x.

## 1. Objectif et périmètre de la V1

Créer ComfySharp dans le sous-dossier `ComfySharp` du projet ComfyUI, avec un dépôt Git indépendant publié dans le nouveau dépôt public `skyline624/ComfySharp`. Le dépôt enfant doit se cloner, se compiler et fonctionner sans le dépôt parent.

La V1 couvre **tout le catalogue local intégré** : génération d'images, vidéo, audio, 3D, traitements associés, workflows et entraînement intégré. Moteur et interface sont réécrits en C#, avec des bibliothèques natives. Le prototype SD1.5 est un jalon technique intermédiaire ; il ne remplace pas cet objectif.

Références figées :

- [Backend 1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a](https://github.com/comfy-org/ComfyUI/tree/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a).
- [Frontend 1.51.10, e7d1c7fc6823e330fdab524610b0000394cb1dbc](https://github.com/Comfy-Org/ComfyUI_frontend/tree/e7d1c7fc6823e330fdab524610b0000394cb1dbc).

Les évolutions suivantes seront traitées par mises à niveau explicites après migration.

| Domaine | Engagement V1 |
|---|---|
| Windows | Windows 11 x64, CPU et NVIDIA/CUDA |
| Linux | Ubuntu 24.04 x64, X11/XWayland, CPU et NVIDIA/CUDA |
| macOS | macOS 14+, Apple Silicon, CPU et Metal/MPS |
| Compatibilité | Fidélité fonctionnelle et tolérances numériques documentées |
| Interface | Application native C#/XAML, fonctions locales de l'éditeur reproduites |
| Services payants distants | Exclus |
| Extensions Python/JavaScript tierces | Préservation dans les documents et diagnostic ; exécution après portage |
| ComfyUI-Manager | Exclu ; SDK d'extensions C# prévu |
| Python | Aucune dépendance dans l'application distribuée ni pour charger les modèles annoncés compatibles |

Les machines Linux/NVIDIA et Mac Apple Silicon seront fournies ultérieurement. Aucune infrastructure cloud payante ne sera engagée.

## 2. Dépôt indépendant et organisation

1. Créer `ComfySharp` dans le projet, sans écraser un dossier existant.
2. Ajouter `/ComfySharp/` à l'exclusion locale `.git/info/exclude` du parent.
3. Initialiser le dépôt enfant, branche d'intégration `main`.
4. Utiliser des branches `codex/<lot>-<fonctionnalite>`.
5. Créer documentation, solution et configuration initiale ; vérifier le bon dépôt avant toute opération Git.
6. Effectuer le premier commit, créer le dépôt GitHub public `skyline624/ComfySharp`, configurer `origin` et pousser.
7. Si le dossier ou le dépôt distant existe, vérifier contenu et identité avant réutilisation. Ne rien écraser.

Le dépôt enfant n'est pas un sous-module. Le parent conserve son historique et son remote.

```text
ComfySharp/
├── ComfySharp.slnx
├── global.json
├── Directory.Build.props
├── Directory.Packages.props
├── README.md
├── LICENSE
├── THIRD_PARTY_NOTICES.md
├── AGENTS.md
├── src/
├── native/
├── tests/
├── tools/
├── docs/
└── .github/workflows/
```

`src` contient application, moteur, nœuds, inférence, médias et stockage C#. `native` contient bridges C ABI et adaptations natives. `tests` couvre contrats, workflows, calcul, interface et installation. `tools` accueille les outils .NET de diagnostic, inventaire, comparaison et packaging. `docs` contient plan, architecture, backlog, compatibilité et procédures. La CI compile, teste et construit les distributions.

Premier commit : présentation du port indépendant, ce plan, références et provenance amont, matrice et backlog, GPLv3 et notices, solution compilable sans fonctions factices annoncées, dépendances centralisées verrouillées et exclusions des modèles, sorties, données personnelles, secrets, caches et binaires. Les audits utiles sont adaptés avec des liens publics figés, sans chemins absolus de la machine.

## 3. Architecture et contrats

| Besoin | Base de qualification |
|---|---|
| Runtime | .NET 10, SDK initial 10.0.300 |
| Interface | Avalonia 12.0.5, C#/XAML, MVVM |
| Canvas | Nodify.Avalonia 2.0.0 derrière la présentation |
| Tenseurs | TorchSharp 0.107.0, libtorch 2.10.0.0 |
| NVIDIA | Distribution native CUDA 12.8 compatible |
| Apple Silicon | libtorch MPS, opérations réelles sur matériel |
| API locale | ASP.NET Core, HTTP et WebSocket |
| Sérialisation | System.Text.Json |
| Stockage | Microsoft.Data.Sqlite et migrations SQL versionnées |
| Images/médias | SkiaSharp, FFmpeg natif, calcul tensoriel |
| Rendu avancé | EGL/OpenGL ES natif avec ANGLE |
| Tests | xUnit, Avalonia Headless, scénarios matériels |

[TorchSharp](https://www.nuget.org/packages/TorchSharp/0.107.0), [Nodify.Avalonia](https://www.nuget.org/packages/Nodify.Avalonia/2.0.0) et [Avalonia](https://www.nuget.org/packages/Avalonia/12.0.5) établissent les versions, pas la compatibilité du catalogue. Les opérations manquantes nécessitent des bindings ou du code natif explicite ; une difficulté ne retire pas silencieusement une fonction.

Deux processus .NET : **Desktop** possède document graphique, édition, affichage et supervision ; **Host** possède API, file, exécution, modèles, tenseurs et stockage. Desktop lance/surveille Host, conserve les documents après crash natif, signale l'échec actif et permet le redémarrage sans rejouer automatiquement les travaux. Seuls prompts, états, diagnostics, métadonnées et aperçus franchissent la frontière.

| Module | Responsabilité |
|---|---|
| Contracts | Prompts, schémas, événements, diagnostics |
| Core | Validation, ordonnanceur, listes, lazy, expansion, cache, annulation |
| Workflow | Documents, édition, compilation en prompt |
| Inference | Modèles, tokenisation, sampling, tenseurs, poids, mémoire |
| Nodes | Catalogue aux identifiants amont |
| Media | Images/audio/vidéo/3D/shaders/métadonnées |
| Storage | Réglages, profils, fichiers, assets |
| Host | Services applicatifs, HTTP/WebSocket |
| Desktop | Avalonia et supervision |
| PluginSdk | Extensions/widgets .NET versionnés |

Contrats obligatoires : `class_type`, noms d'entrées, ordre des sorties, schémas ; séparation document/prompt ; workflows 0.4 et 1, champs inconnus, sous-graphes, IDs composés ; distinction littéral/connexion/liste d'exécution ; événements, erreurs et codecs d'aperçus ; API locale de soumission, file, historique, jobs, fichiers, catalogue et assets. OpenAPI mélange des surfaces cloud : ne pas générer le serveur uniquement depuis ce fichier. Nœuds asynchrones avec capacités explicites de validation personnalisée, empreinte de cache, lazy et expansion.

Ressources : libération native déterministe, GC seul insuffisant pour la VRAM ; propriété partagée des storages/vues ; un prompt actif initialement, concurrence seulement si supportée ; replis CPU explicites et testés. Base/données propres à ComfySharp, import depuis une copie ComfyUI, aucune migration en place. Jobs/historique en mémoire comme la référence. Aucun téléchargement automatique de modèles, suivi d'usage ou appel distant dans le fonctionnement local.

## 4. Lots et critères de passage

### Lot 0 — Dépôt public et socle compilable

Créer structure/dépôt, compilation reproductible, dépendances verrouillées, tests, application Avalonia minimale et Host démarrable, CI Windows/Linux/macOS.

**Passage :** un clone indépendant compile ; Desktop démarre et communique avec Host ; le dépôt public contient uniquement le nouveau projet.

### Lot 1 — Référentiel complet et corpus

Transformer l'audit en manifeste exhaustif du catalogue local : nœuds, variantes, encodeurs, VAE, formats, samplers, schedulers, adapters, widgets, entraînement. Inclure inscriptions dynamiques/conditionnelles ; un import absent ne masque pas une capacité. Associer sources, paramètres, dépendances et scénarios à chaque capacité. Créer fixtures de workflows, bruit, tenseurs, médias, gradients. Fixer hashes des poids et tokeniseurs. Verrouiller les tolérances avant acceptation.

**Passage :** chaque fonction locale a une ligne et un scénario. Le laboratoire Python original est séparé ; l'application et la suite .NET distribuée n'en dépendent pas.

### Lot 2 — Qualification du runtime natif

Vérifier allocation, vues, transferts, RNG, mathématiques, libération, attention, convolutions spatiales/temporelles, audio, géométrie, autograd et mise à jour d'adapter. Réaliser un parcours SD1.5 réel sur RTX 3090. Exécuter les mêmes contrôles sur Linux/CUDA et macOS/MPS. Prototyper médias et surfaces natives.

**Passage :** calculs/gradients requis, traduction des erreurs natives et répétitions sans fuite. [La présence de MPS dans TorchSharp](https://github.com/dotnet/TorchSharp/blob/main/src/TorchSharp/Torch.cs) ne remplace pas des opérations réelles sur matériel.

### Lot 3 — Moteur de workflows complet

Registre/schémas unifiés ; validation avant file ; priorités, exécution partielle, sorties indépendantes ; listes avec répétition du dernier élément et concaténation ; lazy, expansion, sous-graphes, cycles dynamiques ; async, bloqueurs, erreurs, annulation ciblée ; caches objets/résultats puis LRU/pression RAM ; identités réelles/d'affichage.

**Passage :** références du moteur, annulation async, éviction avec consommateurs actifs et isolation des sous-graphes passent.

### Lot 4 — Documents, compilation, éditeur de base

Import/export workflow 0.4/1 et prompts API ; conservation des inconnus ; compilation fidèle virtual/mute/bypass/listes/widgets/sous-graphes ; canvas zoom/pan/sélection/connexions/groupes/reroutes ; copier/coller, duplication, navigation, undo/redo ; onglets, brouillons, sauvegardes.

**Passage :** deux cycles d'import/export sans perte contractuelle ; prompts normalisés égaux aux références. Document/compilateur testables sans fenêtre, indépendants de Nodify.

### Lot 5 — API, fichiers, persistance

Endpoints locaux et alias compatibles ; négociation WS, JSON, trames binaires ; file/jobs/annulation/historique/fichiers/profils ; réglages/userdata/chemins annotés ; SQLite assets/références/tags/métadonnées/ingestion/seeding/prune ; import vers base séparée.

**Passage :** contrats HTTP/WS ; annuler un travail ne touche pas son successeur ; migrations réversibles ; prune marque sans suppression physique.

### Lot 6 — Fondations communes des modèles

Lecteurs safetensors et formats historiques locaux ; détection des architectures, normalisation des clés et composants ; BPE/SentencePiece et pondération ComfyUI ; CLIP/T5 et toutes familles texte/multimodales de référence ; formats latents, conditioning, CFG, masques ; tous samplers/schedulers ; clones, LoRA/adapters, patches, hooks, offload initial.

**Passage :** tokens/masques/métadonnées exacts ; encodeurs/prédictions/étapes de sampling dans les tolérances fixées. Aucun chargement par exécution arbitraire du contenu des checkpoints.

### Lot 7 — Catalogue image complet

Ordre : SD1.5/SD2 ; SDXL/refiner/variantes ; cascades ; DiT/flow et familles image récentes ; contrôles/conditioning visuel/upscalers/analyse ; tous traitements locaux associés.

**Passage :** workflow complet avec vrais poids pour chaque famille et variante fonctionnelle. Le parent d'une architecture ne valide pas automatiquement ses variantes.

### Lot 8 — Vidéo, audio, 3D

Après fondations, trois chantiers parallèles :

| Chantier | Travail | Validation |
|---|---|---|
| Vidéo | Modèles temporels, VAE causaux, références, caches, audiovisuel, codecs | Frames, continuité, dimensions, synchronisation, mémoire |
| Audio | Encodeurs, latents, vocodeurs, resampling, musique, fichiers | Fréquence, durée, canaux, précision, écoute |
| 3D | Modèles, géométrie, textures, multivues, traitements, export | Repères, échelles, géométrie, textures, fichiers |

Une prévisualisation seule ne valide pas une génération ou un traitement.

### Lot 9 — Entraînement intégré

Primitives dès lot 2, fondations lot 6. Porter `TrainLoraNode`, `LoraModelLoader`, `SaveLoRA`, `LossGraphNode` ; Adam/AdamW/SGD/RMSprop ; MSE/L1/Huber/SmoothL1 ; accumulation/batches/buckets/multirésolution/précision mixte ; checkpointing/offload/adapters trainables ; sauvegarde/rechargement compatibles.

**Passage :** gradients et mise à jour comparés sur mini-lot figé ; entraînement court, sauvegarde, rechargement et réutilisation en génération. Ne pas ajouter un entraînement universel absent de la référence.

### Lot 10 — Quantification et mémoire avancée

FP8/FP4/INT8 présents dans la référence, layouts/scales/kernels ; attention, chargement partiel, transferts, mémoire épinglée, préchargement ; optimisations natives sans wrappers Python. Distinguer format chargé, calcul accéléré et déquantification.

**Passage :** précision/mémoire par format matériellement pertinent. Déquantification correcte n'est pas accélération native FP4/FP8.

### Lot 11 — Interface avancée et SDK C#

Tous widgets dont DynamicCombo/Autogrow/MatchType ; masque/painter/compositor/courbes/outils spatiaux ; audio/vidéo/3D/GLSL ; workflows incorporés aux médias ; recherche/templates/galerie/réglages/raccourcis/localisation/présentation ; SDK nœuds/widgets avec manifeste versionné.

Shaders/compositor via natif, [ANGLE](https://github.com/google/angle) comme base GLES Windows/Linux/Metal.

**Passage :** chaque entrée du catalogue possède édition et sérialisation fonctionnelles ; extensions inconnues conservées et signalées.

### Lot 12 — Qualification et publication V1

Fermer toutes lignes obligatoires ; installations propres/mises à jour/imports ; distributions autonomes `win-x64`, `linux-x64`, `osx-arm64` ; notices, SBOM, checksums, installation ; publier capacités et matériel validés. Premières distributions portables, explicitement non signées si applicable. Aucun achat automatique de signature ou location matérielle.

## 5. Tests et définition de terminé

| Niveau | Contenu | Où |
|---|---|---|
| Rapide | JSON, documents, validation, ordonnanceur, commandes UI | Chaque changement, trois OS |
| UI sans fenêtre | Arbre visuel, widgets, commandes, saisies | CI Avalonia Headless |
| Élémentaire | Opérations, gradients, lecteurs, tokenisation | CPU CI, GPU postes dédiés |
| Intégration | Workflows réels avec poids identifiés par famille | Campagnes matérielles |
| Ressources | Répétitions, annulation, offload, modèles | CPU/GPU de référence |
| Installation | Démarrage/parcours sans Python ni Node | Environnements propres |

[Headless](https://docs.avaloniaui.net/docs/testing/setting-up-the-headless-platform) ne remplace pas fenêtres, codecs et GPU réels. Runners GPU auto-hébergés : uniquement révisions approuvées ; contributions externes en CI publique isolée.

Statuts séparés pour chaque capacité : **à porter**, **implémentée**, **vérifiée unitairement**, **validée en workflow réel**, **validée sur plateformes requises**. Chaque preuve précise commit, SHA-256 des poids, dtype, backend, matériel, paramètres et résultats. Absence de poids ou matériel n'est jamais réussite. Poursuivre le travail indépendant pendant l'attente de Linux/Mac.

Publication V1 si et seulement si :

1. Tout le catalogue local implémenté, exclusions distantes documentées.
2. Formats, schémas et workflows contractuels pris en charge.
3. Preuves réelles de chaque famille sur les configurations requises.
4. Entraînement et réutilisation d'adapters validés.
5. Tolérances numériques verrouillées respectées.
6. Aucune fuite persistante après erreurs, annulations, changements de modèles.
7. Interface locale entièrement C#/XAML.
8. Trois distributions sans Python, pip, Node ou frontend JavaScript.
9. Dépôt public avec sources, notices, compilation et matrice.
10. Aucun nœud factice, test ignoré ou scénario seulement synthétique pour annoncer une famille complète.

Les 0.x annoncent leur couverture réelle. L'ancienne estimation MVP ne s'applique pas : le calendrier de la V1 complète dépend du manifeste lot 1 et des qualifications lot 2.
