> Audit statique antérieur à l’implémentation. Les choix définitifs et le périmètre V1 sont fixés par [le plan approuvé](../MIGRATION.md). Les observations ci-dessous ne sont pas des résultats de tests du port C#.

# API, données et réécriture de l’interface en C#

## Périmètre et références de l’audit

Analyse statique du 11 septembre 2026. Cible confirmée : application sans Python, interface également réécrite en C#, bibliothèques natives autorisées.

| Source | Référence auditée | Localisation |
| --- | --- | --- |
| ComfyUI backend | `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a` | `le clone de référence` |
| ComfyUI frontend | tag `v1.51.10`, commit `e7d1c7fc6823e330fdab524610b0000394cb1dbc` | `le clone frontend de référence` |

Le tag frontend correspond au paquet demandé par [requirements.txt](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/requirements.txt#L1). Son dépôt a été cloné séparément avec une profondeur de 1, après vérification du tag distant. Le backend n’inclut pas les sources Vue/TypeScript complètes : [README.md](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/README.md#L402) et [frontend_management.py](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/app/frontend_management.py#L232) décrivent la distribution de fichiers statiques par paquet Python.

Inventaire du frontend, distinct des métriques backend : **5 593 fichiers suivis**, dont **2 470 fichiers `.ts` dans `src/`**, **709 `.vue`**, **1 173 `.test.ts` dans `src/`** inclus dans les 2 470 fichiers TypeScript, **286 spécifications Playwright** dans `browser_tests/` et **147 fixtures JSON** dans `browser_tests/assets/`. Ces nombres sont des fichiers, pas des cas de test ni des fonctions à porter. Le dépôt contient également le site commercial et des fonctions cloud : tout cet inventaire ne relève pas de l’application locale.

L’audit frontend est ciblé sur les contrats qui conditionnent une conversion correcte : schémas, compilation du workflow en prompt, sous-graphes, widgets, extensions, persistance et spécifications UI. Il ne constitue pas une lecture de chacun des 5 593 fichiers. Aucun paquet installé, aucun build lancé et aucun test exécuté. Les résultats ci-dessous sont des observations du code et des recommandations d’architecture, pas une preuve de compatibilité obtenue à l’exécution.

## Conclusion technique

Le serveur HTTP, les fichiers utilisateur, les profils, les catalogues et SQLite sont transposables en C# avec un risque maîtrisable. L’interface exige un chantier distinct et important : le frontend actuel gère une partie de la sémantique des workflows avant leur soumission au moteur. Un éditeur de boîtes reliées ne remplace donc pas à lui seul ComfyUI.

La cible devrait isoler trois représentations : **document visuel conservé sans perte**, **modèle d’édition C#**, **graphe exécutable normalisé**. Un compilateur explicite relie les deux derniers. Les API HTTP/WebSocket deviennent un adaptateur de compatibilité ; une UI C# peut appeler les mêmes services applicatifs directement, ou passer par un processus local si l’isolation du moteur est retenue. Le choix des processus appartient à l’architecture globale.

La compatibilité des workflows, celle des endpoints et celle des extensions sont trois engagements différents. Un nœud personnalisé Python et une extension JavaScript qui modifie des objets LiteGraph ne peuvent pas fonctionner tels quels dans cette cible C#.

## 1. Cartographie du serveur local

### Composition

[PromptServer](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/server.py#L215) compose le gestionnaire utilisateur, les modèles, les extensions, les sous-graphes, les remplacements de nœuds, les routes internes, la file de prompts et le canal des événements. Il crée une application `aiohttp`, la table des WebSockets et la racine frontend.

[add_routes](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/server.py#L1220) ajoute les routes des gestionnaires et un sous-service `/internal`. Les routes HTTP non statiques de sa table sont aussi enregistrées avec le préfixe `/api`. Il faut conserver les chemins réellement utilisés par les clients : les endpoints historiques `/prompt`, `/queue`, etc. possèdent un alias `/api/...`, alors que les routes jobs portent déjà `/api/jobs` dans leur déclaration. La boucle ajoute mécaniquement aussi un double préfixe à ces dernières ; ce détail historique ne justifie pas d’exposer ces doublons comme nouvelle API publique.

Les routes des assets sont enregistrées directement sur l’application et non dans cette boucle. Les extensions peuvent ajouter des routes Python et des fichiers JavaScript, donc le nombre d’endpoints d’un environnement avec custom nodes dépend de ses plugins.

### Surfaces HTTP à porter

| Surface | Routes principales | Comportement / contrainte |
| --- | --- | --- |
| Exécution | `GET/POST /prompt` | Statut global ; validation et soumission d’un graphe API, priorité, identifiant et métadonnées |
| File | `GET/POST /queue` | Listes running/pending ; vider ou supprimer des éléments |
| Historique | `GET /history`, `GET /history/{prompt_id}`, `POST /history` | Résultats par prompt, pagination historique, suppression et purge |
| Jobs | `GET /api/jobs`, `GET /api/jobs/{job_id}` | Projection normalisée de la file et de l’historique, filtres/statuts/tri/pagination |
| Annulation | `POST /interrupt`, `POST /api/jobs/{job_id}/cancel`, `POST /api/jobs/cancel` | Interruption ciblée/globale ; annulation idempotente simple ou par lot |
| Mémoire | `POST /free` | Demandes de libération mémoire et déchargement des modèles transmises à la file |
| Catalogue nœuds | `GET /object_info`, `GET /object_info/{node_class}` | Métadonnées d’entrées/sorties nécessaires à l’éditeur et à la validation |
| Catalogue modèles | `GET /models`, `/models/{folder}`, `/embeddings` | Catégories, noms relatifs et embeddings |
| Modèles expérimentaux | `/experiment/models`, `/experiment/models/{folder}`, `/experiment/models/preview/{folder}/{path_index}/{filename}` | Fichiers avec métadonnées et images de prévisualisation |
| Fichiers | `POST /upload/image`, `POST /upload/mask`, `GET /view`, `GET /view_metadata/{folder_name}` | Multipart, masques, images/canaux et métadonnées safetensors |
| Profils | `GET/POST /users` | Profils locaux, création et choix du contexte utilisateur |
| Réglages | `GET/POST /settings`, `/settings/{id}` | Valeurs JSON par profil ; mise à jour par fusion pour `/settings` |
| Documents | `GET /userdata`, `GET /v2/userdata`, `GET/POST/DELETE /userdata/{file}`, `POST /userdata/{file}/move/{dest}` | Répertoires, fichiers arbitraires, sauvegarde/renommage des workflows |
| Découverte UI | `/extensions`, `/workflow_templates`, `/i18n`, `/global_subgraphs`, `/global_subgraphs/{id}`, `/node_replacements` | Catalogues complémentaires et migrations de nœuds |
| Système | `/system_stats`, `/features` | Capacités du serveur et état du matériel/runtime |
| Interne | `/internal/logs`, `/logs/raw`, `/logs/subscribe`, `/folder_paths`, `/files/{directory_type}` sous le préfixe interne | Console et informations locales, contrat explicitement instable |
| Assets | `/api/assets`, `/api/assets/{id}`, `/content`, `/tags`, `/from-hash`, `/hash/{hash}`, `/api/tags`, seeding/prune | Catalogue SQLite facultatif, fichiers, tags, métadonnées et indexation |

Références : [server.py](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/server.py#L337), [jobs](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/server.py#L821), [UserManager](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/app/user_manager.py#L123), [AppSettings](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/app/app_settings.py#L36), [ModelFileManager](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/app/model_manager.py#L30), [routes assets](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/app/assets/api/routes.py#L295), [routes internes](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/api_server/routes/internal/internal_routes.py#L20). Le [README interne](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/api_server/routes/internal/README.md#L1) exclut une garantie de stabilité.

### Pourquoi `openapi.yaml` ne suffit pas

Le fichier décrit aussi des mécanismes cloud qui ne sont pas implémentés par le serveur local audité : clés API/JWT Firebase/cookie de session, billing, workflows versionnés et publication, tâches distantes, réponses `/view` redirigées vers des URLs GCS. Voir [authentification déclarée](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/openapi.yaml#L1596), [billing](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/openapi.yaml#L2602), [workflows versionnés](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/openapi.yaml#L4917) et [réponse de visualisation](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/openapi.yaml#L4862).

Il faut produire une **spécification locale vérifiée à partir des handlers et des tests**, puis générer éventuellement DTO/clients C# depuis cette spécification. Générer directement tout le serveur depuis le YAML actuel ajouterait des endpoints hors périmètre et pourrait introduire de fausses exigences d’authentification. À l’inverse, ce YAML ne décrit pas seul toute la négociation et les trames WebSocket locales.

## 2. Contrat de soumission, jobs et annulation

[POST /prompt](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/server.py#L1072) reçoit notamment `prompt`, `client_id`, `extra_data`, une priorité `number` ou l’indicateur `front`, `partial_execution_targets`, et éventuellement `prompt_id`. Si le client fournit ce dernier, il doit être un UUID dans sa forme canonique minuscule avec tirets ; sinon le serveur le génère. Les timestamps de création sont des millisecondes Unix.

Avant insertion, le serveur applique les remplacements de nœuds puis appelle la validation du moteur. Réponse positive : `{prompt_id, number, node_errors}`. Une validation invalide renvoie HTTP 400 avec `{error, node_errors}`. Les données sensibles sont retirées de `extra_data` et stockées à part ; [la réponse de file les élimine](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/server.py#L69). Cette séparation doit survivre à la conversion et aux logs.

Le DTO interne de file reste un tuple Python : priorité, identifiant, graphe, métadonnées, sorties à exécuter et informations sensibles. Il convient de le remplacer par un type C# nommé, tout en conservant la forme JSON historique dans l’adaptateur HTTP si des clients existants sont pris en charge.

[GET /api/jobs](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/server.py#L891) combine la file en cours avec l’historique en mémoire. Il ne lit pas une table de jobs SQLite. Les statuts sont `pending`, `in_progress`, `completed`, `failed`, `cancelled`, définis dans [jobs.py](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_execution/jobs.py#L23). La projection des sorties différencie résumé de liste, détail complet, nombre de fichiers et nombre d’éléments prévisualisables. Elle extrait aussi les dates et erreurs des événements d’exécution.

L’annulation moderne est idempotente : un job terminé ou inconnu retourne HTTP 200 avec `cancelled: false`. Un job en cours est interrompu et un job en attente est retiré. Le lot valide d’abord tous ses identifiants puis annule ceux encore actifs. [La vérification atomique](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/server.py#L957) évite qu’une annulation demandée pour A interrompe B entre deux snapshots. Cette sémantique exige un mécanisme de synchronisation associé au job en cours ; un simple booléen global C# ne suffit pas.

Attention : supprimer un job en attente n’implique pas automatiquement une entrée persistante `cancelled`. Si la cible ajoute un journal durable, ce sera une évolution explicite à spécifier, pas une simple reproduction du stockage actuel.

## 3. WebSocket et progression

### Session et texte JSON

[GET /ws](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/server.py#L269) prend `clientId` en query string ou crée un identifiant. Le serveur envoie immédiatement `status` avec la file et `sid`. Lors d’une reconnexion du client qui exécute, il peut renvoyer `executing` avec le nœud actuel.

Les messages JSON sont enveloppés dans `{ "type": événement, "data": charge }` : [send_json](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/server.py#L1382). Le premier message client peut être `feature_flags`; le serveur stocke les capacités pour cette connexion et répond avec les siennes. Le registre inclut notamment `supports_preview_metadata`, `supports_model_type_tags`, `max_upload_size`, `node_replacements`, `assets` : [feature_flags.py](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_api/feature_flags.py#L114).

L’UI consomme notamment `status`, `execution_start`, `executing`, `executed`, `execution_cached`, `execution_success`, `execution_error`, `execution_interrupted`, `progress`, `progress_state`, les prévisualisations et `logs`. La liste côté client ne prouve pas que chaque événement soit émis par chaque backend ; elle documente les cas à mapper : [api.ts](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/src/scripts/api.ts#L173).

[send_sync](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/server.py#L1392) publie depuis les threads d’exécution vers la boucle asynchrone. En C#, un canal d’événements séparé permet de conserver l’ordre et de ne pas bloquer l’inférence sur le rendu d’une prévisualisation. La politique de limitation des prévisualisations et des clients lents devra être mesurée et spécifiée.

### Trames binaires exactes

Tous les entiers ci-dessous sont des **uint32 big-endian**. [protocol.py](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/protocol.py#L2), [encode_bytes et send_image](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/server.py#L1302), [send_image_with_metadata](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/server.py#L1335), [send_progress_text](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/server.py#L1469).

| Type | Structure réseau |
| --- | --- |
| `1` : image | `[type=1:4 octets][format:4 octets][image encodée]`, format `1=JPEG`, `2=PNG` |
| `2` : image non encodée | Commande interne de dispatch convertie vers le type 1 ; ne pas traiter comme un format d’image réseau supplémentaire |
| `3` : progression texte locale | `[type=3][taille node_id][node_id UTF-8][texte UTF-8]` |
| `4` : image avec métadonnées | `[type=4][taille JSON][JSON UTF-8][image encodée]`, avec `image_type` ajouté aux métadonnées |

Le frontend sait aussi décoder une variante du type 3 incluant un identifiant de prompt quand la capacité `supports_progress_text_metadata` est annoncée : [api.ts](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/src/scripts/api.ts#L768). Le backend local audité ne l’annonce pas dans son registre et produit la trame plus courte. La cible doit annoncer uniquement les variantes réellement implémentées.

Dans l’application native, les mêmes événements peuvent être des types C# plutôt que des JSON, mais conserver un codec réseau indépendant est utile pour la compatibilité des scripts/API. La reconnexion, le changement de workflow actif et l’affectation des previews au bon job doivent être testés séparément : une image valide affichée dans le mauvais onglet constitue une erreur fonctionnelle.

## 4. Persistance et assets

### Trois stockages différents

| Donnée | Stockage actuel | Conséquence C# |
| --- | --- | --- |
| File et historique d’exécution | Mémoire, `PromptQueue.history`, limite configurée dans le code à 10 000 | Aucun engagement actuel de reprise après arrêt ; décider séparément d’une durabilité |
| Profils, réglages, workflows et autres userdata | Fichiers sous le répertoire utilisateur | Import/export et chemins compatibles ; préserver les champs JSON inconnus |
| Catalogue d’assets | SQLite + fichiers de contenu sur disque | Porter le schéma, l’indexation et les règles d’identité, pas seulement une liste de fichiers |

Références : [PromptQueue](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L1249), [users.json](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/app/user_manager.py#L56), [comfy.settings.json](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/app/app_settings.py#L11), [URL SQLite par défaut](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/app/database/db.py#L65).

La base par défaut se trouve sous le répertoire utilisateur effectif, `comfyui.db`. Le démarrage gère une ancienne localisation, un backup avant migration, la restauration en cas d’échec et un verrou de fichier empêchant deux processus de partager la même base. Les clés étrangères SQLite sont activées. Les migrations Alembic vont de `0001_assets` à `0006_add_loader_path` : [db.py](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/app/database/db.py#L181), [migration 0006](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/alembic_db/versions/0006_add_loader_path.py#L17).

Pour une nouvelle application, le choix le plus prudent est de créer sa base séparée et d’importer une copie versionnée des données existantes. Une lecture directe du schéma ComfyUI peut être utile ; faire migrer en place par deux outils différents nécessite une politique de versions et des tests de rollback explicites. Alembic lui-même ne doit pas devenir une dépendance d’exécution de la cible sans Python.

### Schéma métier des assets

[models.py](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/app/assets/database/models.py#L26) distingue :

- `assets` : identité de contenu, hash facultatif, taille, MIME et date de création ;
- `asset_references` : référence métier/chemin, propriétaire, nom, `file_path`, `loader_path`, dates, état manquant/à vérifier, enrichissement, preview, métadonnées utilisateur/système et `job_id` ;
- `asset_reference_meta` : projection de métadonnées typées pour le filtrage ;
- `asset_reference_tags` et `tags` : association et dictionnaire de tags.

Un hash de contenu n’est donc pas interchangeable avec l’identifiant d’une référence appartenant à un utilisateur. Plusieurs références peuvent pointer vers le même contenu. Les tags sont sensibles à la casse après [la migration 0005](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/alembic_db/versions/0005_allow_case_sensitive_tags.py#L12). Les listes d’assets utilisent un curseur lié au champ et au sens du tri, pas seulement un offset : [cursor.py](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/app/assets/services/cursor.py#L69).

Le système assure ingestion, hash Blake3, enregistrement des sorties, upload multipart, recherche, tags, preview, seeding et rapprochement avec le système de fichiers. `prune` marque les références hors racines comme manquantes et n’efface pas les fichiers : [route prune](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/app/assets/api/routes.py#L926). Confondre ce comportement avec un nettoyage physique ferait perdre des données.

Cette fonctionnalité est conditionnée par `enable_assets` et le code prévient que son contrat interne reste évolutif : [enregistrement](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/server.py#L257), [avertissement interne](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/app/assets/api/routes.py#L86). Un MVP peut commencer avec les fichiers classiques et annoncer `assets=false`. Il ne doit pas présenter une API assets partiellement correcte comme une parité complète.

## 5. Fichiers, sécurité et contexte utilisateur

[folder_paths.py](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/folder_paths.py#L257) gère les suffixes `[input]`, `[output]`, `[temp]`, les racines de modèles, les noms relatifs et les chemins de sauvegarde. `/view` comprend aussi des noms `blake3:...` résolus via le catalogue d’assets : [server.py](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/server.py#L521). Ces formes doivent être traduites par un service de résolution ; elles ne sont pas des chemins système bruts.

Les profils `--multi-user` s’appuient sur l’en-tête `comfy-user` et `users.json`, sans validation de mot de passe dans [get_request_user_id](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/app/user_manager.py#L59). C’est un mécanisme de sélection de profil local, pas une preuve d’authentification pour une exposition Internet. Les noms commençant par `__` sont réservés aux utilisateurs système et ne doivent pas être accessibles par les routes publiques.

Les contrôles à conserver dans l’adaptateur fichiers/API sont concrets :

- Confinement dans les racines autorisées ; résolution des liens symboliques et équivalents Windows avant lecture/écriture ; différences de volume, casse, noms absolus, décodage URL et traversées. [is_within_directory](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/folder_paths.py#L326) utilise `realpath`, alors que certains chemins d’upload/userdata effectuent seulement une vérification lexicale. Une traduction C# doit centraliser ce contrôle sans supposer que `GetFullPath` résout les jonctions.
- Protection contre les requêtes web cross-site sur le serveur local. [Le middleware actuel](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/server.py#L159) vérifie `Sec-Fetch-Site` et certaines correspondances Host/Origin ; activer CORS change cette politique.
- Limite de taille d’upload et streaming des fichiers ; éviter de charger plusieurs gigaoctets d’assets dans des tableaux temporaires managés.
- Téléchargement forcé des contenus actifs et `nosniff`. `/userdata` force `attachment`, avec une exception limitée à `user.css`; `/view` et les assets empêchent le rendu actif HTML/JS/XML/SVG, sauf SVG chargé comme image. Voir [UserManager](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/app/user_manager.py#L334) et [content assets](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/app/assets/api/routes.py#L426).
- Conservation de `Vary: Sec-Fetch-Dest` et `Cache-Control: no-store` lorsque le comportement SVG dépend du contexte. Une réponse mise en cache ne doit pas convertir un téléchargement sûr en document actif.
- Exclusion des secrets des historiques, événements et logs. Les identifiants `clientId`, `prompt_id` et `comfy-user` ont des rôles distincts.

Avec une UI native sans navigateur, une partie du risque de XSS disparaît de l’interface principale, mais subsiste si l’application conserve l’API de visualisation ou rend du HTML/Markdown/aperçus actifs. Les risques de chemins et de décodeurs médias natifs restent présents.

Les modules `comfy_config` traitent principalement les métadonnées TOML des paquets de nœuds : auteur/nom/version/licence, OS/accélérateurs, versions ComfyUI/frontend et ressources web/modèles. Voir [types.py](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_config/types.py#L47). Ils ne constituent pas à eux seuls la configuration générale de l’application. Un manifeste de plugin C# peut reprendre ces métadonnées descriptives, mais les dépendances Python et le champ `requires-python` ne deviennent pas automatiquement des dépendances NuGet.

## 6. Réécriture de l’interface : ce qu’il faut réellement remplacer

### État visuel et format de fichier

Le schéma frontend [workflowSchema.ts](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/src/platform/workflow/validation/schemas/workflowSchema.ts#L15) admet des identifiants de nœuds entiers ou chaînes, des types de slots historiques variés, des vecteurs sous plusieurs formes et `widgets_values` en tableau ou objet. Il conserve de nombreux champs inconnus avec `passthrough`.

Il distingue le format historique 0.4, dont les liens sont des tableaux, et le format 1, dont les liens sont des objets et qui structure les sous-graphes : [format 0.4](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/src/platform/workflow/validation/schemas/workflowSchema.ts#L328), [format 1](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/src/platform/workflow/validation/schemas/workflowSchema.ts#L380). Positions, dimensions, groupes, reroutes, zoom, décalage, couleurs, paramètres de widgets, propriétés de plugins et définitions imbriquées sont des données de document.

Conséquence : importer un workflow inconnu doit préserver ses données et afficher des nœuds manquants identifiables. Refuser son exécution avec un diagnostic précis est préférable à une sauvegarde qui supprime ses champs. L’import/export doit pouvoir faire plusieurs allers-retours sans dérive des identifiants, liens ou valeurs.

### Compilateur document → prompt

[graphToPrompt](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/src/utils/executionUtil.ts#L26) effectue notamment :

1. application des nœuds virtuels au graphe ;
2. sérialisation du document et normalisation de certains slots ;
3. création des nœuds exécutables et expansion des sous-graphes ;
4. suppression des nœuds muets, bypass et virtuels du prompt effectif ;
5. sérialisation éventuellement asynchrone des widgets ;
6. enveloppement des valeurs tableaux dans `__value__`, et balisage spécifique `CURVE`, pour les différencier des connexions ;
7. résolution des liens en `[nodeId chaîne, index de sortie]`, y compris les liens traversant les frontières de sous-graphes ;
8. création de `{class_type, inputs, _meta}` et élimination des entrées qui pointent vers des nœuds supprimés.

[ExecutableNodeDTO](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/src/lib/litegraph/src/subgraph/ExecutableNodeDTO.ts#L59) fabrique des identifiants hiérarchiques comme `1:2:3` selon le chemin d’instances. [resolveInput](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/src/lib/litegraph/src/subgraph/ExecutableNodeDTO.ts#L142) détecte les récursions, traverse les frontières et résout aussi les valeurs de widgets promus. Changer ces IDs sans table de correspondance casserait les erreurs et la progression renvoyées vers les nœuds visibles.

La cible doit avoir un compilateur C# testable indépendamment du framework de rendu. Il doit produire simultanément le prompt normalisé et une table de correspondance entre identités visuelles, instances de sous-graphes et identités d’exécution.

### Widgets et paramètres

Le registre actuel inclut les primitives INT/FLOAT/BOOLEAN/STRING/COMBO ainsi que Markdown, image upload, couleur, comparaison d’images, bounding boxes, chart, galerie, painter, compositor, textarea, curve, range, video edit et resolution preview : [widgets.ts](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/src/scripts/widgets.ts#L225). Ces widgets ne sont pas tous nécessaires au premier pipeline image.

Deux règles différentes doivent être portées : `widget.serialize` décide de la présence dans le fichier workflow, tandis que `widget.options.serialize` décide de la présence dans les entrées API. Le contrôle `control_after_generate` peut donc être conservé dans le document et absent du prompt : [documentation dédiée](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/docs/WIDGET_SERIALIZATION.md#L1).

La seed et certains combos ont des modes `fixed`, `increment`, `decrement`, `randomize`, avec des hooks avant/après queue : [widgets.ts](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/src/scripts/widgets.ts#L138). Ces comportements appartiennent au frontend actuel. Les omettre peut rendre deux soumissions successives différentes de ComfyUI alors que le moteur est correct. Les nombres de seed nécessitent aussi une représentation explicite, sans perte lors du passage par JSON et les contrôles numériques.

Les sous-graphes possèdent des entrées/sorties promues et des valeurs par instance. La version auditée documente des limites de propagation de `MatchType` et `Autogrow` à travers leurs frontières : [limitations](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/src/core/graph/subgraph-dynamic-input-limitations.md#L1). Il faut définir la compatibilité avec les comportements supportés, sans promettre que des cas déjà limités en amont fonctionneront automatiquement.

### Extensions et état applicatif

Le frontend actuel [charge dynamiquement des modules JavaScript](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/src/services/extensionService.ts#L44) et permet l’ajout de commandes, menus, raccourcis, settings, panneaux et widgets : [registerExtension](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/src/services/extensionService.ts#L74). Les extensions ont aussi des callbacks sur les objets du graphe.

Il faut créer un SDK C# versionné, avec des contrats distincts pour nœuds de calcul, widgets et contributions UI. Les anciennes extensions JavaScript ne sont pas directement compatibles ; une table de support doit identifier celles portées, remplaçables par le noyau, ou indisponibles. Introduire un navigateur ou une VM JavaScript pour les reproduire serait un choix de périmètre supplémentaire, absent de la cible confirmée.

L’éditeur gère également onglets, brouillons, chargements asynchrones, historique undo/redo, import/export, préférences, miniatures, galerie et progression. [workflowService](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/src/platform/workflow/core/services/workflowService.ts#L118) restaure par exemple le viewport dans le document. Les fonctions de rendu et la gestion du document doivent être séparées pour que changer d’onglet ne modifie pas les données d’un autre workflow.

### Matrice fonctionnelle de la cible

| Domaine UI | MVP utilisable | Extension de parité |
| --- | --- | --- |
| Canvas | Pan/zoom, sélection, déplacement, liens typés, suppression, undo/redo, copier/coller | Reroutes riches, groupes imbriqués, minimap, rendu de très grands graphes, disposition avancée |
| Nœuds | Catalogue des nœuds du pipeline retenu, recherche, champs de base, diagnostics de nœud manquant | Widgets dynamiques, tous les types avancés, badges, custom widgets |
| Fichiers workflow | Ouvrir/sauver JSON 0.4 et prompt API, préservation des champs inconnus | Format 1 complet, migration de sous-graphes, import de métadonnées multimédia étendu |
| Exécution | Queue, interruption, progression, erreurs rattachées aux nœuds, aperçu image, historique de session | Exécution partielle, autoqueue, préférences preview avancées, plusieurs contextes connectés |
| Paramètres | Seed et contrôles avant/après soumission, modèles, dossiers, préférences essentielles | Éditeur de raccourcis, thèmes, localisation complète, personnalisation des panneaux |
| Fichiers générés | Parcourir input/output, enregistrer et ouvrir une image, importer PNG avec workflow | Catalogue assets, tags, recherche, versions/enrichissement, audio/vidéo/3D |
| Sous-graphes | Détection et conservation des sous-graphes non encore exécutables ; aucune suppression silencieuse | Création/édition, expansion, nesting, promotion, valeurs indépendantes par instance |
| Édition médias | Chargement image et masque simple si requis par le pipeline | Painter, compositor, courbes, comparaison, éditeur vidéo, 3D, capture audio/webcam |
| Extensions | SDK minimal et liste fermée de nœuds C# validés | Plugins C# versionnés, contributions UI et compatibilité des packs portés |
| API | Endpoints d’exécution/fichiers indispensables et protocole d’événements | Parité locale étendue, routes expérimentales/internes si nécessaires |

Cette matrice est une proposition de lotissement. Une parité totale dès le MVP rendrait le projet beaucoup plus long que le portage d’un premier pipeline de génération.

## 7. Vérification à préparer avant développement

Les tests existants fournissent des cas de référence, mais leurs assertions et fixtures doivent être adaptées aux contrats C#. Il ne s’agit pas de traduire tous les tests Python/Playwright à l’identique.

| Axe | Référence existante | Validation de la cible |
| --- | --- | --- |
| Schéma | [workflowSchema.test.ts](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/src/platform/workflow/validation/schemas/workflowSchema.test.ts#L9) | Import des variantes, champs inconnus, IDs mixtes, valeurs et liens ; deux allers-retours sans perte |
| Sous-graphes | [ExecutableNodeDTO.test.ts](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/src/lib/litegraph/src/subgraph/ExecutableNodeDTO.test.ts#L1), [subgraphSerialization.spec.ts](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/browser_tests/tests/subgraph/subgraphSerialization.spec.ts#L333) | Prompt développé identique pour références sélectionnées, ID remapping, valeurs promues, copies indépendantes, cycles |
| État des onglets | [workflowPersistence.spec.ts](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/browser_tests/tests/workflowPersistence.spec.ts#L153) | Sauvegarde du bon onglet, liens et valeurs stables, brouillons récupérables, absence d’URLs temporaires dans l’export |
| Exécution UI | [execution.spec.ts](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/browser_tests/tests/execution.spec.ts#L100) | Erreurs, previews et progression affectées au bon graphe, exécution ciblée si annoncée |
| Protocole | [websocket_feature_flags_test.py](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests-unit/websocket_feature_flags_test.py#L1) | Trames binaires de référence, endianness, UTF-8, négociation, reconnexion, tailles invalides |
| Jobs | [test_jobs.py](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests/execution/test_jobs.py#L63), [jobs_cancel_test.py](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests-unit/jobs_cancel_test/jobs_cancel_test.py#L331) | Statuts, UUID, pagination, idempotence et course pending→running/fin de job |
| Persistance | [test_migrations.py](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests-unit/app_test/test_migrations.py#L49), [database_path_test.py](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests-unit/app_test/database_path_test.py#L7) | Import d’une copie de base, backup, rollback, verrou, tags sensibles à la casse |
| Fichiers | [traversées](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests-unit/security_test/test_ghsa_779p_03_annotated_traversal.py#L60), [userdata XSS](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests-unit/security_test/test_ghsa_779p_04_userdata_xss.py#L47), [SVG et cache](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests-unit/security_test/test_ghsa_779p_06_inline_svg_image_dest.py#L137) | Traversées Windows et liens, comptes système, MIME actifs, cache par destination, uploads volumineux |

Un corpus contractuel doit être versionné avec les deux commits : workflows simples, reroutes, bypass/mute, seeds, sous-graphes, sorties image/texte, messages d’erreur, previews et catalogue de nœuds. Les identifiants aléatoires et dates doivent être normalisés lors des comparaisons, sans masquer les différences sémantiques.

La première recette produit doit démontrer une installation sur une machine sans Python : ouverture de l’UI native, chargement du workflow de référence, découverte des modèles, soumission, annulation, affichage du résultat et réouverture du document sans perte. Les benchmarks de qualité/vitesse du moteur sont traités dans l’audit inférence ; ceux du canvas doivent couvrir la fluidité, les allocations et le nombre de nœuds représentatif des workflows retenus.

## 8. Licence et limites de couverture

Le backend contient la GPL version 3 : [LICENSE](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/LICENSE#L1). Le frontend audité déclare explicitement `GPL-3.0-only` dans [package.json](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/package.json#L7) et fournit son propre texte de licence. Une conversion directe de code doit être planifiée comme une adaptation de ces sources ; changer de langage ne suffit pas à supprimer leurs conditions. La distribution, les composants natifs et leurs licences doivent être vérifiés dans le lot juridique/dépendances de la migration.

Sont hors audit détaillé de ce document : tous les plugins tiers, chaque widget média, les produits Comfy Cloud/Comfy Desktop, le site marketing, le moteur de calcul, les codecs natifs et les licences de tous les modèles. L’absence de port de ces fonctions doit figurer dans la matrice de compatibilité livrée avec le premier exécutable. Les sources clonées permettent de poursuivre l’audit UI sans inclure artificiellement son volume dans celui du backend.
