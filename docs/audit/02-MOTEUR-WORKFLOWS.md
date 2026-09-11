> Audit statique antérieur à l’implémentation. Les choix définitifs et le périmètre V1 sont fixés par [le plan approuvé](../MIGRATION.md). Les observations ci-dessous ne sont pas des résultats de tests du port C#.

# Moteur de workflows, contrats de nœuds et migration en C#

Audit statique du backend ComfyUI au commit `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`, vérifié dans `le clone de référence`. Les références de code correspondent à cette révision. Aucun lancement de ComfyUI, aucune installation et aucun test d'exécution n'ont été effectués pour cette analyse.

La cible demandée est une application dont le backend **et l'interface utilisateur sont réécrits en C#**, avec bibliothèques natives si nécessaire et sans runtime, processus, passerelle ni plugin Python dans le produit distribué. Le frontend actuel peut servir de référence de comportement ; sa conservation comme interface livrée ne constitue pas la cible. Ce document traite le moteur, ses contrats de données et les exigences qu'il impose à la nouvelle interface. L'inférence détaillée et l'inventaire des endpoints relèvent des autres analyses.

Le moteur peut être réécrit en C#, mais la compatibilité exige de reproduire ses contrats dynamiques : listes, évaluation différée, expansion de graphes, traitements asynchrones et invalidation des caches. Une traduction mécanique des classes ne suffit pas. La compatibilité des workflows sera limitée au catalogue des nœuds et extensions effectivement portés.

## 1. Flux d'exécution actuel

### Initialisation

`main.py` applique les répertoires modèles, entrées, sorties et utilisateur, ainsi que la configuration des chemins supplémentaires. Il peut ensuite exécuter les scripts de démarrage des extensions Python. Dans la cible, la configuration doit être conservée ou importée ; les scripts doivent être remplacés par des initialisations de plugins .NET explicitement portés. Sources : [configuration des chemins](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/main.py#L140), [scripts de démarrage](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/main.py#L183).

L'initialisation des nœuds enregistre successivement les versions des API publiques, les nœuds supplémentaires intégrés, les nœuds d'API et les extensions externes. Le registre associe les noms de types utilisés par les workflows à leurs classes. Sources : [ordre d'initialisation](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/nodes.py#L2571), [registre intégré](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/nodes.py#L2068).

### File de travaux et worker

Le démarrage crée **un seul worker de prompts**. Celui-ci conserve un `PromptExecutor` entre travaux, lit la file, exécute le prompt, renseigne l'historique et assure les opérations de nettoyage mémoire. La file utilise un tas de priorité, un verrou réentrant, une condition, un registre des travaux actifs et un historique. Il faut donc conserver les priorités ; une simple file FIFO modifierait le comportement. Sources : [création du worker](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/main.py#L560), [boucle du worker](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/main.py#L351), [PromptQueue](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L1251).

### Validation puis exécution

1. Le validateur recherche les nœuds marqués `OUTPUT_NODE`, éventuellement filtrés par la sélection d'exécution partielle.
2. Il valide récursivement les dépendances de ces sorties, les types, les littéraux, les validateurs personnalisés et les cycles.
3. Une sortie invalide peut être ignorée tandis que les autres sorties valides restent exécutables. La validation globale réussit dès qu'au moins une sortie valide subsiste.
4. L'exécuteur remet à zéro l'interruption, initialise le contexte du prompt, met à jour les clés et contenus des caches, puis construit l'ordonnanceur dynamique.
5. Chaque étape renvoie `SUCCESS`, `FAILURE` ou `PENDING`. `PENDING` peut signifier dépendance lazy découverte, expansion en attente ou tâche asynchrone inachevée.
6. Les résultats UI et leurs métadonnées alimentent l'historique. Les tenseurs et objets d'exécution sont conservés selon leurs références et les politiques de cache, pas sérialisés dans cet historique.

Sources : [validation du prompt](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L1128), [validation récursive](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L846), [sorties valides](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L1203), [résultat de validation partielle](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L1247), [cycle du prompt](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L730), [étape d'exécution](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L438), [historique](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L831).

### Frontière recommandée en C#

Les responsabilités suivantes doivent rester distinctes :

| Composant proposé | Responsabilité |
|---|---|
| `PromptQueue` | Priorité, sélection du travail actif, annulation ciblée |
| `PromptValidator` | Résolution des types, validation et normalisation des entrées |
| `DynamicGraphScheduler` | Dépendances, reprises, expansions, tâches externes |
| `NodeInvoker` | Contrat d'appel, listes, contexte, normalisation des sorties |
| `ExecutionCache` | Clés, invalidation, références et politique d'éviction |
| `ExecutionResult` et événements | Résultats, progression et erreurs structurées |
| Adaptateur de présentation | Transformation des événements pour l'interface C# |

Le cœur doit être indépendant de l'interface, du transport HTTP/WebSocket et du stockage d'historique. Cette séparation est également demandée par les [frontières d'architecture du dépôt](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/AGENTS.md#L30). Le code actuel contient encore des appels serveur dans `execution.py` ; il faut préserver leur signification observable au moyen d'événements, sans imposer cette dépendance à la nouvelle architecture.

## 2. Contrat du graphe et des valeurs

Le prompt exécutable est un dictionnaire `node_id -> { class_type, inputs }`. Il est distinct du document de disposition graphique de l'éditeur. Les IDs sont des chaînes et les connexions sont représentées par `[node_id, output_index]`. Le prédicat historique accepte un index numérique ; cela ne prouve pas qu'un index fractionnaire soit utilisable par les opérations qui l'indexent ensuite. Il faut tester séparément les graphes valides et les erreurs sur indices mal formés. Sources : [détection des liens](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_execution/graph_utils.py#L1), [sérialisation d'un nœud](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_execution/graph_utils.py#L106).

Les contrats à figer sont les noms exacts de `class_type`, les noms des entrées, l'ordre et le nombre des sorties, les catégories de paramètres, les valeurs par défaut et les métadonnées. Les identifiants doivent rester stables même si les classes C# utilisent d'autres noms.

Trois concepts doivent être distingués :

- La **liste de valeurs d'exécution**, sur laquelle le moteur répète les appels de nœuds.
- Le **batch porté par un tenseur**, traité par les opérations d'inférence ou d'image.
- Le **tableau JSON littéral**, qui constitue une valeur de widget ou une structure métier.

Les tableaux JSON sont interprétés comme connexions lors de la validation. Le wrapper `{"__value__": [...]}` autorise un tableau littéral et le validateur le déroule dans le prompt. Un importeur qui désérialise immédiatement tous les tableaux vers des listes ordinaires perdrait cette distinction. Source : [déroulement des valeurs littérales](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L983).

Une entrée `rawLink` reçoit le lien lui-même au lieu du résultat du nœud source. Les entrées non déclarées sont normalement écartées ; V3 permet de les accepter explicitement avec `accept_all_inputs`. Sources : [résolution des entrées](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L159), [liens bruts](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L174), [entrées non déclarées](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L189), [option V3](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_api/latest/_io.py#L1751).

Les entrées cachées comprennent le prompt, le graphe dynamique, l'ID du nœud, les métadonnées PNG et, pour certains nœuds, des données d'authentification. Elles doivent être injectées dans un contexte d'appel et rester séparées des valeurs persistées par défaut dans les workflows. Source : [préparation du contexte des entrées](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L193).

Les sorties circulant entre nœuds comprennent des objets métier et des tenseurs natifs, pas seulement des valeurs sérialisables. La cible doit utiliser une représentation explicite `Literal | Link` à l'import, puis un conteneur de valeurs d'exécution extensible. Un moteur reposant uniquement sur `JsonElement` ne convient pas. Les structures latentes et de conditionnement doivent conserver leurs métadonnées extensibles ; un typage C# qui supprimerait les champs inconnus casserait les familles de nœuds qui les utilisent.

## 3. Unification des contrats V1 et V3

### Contrat V1

Le contrat historique repose sur `INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION`, `CATEGORY`, `OUTPUT_NODE`, `INPUT_IS_LIST`, `OUTPUT_IS_LIST`, `IS_CHANGED` et `VALIDATE_INPUTS`. Le moteur découvre les capacités par inspection de classes et appelle la méthode nommée dans `FUNCTION`. Exemple : [CLIPTextEncode](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/nodes.py#L56).

### Contrat V3

V3 fournit un schéma et des méthodes nommées :

| Élément | Rôle | Source |
|---|---|---|
| `io.Schema` | Identité, ports, catégories, drapeaux, comportement | [Schema](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_api/latest/_io.py#L1699) |
| `define_schema` | Description du nœud | [définition](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_api/latest/_io.py#L1981) |
| `execute` | Calcul synchrone ou asynchrone | [exécution](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_api/latest/_io.py#L1987) |
| `validate_inputs` | Validation personnalisée | [validation](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_api/latest/_io.py#L1992) |
| `fingerprint_inputs` | Empreinte d'invalidation | [empreinte](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_api/latest/_io.py#L2000) |
| `check_lazy_status` | Dépendances à évaluer | [lazy](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_api/latest/_io.py#L2007) |
| `NodeOutput` | Résultats, UI, expansion et blocage | [sortie](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_api/latest/_io.py#L2337) |

V3 n'est pas un moteur d'exécution séparé. Il normalise les résultats et expose les propriétés historiques au moteur existant. La cible devrait donc utiliser **un descripteur de nœud commun**, accompagné de règles d'import et de compatibilité pour les représentations V1 et V3. Sources : [normalisation des résultats](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_api/latest/_io.py#L2045), [informations compatibles V1](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_api/latest/_io.py#L2097), [entrées V1](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_api/latest/_io.py#L2224), [initialisation des propriétés](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_api/latest/_io.py#L2239).

Le contrat cible peut être exprimé par une interface `IComfyNode` exposant un schéma et une méthode `ExecuteAsync(NodeInputs, NodeExecutionContext, CancellationToken)` qui renvoie un `NodeResult`. Les capacités de validation, empreinte, lazy et expansion doivent être explicites et facultatives. Cette proposition est une architecture à implémenter, pas du code produit déjà disponible.

### Entrées dynamiques et isolation

`Autogrow`, `DynamicCombo` et `DynamicSlot` permettent de faire varier les entrées selon le prompt. Les entrées sont finalisées à partir des valeurs présentes, puis leurs chemins plats sont restructurés en dictionnaires imbriqués avant l'appel. Un schéma statique généré une fois au démarrage ne couvre pas ce contrat. Sources : [Autogrow](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_api/latest/_io.py#L1083), [DynamicCombo](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_api/latest/_io.py#L1220), [DynamicSlot](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_api/latest/_io.py#L1273), [finalisation](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_api/latest/_io.py#L1878), [reconstruction des entrées](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_api/latest/_io.py#L1935).

Les invocations V3 préparent une copie de la classe et y injectent les données cachées. En C#, il faut définir un contexte immuable par invocation et une durée de vie explicite des instances ; la reproduction des métaclasses Python n'apporte pas de compatibilité utile. Source : [préparation de l'appel V3](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_api/latest/_io.py#L2085).

## 4. Sémantique des listes

L'invocation d'un nœud sur les listes suit des règles précises :

1. Par défaut, le nombre d'appels correspond à la plus longue liste d'entrée.
2. Une liste plus courte répète son dernier élément.
3. Sans entrée, le moteur réalise un appel avec des arguments vides.
4. `INPUT_IS_LIST=True` transmet toutes les listes en un seul appel.
5. Les résultats sont regroupés par port ; `OUTPUT_IS_LIST=True` concatène les éléments de ce port, sinon chaque résultat d'appel devient un élément de liste.

Sources : [application sur listes](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L243), [fusion des résultats](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L322).

Ce comportement ne correspond ni à `Zip`, ni à un produit cartésien, ni au broadcasting tensoriel. Exemple : des entrées `[a, b, c]` et `[x]` provoquent `(a, x)`, `(b, x)` puis `(c, x)`. Les listes vides, entrées absentes, tenseurs en liste, valeurs imbriquées et bloqueurs doivent disposer de tests spécifiques. La validation des cas extrêmes doit distinguer les comportements intentionnels des erreurs incidentes du moteur historique.

## 5. Ordonnanceur, lazy et expansion

### Évaluation différée

L'ordonnanceur maintient les dépendances bloquantes. Les entrées marquées lazy sont exclues de la dépendance initiale. Le nœud reçoit les valeurs disponibles et les marqueurs d'absence, appelle `check_lazy_status`, puis les ports manquants demandés deviennent des dépendances fortes. L'étape renvoie `PENDING` et peut être reprise. Sources : [structure des dépendances](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_execution/graph.py#L106), [ajout des nœuds](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_execution/graph.py#L138), [promotion en lien fort](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_execution/graph.py#L120), [appel lazy](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L507).

Le port doit éviter que ces reprises exécutent plusieurs fois les effets de bord du calcul final. `check_lazy_status` et `execute` doivent rester deux opérations distinctes. Une branche non demandée ne doit pas être calculée simplement parce qu'elle apparaît dans le graphe.

### Expansion en sous-graphes

Une expansion ajoute des nœuds éphémères, enregistre leur parent et leur identité d'affichage, prépare leurs sous-caches, puis attend les sorties nécessaires avant de résoudre les liens retournés et de reprendre le parent. Source : [traitement des expansions](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L579).

`DynamicPrompt` conserve le graphe original, les nœuds éphémères, la relation parent, l'ID réel et l'ID affiché. Cette distinction sert à rattacher les erreurs et résultats au bon nœud visible. Source : [DynamicPrompt](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_execution/graph.py#L21).

`GraphBuilder` génère les préfixes d'ID. Son préfixe par défaut utilise actuellement de l'état de classe ; un commentaire du moteur reconnaît la difficulté avec les fonctions asynchrones. La cible doit calculer les identifiants à partir d'un contexte d'exécution, de l'ID du nœud et de l'index d'appel, avec une politique explicite de collision. Sources : [GraphBuilder](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_execution/graph_utils.py#L13), [initialisation du préfixe](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L540).

### Cycles et ordre d'exécution

Les cycles sont détectés pendant la validation statique puis pendant l'exécution dynamique. Lorsque le graphe n'a plus de nœud disponible, l'ordonnanceur distingue l'attente d'une tâche externe d'un cycle. Il rattache autant que possible une erreur dynamique au nœud affiché. Source : [sélection d'une étape](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_execution/graph.py#L242).

L'heuristique actuelle favorise les sorties et les nœuds asynchrones, puis les nœuds proches d'une sortie, afin de présenter les premiers résultats rapidement. La parité fonctionnelle doit porter sur les dépendances et les sorties ; un ordre total entre branches indépendantes ne doit pas être inventé comme garantie. Source : [choix du prochain nœud](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_execution/graph.py#L275).

## 6. Asynchronisme, contexte et annulation

Les fonctions asynchrones sont transformées en tâches. Les appels correspondant aux listes peuvent progresser simultanément. Lorsqu'une tâche n'est pas achevée, seul son nœud reçoit un blocage externe et les autres branches prêtes continuent. Sources : [invocation asynchrone](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L294), [attente des tâches](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L554), [blocage externe](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_execution/graph.py#L168).

Le worker de prompts est unique et les nœuds synchrones ne sont pas automatiquement parallélisés. Une parallélisation générale des nœuds GPU dans le port changerait les pics mémoire, les accès partagés et potentiellement les résultats. La première politique recommandée est un seul prompt actif, une voie de calcul native sérialisée et de la concurrence explicite pour les opérations asynchrones adaptées.

Le contexte courant contient l'ID de prompt, l'ID de nœud et l'index de liste. Python utilise `contextvars`. Un contexte transmis explicitement, ou un usage maîtrisé d'`AsyncLocal`, convient en C# à condition de restaurer le contexte après chaque invocation. Source : [contexte d'exécution](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_execution/utils.py#L4).

L'annulation actuelle est coopérative : elle est vérifiée avant l'invocation, repose sur un signal remis à zéro au début du prompt et produit un événement distinct des erreurs ordinaires. L'annulation ciblée d'un travail actif est protégée par le verrou de file afin d'éviter une course qui annulerait le travail suivant. Sources : [vérification avant nœud](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/nodes.py#L48), [point de contrôle](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L258), [début du prompt](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L730), [annulation atomique](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L1323), [gestion de l'interruption](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L619).

Le test `test_async_cancellation` contient seulement `pass` et un TODO ; il ne prouve donc aucune couverture d'annulation asynchrone. Source : [test incomplet](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests/execution/test_async_nodes.py#L392).

La cible doit définir la propagation du `CancellationToken`, l'annulation et l'attente des tâches enfants, la libération des handles et les points d'interruption entre opérations natives. Un kernel natif déjà engagé n'est pas nécessairement préemptible. La nouvelle interface C# doit distinguer la demande d'annulation de sa prise en compte effective et empêcher qu'une progression tardive soit attribuée au travail suivant.

## 7. Caches et durée de vie des ressources

### Deux caches de nature différente

Le cache des objets nœuds utilise `(node_id, class_type)` ; il conserve les instances. Le cache des résultats utilise une signature structurelle du nœud et de son ascendance. Les instances et les résultats doivent rester deux responsabilités distinctes. Sources : [ensemble des caches](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L116), [clés d'instances](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_execution/caching.py#L67), [clés de résultats](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_execution/caching.py#L82).

Les signatures incluent type, constantes, liens normalisés selon l'ordre déterministe des ancêtres et résultat de `IS_CHANGED` ou `fingerprint_inputs`. Cette structure autorise la réutilisation entre nœuds ayant les mêmes entrées. `NOT_IDEMPOTENT` ajoute l'ID du nœud à la signature : cela interdit principalement le partage avec un autre nœud identique ; ce drapeau ne signifie pas « recalculer obligatoirement à chaque prompt ». Source : [signature immédiate et ancêtres](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_execution/caching.py#L109).

Le calcul de l'empreinte reçoit les constantes et non les résultats amont. Une exception d'empreinte produit `NaN` pour invalider le cache. Un hash JSON ordinaire ne reproduit ni cette politique, ni les valeurs non hashables, ni la normalisation des liens. Source : [IsChangedCache](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L60).

### Politiques disponibles

| Politique | Comportement à conserver | Source |
|---|---|---|
| Classique hiérarchique | Nettoyage selon le prompt vivant et sous-caches d'expansion | [HierarchicalCache](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_execution/caching.py#L361) |
| Désactivée | Absence de persistance des résultats intermédiaires dans ce cache | [NullCache](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_execution/caching.py#L410) |
| LRU | Suivi des générations, limite et réutilisation des clés | [LRUCache](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_execution/caching.py#L439) |
| Pression RAM | Éviction tenant compte de la mémoire, de l'âge et de l'usage actif | [RAMPressureCache](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_execution/caching.py#L522) |

Le défaut du worker dans cette révision est le cache à **pression RAM**. Il ne faut pas déduire le défaut du seul constructeur du cache. Source : [sélection du cache](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/main.py#L363).

L'éviction sous pression RAM inspecte les storages CPU pour éviter de compter plusieurs fois les vues d'un même stockage, favorise l'abandon d'anciens objets de modèles et tient compte des workflows actifs. Une mesure limitée à la taille des objets managés serait insuffisante. Source : [libération sous pression](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_execution/caching.py#L550).

L'ordonnanceur conserve également les références de résultats nécessaires aux consommateurs actifs et peut réinsérer une valeur dans le cache lors de sa consultation. **Éviction d'une entrée et destruction d'un tenseur doivent rester différentes.** La cible nécessite une propriété partagée ou un comptage de références des handles natifs, avec destruction déterministe après leur dernier utilisateur. Sources : [liens de cache](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_execution/graph.py#L209), [consultation et réinsertion](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_execution/graph.py#L220), [propagation des valeurs](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_execution/graph.py#L230).

### Fournisseurs de cache

L'API publique offre lookup/store asynchrones, sélection des valeurs et callbacks de début/fin de prompt. La clé externe est canonisée, typée puis hachée en SHA-256 ; les valeurs non comparables à elles-mêmes, dont `NaN`, empêchent l'usage du cache externe. Une simple chaîne JSON de la configuration du nœud serait incorrecte. Sources : [contrat fournisseur](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_api/latest/_caching.py#L19), [canonisation](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_execution/cache_provider.py#L56), [sérialisation et exclusion](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_execution/cache_provider.py#L88).

Cette API peut devenir une extension .NET. Les fournisseurs Python ne seront pas chargeables dans la cible sans Python. Les stores asynchrones doivent avoir une politique explicite de fin de travail et d'arrêt de l'application afin de garantir la propriété des ressources qu'ils consultent.

## 8. Validation, erreurs et écarts de langage

Les catégories d'erreur et leurs données structurées doivent être conservées ou traduites par un adaptateur de compatibilité : `type`, `message`, `details`, `extra_info`, identité de nœud et sorties dépendantes. La validation personnalisée peut prendre en charge des entrées et remplacer certaines vérifications standard. Les unions de types acceptent par défaut une intersection ; `*`, `MatchType` et certains types historiques reçoivent un traitement particulier. Sources : [validation récursive](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L846), [compatibilité des types](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_execution/validation.py#L4).

La conversion des littéraux emploie les conversions Python `int`, `float`, `str` et `bool`, puis modifie les entrées du prompt. Les conversions C# directes ne sont pas identiques. Par exemple, une chaîne non vide `"false"` devient vraie avec `bool` Python. Il faut une couche de coercition de compatibilité explicite et des fixtures portant sur les booléens, chaînes, nombres, valeurs nulles et tableaux enveloppés. Source : [coercitions](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L992).

`ExecutionBlocker` se propage aux consommateurs sans les invoquer. Un message peut produire un événement d'erreur, tandis qu'une absence de message bloque silencieusement. Ce mécanisme ne doit pas être assimilé à une exception fatale de tout le prompt. Sources : [ExecutionBlocker](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_execution/graph_utils.py#L140), [traitement des bloqueurs](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L516).

Les erreurs de mémoire GPU déclenchent un traitement particulier et un déchargement des modèles. Le port doit exposer une catégorie métier `OutOfMemory` indépendante du type d'exception de la bibliothèque native, puis appliquer une politique de nettoyage contrôlée. Source : [gestion OOM](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L641).

Un cas est explicitement signalé comme incertain dans le code : les sorties UI combinées à une expansion peuvent être mises en cache avant la fin des sous-graphes. Il s'agit d'un commentaire de risque dans la source, pas d'un défaut reproduit pendant cet audit. Définir un cas de référence et documenter toute correction volontaire. Source : [commentaire sur UI et expansion](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L408).

## 9. Extensions et interface C#

Le chargeur actuel importe du code Python, accepte `NODE_CLASS_MAPPINGS` ou `comfy_entrypoint`, exécute `on_load`, extrait les schémas et enregistre éventuellement des ressources web. Chaque étape suppose du code Python exécutable ou des extensions frontend historiques. Sources : [chargeur d'extensions](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/nodes.py#L2246), [contrat ComfyExtension](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_api/latest/__init__.py#L116).

Les extensions existantes doivent être portées en C#, remplacées par des implémentations C#/natives compatibles ou signalées comme non prises en charge. La présence de leur ID dans un workflow ne suffit pas à rendre leur implémentation disponible.

Le SDK cible devrait fournir un manifeste versionné, un registre `class_type`, les schémas, une initialisation, des capacités d'exécution et des contributions d'interface C#. Les widgets et comportements interactifs ajoutés par JavaScript dans les extensions actuelles doivent eux aussi être réécrits pour la nouvelle UI ; conserver `WEB_DIRECTORY` ne résout pas cette contrainte.

L'interface C# devra notamment :

- Afficher les ports et widgets définis par les schémas et actualiser les entrées dynamiques.
- Préserver les IDs métier et les données de disposition lors d'un import/export de workflow.
- Présenter les états en attente, en cours, en cache, terminés, bloqués et annulés.
- Rattacher les nœuds éphémères à leur parent affiché et montrer les erreurs au bon endroit.
- Restaurer les sorties UI mises en cache, y compris les sorties intermédiaires qui ne déclenchent pas seules l'exécution.
- Donner avant exécution un inventaire des nœuds manquants et des extensions à convertir.

Le drapeau V3 `has_intermediate_output` conserve des sorties UI sans faire du nœud une racine d'exécution. Cela impose une distinction dans le modèle de présentation. Source : [sorties intermédiaires](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_api/latest/_io.py#L1755).

La compatibilité annoncée devra être : **import et exécution des workflows appartenant au catalogue porté et vérifié**. Elle ne devra pas être présentée comme une compatibilité d'exécution des plugins Python ou des contributions JavaScript historiques.

## 10. Tests de parité

### Scénarios existants à transformer en fixtures autonomes

| Domaine | Sources principales |
|---|---|
| Lazy et cache complet/partiel | [lazy](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests/execution/test_execution.py#L239), [cache complet](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests/execution/test_execution.py#L256), [cache partiel](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests/execution/test_execution.py#L273) |
| Validation personnalisée et kwargs | [littéraux](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests/execution/test_execution.py#L313), [types liés](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests/execution/test_execution.py#L328), [kwargs](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests/execution/test_execution.py#L390) |
| Cycles statiques et dynamiques | [cycle statique](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests/execution/test_execution.py#L401), [cycle dynamique](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests/execution/test_execution.py#L415) |
| Empreintes et entrées non déclarées | [IS_CHANGED](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests/execution/test_execution.py#L449), [entrées non déclarées](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests/execution/test_execution.py#L472), [empreinte et sorties](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests/execution/test_execution.py#L555) |
| Boucles, expansions et lazy mixte | [boucle](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests/execution/test_execution.py#L486), [expansions mixtes](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests/execution/test_execution.py#L505), [lazy mixte](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests/execution/test_execution.py#L524) |
| Réutilisation des sorties | [réutilisation](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests/execution/test_execution.py#L541) |
| Async et expansion | [branches asynchrones](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests/execution/test_execution.py#L577), [expansion](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests/execution/test_execution.py#L608) |
| Bloqueurs et listes | [sortie liste bloquée](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests/execution/test_execution.py#L651) |
| Exécution partielle | [sorties incluses](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests/execution/test_execution.py#L675), [dépendances](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests/execution/test_execution.py#L719), [liste vide](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests/execution/test_execution.py#L808) |
| Cache et absence de client | [résultats sans client](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests/execution/test_execution.py#L821) |
| Erreurs async et récupération | [erreur](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests/execution/test_async_nodes.py#L198), [récupération](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests/execution/test_async_nodes.py#L239), [erreur sync pendant async](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests/execution/test_async_nodes.py#L262) |
| Async lazy, cache et graphe dynamique | [lazy](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests/execution/test_async_nodes.py#L155), [cache](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests/execution/test_async_nodes.py#L316), [graphe dynamique](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests/execution/test_async_nodes.py#L339) |
| Isolation de la progression | [clients distincts](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests/execution/test_progress_isolation.py#L117), [client absent](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests/execution/test_progress_isolation.py#L203) |
| Canonisation, hash et NaN | [canonisation](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests-unit/execution_test/test_cache_provider.py#L29), [hash](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests-unit/execution_test/test_cache_provider.py#L130), [NaN](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests-unit/execution_test/test_cache_provider.py#L199) |
| Unions et types spéciaux | [validation de types](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests-unit/execution_test/validate_node_input_test.py#L5) |

La suite d'intégration du moteur lance actuellement un serveur Python en mode CPU et paramètre les caches classique, LRU et désactivé. Ces scénarios fournissent une base de comportement ; ils ne prouvent pas la parité du défaut RAM et ne peuvent pas être livrés tels quels dans une suite autonome sans Python. Source : [configuration de la suite](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests/execution/test_execution.py#L185).

### Vérifications supplémentaires prioritaires

1. Annulation async réelle, erreur pendant tâche enfant et absence d'événements du prompt précédent après démarrage du suivant.
2. Collision et isolation des IDs de sous-graphes lors d'appels asynchrones et de plusieurs expansions d'un même nœud.
3. Répétition du dernier élément de liste, distinction entre listes et batches, listes imbriquées et absence d'entrée.
4. Coercions de littéraux, wrappers `__value__`, unions, `rawLink` et validation personnalisée.
5. Durée de vie de tenseurs partagés après éviction, suppression d'un consommateur et annulation.
6. Invalidation après modification du contenu d'un fichier d'entrée, changement d'empreinte et valeurs `NaN`.
7. Cache sous pression RAM : seuils, vues de même stockage, références actives et reprise après éviction.
8. Sous-graphe avec sorties UI, restauration des sorties intermédiaires et absence de client connecté.
9. V3 Autogrow, DynamicCombo et DynamicSlot imbriqués, sérialisation du schéma et reconstruction des valeurs.
10. Catalogue exact de nœuds portés, erreurs lisibles sur les nœuds manquants et contributions de widgets dans l'UI C#.
11. Installation et lancement du produit distribué sur une machine sans Python ; inspection des dépendances, des processus enfants et des mécanismes de plugins pour confirmer l'absence de dépendance Python.

La nouvelle suite doit être écrite en .NET avec graphes JSON, nœuds synthétiques C# et résultats attendus. Les tests du moteur peuvent être déterministes et indépendants des modèles lourds. Les tests numériques et GPU doivent rester une couche séparée, avec tolérances et politiques mémoire adaptées aux opérations natives.

## 11. Ordre de migration proposé pour ce périmètre

| Étape | Livrable vérifiable |
|---|---|
| Contrats et fixtures | Formats de prompt, descripteurs de nœuds, valeurs, erreurs et catalogue initial |
| Validation | Résolution des nœuds, types, coercions, sorties partielles et cycles statiques |
| Ordonnanceur de base | Dépendances, invocation, regroupement des sorties et sémantique des listes |
| Comportements dynamiques | Lazy, expansions, contextes, tâches async, annulation et cycles dynamiques |
| Cache et ressources | Cache classique, empreintes, références de handles et évictions sûres |
| Catalogue natif minimal | Un parcours de génération complet avec les nœuds nécessaires effectivement portés |
| Interface C# | Édition du graphe, schémas dynamiques, progression, erreurs, prévisualisation et import/export |
| Pression RAM et extensions | Politique mémoire vérifiée, SDK .NET et premières extensions portées |

Le moteur, les nœuds synthétiques, l'import de graphes et une partie de l'interface C# peuvent progresser indépendamment de l'implémentation complète des modèles. L'acceptation de chaque étape doit reposer sur des fixtures et des scénarios exécutés dans le futur port. Cet audit décrit les comportements observés statiquement et les travaux nécessaires ; il n'établit pas qu'une conversion existe déjà ni qu'elle a passé des tests.
