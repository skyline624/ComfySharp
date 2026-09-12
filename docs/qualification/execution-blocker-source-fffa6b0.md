# ExecutionBlocker — référence source V1

La collecte stdlib du laboratoire publié au commit `fffa6b06bbde2a76e2fe40e9097f03c7b169cee8` a exécuté 24 cas prospectifs. Son fichier `execution-blocker.reference.json` est conservé octet pour octet dans les fixtures Core : 25 302 octets, SHA256 `062a71742134c9fdc93c78b48348446c077ae5d20c1309772ec3c9b43cba2a72`. Le protocole a le SHA256 `861118bdd79dc584185ad6f01fd63d89e5fde8345dd071913469b99ce1e40916`.

Le backend est `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`. Les six déclarations inchangées sont `ExecutionBlocker`, `_async_map_node_over_list`, `resolve_map_node_over_list_results`, `merge_result_data`, `get_output_from_returns` et le callback `execution_block_cb`. L'audit indépendant vérifie leurs fichiers Git, segments, AST et lignes, les trois fichiers publiés du laboratoire, l'ordre des 24 cas et leurs entrées inchangées. Il ne réexécute pas la source et n'utilise aucun résultat C# pour construire les expected.

`ExecutionBlockerReferenceTests` vérifie 23 comparaisons avec le vrai `EngineService` :

- 21 cas directs vérifient les appels réellement effectués, les slots fusionnés, les positions/messages des marqueurs, l'UI émise et les champs de diagnostic présents dans les deux systèmes. Les exceptions de la source deviennent des diagnostics moteur ; les types Python ne sont pas assimilés aux exceptions .NET.
- Deux cas lazy vérifient les lignes actives transmises au hook agrégé, leurs appariements après repeat-last et les noms de dépendances effectivement résolus. Les sorties et messages du job final ne sont pas comparés aux retours du mapper lazy : celui-ci a été capturé sans callback d'exécution.

L'entrée de test décode les tags `$blocker` exclusivement dans des nœuds sources typés. Aucun tag n'est envoyé comme sentinelle JSON au produit. Les corps des nœuds sont les petites opérations déclarées dans le protocole : echo, liste, constante, blocage total/partiel, UI et exception. L'exécution async utilise réellement `Task.Yield`, puis le moteur attend son résultat. Le nombre de Tasks Python en attente n'a pas d'équivalent contractuel côté moteur et n'est pas comparé.

Le 24e cas, `interrupt-before-block`, est conservé dans la fixture mais exclu de ces comparaisons : son adaptateur source interrompt au second contrôle interne du mapper, point que l'API .NET n'expose pas. L'annulation coopérative et le nettoyage restent couverts séparément par les tests comportementaux. Les contextes Python, indices de pre-execute, métadonnées client et champs websocket absents du contrat .NET ne sont pas inventés.

Les 23 comparaisons passent localement sous Windows, dans une suite Core de 136 tests. La compilation de la solution termine sans avertissement ni erreur ; la suite complète suivante passe **1 444 tests distincts**, sans échec ni test ignoré. Les résultats ciblés ne sont pas ajoutés à ce total. Les [empreintes et comptes de vérification](execution-blocker-source-fffa6b0.json) identifient les fichiers de référence et les TRX de cette exécution.

La CI exécute désormais Core, Workflow, Tokenization, Desktop headless et Host avant les portes de référence Inference. Le Host exerce aussi des primitives tensorielles ; cette organisation ne signifie pas que tous ces tests sont dépourvus de code natif. Les processus de référence Inference restent séparés pour préserver les contrôles du premier accès natif. Un échec numérique ultérieur conserve son statut d'échec.

Cette preuve locale ne porte pas sur PromptExecutor complet, le restaging lazy, les graphes dynamiques, V3, le cache persistant, les modèles ou la disponibilité V1. L'exécution de cette nouvelle tranche sur les autres OS reste à vérifier dans sa propre campagne CI.
