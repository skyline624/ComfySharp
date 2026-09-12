# Littéraux de prompt et casse des clés HTTP

Le Host préserve désormais la casse des clés JSON reçues, et Desktop conserve celle des réponses : les IDs de nœuds, noms d'entrées et clés des dictionnaires ne sont plus confondus lors de ces désérialisations HTTP. **245 tests ciblés passent**, dont 33 nouveaux. La [preuve JSON](prompt-literals-host.json) conserve les quatre TRX, les hashes des fichiers, les sources relues et un smoke Desktop réel séparé.

Le défaut a été révélé par un vrai parcours `PromptCompiler → /prompt → JobQueue → PreviewAny → /history`. L'objet `{"__VALUE__":1,"nested":{"__value__":2}}` produisait `1`, alors que le dictionnaire devait être conservé. Les options JSON Web rendaient les clés du `JsonObject` insensibles à la casse ; `EngineService.Unwrap` retrouvait donc `__VALUE__` en demandant `__value__`. Le même mécanisme pouvait confondre des IDs ou inputs différant uniquement par leur casse.

La correction ajoute, au démarrage du Host, `ConfigureHttpJsonOptions` avec `PropertyNameCaseInsensitive=false`, précédé d'un commentaire. Ce sont exactement les deux lignes ajoutées à `Program.cs`. La portée est celle des options JSON HTTP des routes minimal API, pas une transformation réservée au seul `/prompt`. Le compilateur et le déballage dans Core restent inchangés.

## Résultats ciblés

| Exécution conservée | Succès | Échecs | Ignorés |
|---|---:|---:|---:|
| Workflow avec les nouveaux tests de littéraux | 49 | 0 | 0 |
| Host avant correction HTTP | 153 | 1 | 0 |
| Host après correction HTTP et sept régressions de casse | 161 | 0 | 0 |
| Desktop après correction de la lecture des réponses | 35 | 0 | 0 |

Les **245 succès** correspondent à Workflow 49 + Host corrigé 161 + Desktop 35, trois ensembles disjoints. Le Host initial de 154 cas recouvre le second et n'est pas additionné ; son unique échec sur `__VALUE__` reste dans la preuve. Les compteurs, résultats individuels et erreurs globales des quatre TRX ont été vérifiés en lecture seule. Le smoke ne compte pas comme un test xUnit supplémentaire.

Les 33 nouveaux cas comprennent :

- **18 Workflow** : tableaux enveloppés, objets et null transmis, persistance 0.4/1.0, layouts de widgets nommés/positionnels, clones indépendants, priorité d'une connexion réelle, undo/redo.
- **8 Host de littéraux** : vrai compilateur, validation HTTP, queue et nœuds `PreviewAny`/`PrimitiveString`, enveloppes explicites, collisions de clé réservée et déballage unique.
- **7 Host de casse** : deux IDs `item`/`ITEM` réellement exécutés, `SOURCE` seul refusé avant enregistrement d'un job, coexistence de `source`/`SOURCE` et `__value__`/`__VALUE__` dans les deux ordres, dictionnaires imbriqués préservés sans déballage récursif.

Les fixtures de compilation déclarent explicitement un widget `source` pour le vrai `PreviewAny`. Elles ne changent pas sa définition produit sans widget persistant et ne remplacent aucun nœud côté Host. Les comparaisons d'objets portent sur le contenu JSON, sans imposer une nouvelle équivalence de formatage Python.

## Lecture Desktop et smoke réel

`HostSupervisor` présentait le même risque en lisant les réponses avec les options Web implicites de `GetFromJsonAsync`. Il utilise maintenant explicitement `JsonSerializerOptions.Default` pour la requête de santé et pour sa méthode GET générique, notamment l'historique. Le diff vérifié ajoute seulement l'import JSON et ces deux arguments. Il ne change pas globalement les règles de `JsonNode`, ni ne reconstruit les réponses avec un objet factice.

Les 35 tests Desktop passent après la modification. Le smoke conservé ouvre aussi la vraie fenêtre Desktop, démarre le Host supervisé et exécute les parcours texte et sigmas CPU déjà présents. Une nouvelle étape soumet directement deux vrais `PreviewAny` aux IDs `preview` et `PREVIEW`, puis lit l'historique via le même `HostSupervisor.GetAsync` que l'application. Le smoke exige deux clés distinctes et leurs textes respectifs `lowercase`/`uppercase`. Cette étape vérifie la réponse reçue ; elle ne revendique pas l'édition visuelle de ces deux nœuds dans le document.

Les fichiers retenus indiquent un **exit code 0**, le message de réussite incluant l'historique sensible à la casse et un stderr vide. Le script sélectionne explicitement les apphosts Desktop Release et Host Release Windows CPU, avec un `data-dir` isolé. Son `finally` prévoit l'arrêt/disposal du processus si nécessaire et restaure les deux variables d'environnement antérieures ; la fermeture de la fenêtre dispose son Host supervisé. Les hashes du script et des trois fichiers de résultat figurent dans le JSON. Il s'agit d'une preuve de processus et de ses assertions, pas d'un manifeste d'identité des bibliothèques natives. Le relecteur n'a pas relancé le smoke.

## Provenance et limites

Le [frontend figé, `executionUtil.ts`](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/src/utils/executionUtil.ts#L111), enveloppe les tableaux ordinaires et conserve les objets ; les connexions sont ensuite écrites à la place des widgets. Le blob LF vérifié compte 4 794 octets, SHA-256 `264e11395f28880603e7f9fe36d2e61c74cd59d6d88be990225cce8510e3f816`. Sa branche spéciale `curve`, les serializers asynchrones et la résolution complète LiteGraph ne sont pas couverts par cette tranche.

Le [backend figé, `execution.py`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L987), prend aussi en charge la clé exacte `__value__`. Son blob vérifié compte 61 543 octets, SHA-256 `e4058e4cc03e89753f41a62f51d4534933a35e243358aa6796a65a9fd0bdda28`. Les tests présents décrivent le contrat C# : ils ne constituent pas une exécution de cette source et ne prouvent pas la parité de toute la validation HTTP amont, qui peut modifier les inputs avant leur résolution. Le laboratoire Autogrow collecté séparément n'est pas utilisé comme oracle de ces tests HTTP.

La preuve enregistre les hashes bruts et canoniques LF des trois fichiers de tests, de `Program.cs`, de `HostSupervisor.cs`, de `MainWindow.axaml.cs`, du projet Host.Tests et de ses quatre locks. La référence `Host.Tests → Workflow` reste une dépendance de tests. Les locks régénérés par la racine ajoutent uniquement ce projet ; la comparaison JSON avec leur base confirme que les versions, hashes et autres enregistrements des packages externes sont inchangés.

La base de lecture est `8dad64f0cf9b04548ee722b7533776b25d3b4c46`, qui ne contenait pas encore ces changements testés. Les empreintes décrivent donc leur état non commité lors de l'audit, sans attestation artificielle d'un commit global. La racine rapporte une compilation de solution sans avertissement ni erreur avant la correction HTTP, puis une compilation du projet Host après celle-ci et du projet Desktop.Tests après la correction client, également sans avertissement ni erreur. Les TRX attestent les résultats ciblés ; ils ne prouvent pas à eux seuls ces commandes de compilation.

Aucune campagne complète de 1 550 tests, qualification multi-OS, nouvelle référence source exécutée ou compatibilité de modèles natifs n'est revendiquée ici.
