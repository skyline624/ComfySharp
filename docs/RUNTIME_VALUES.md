# Valeurs d'exécution et ressources natives

Le prompt HTTP et le document restent des données JSON. Les valeurs intermédiaires d'un graphe peuvent en revanche contenir des tenseurs, modèles, images ou structures composites. Leur durée de vie appartient au moteur et au Host, indépendamment de la sérialisation du document et du GC.

## Contrat des nœuds

`IRuntimeNode` reçoit un `RuntimeNodeContext` et un dictionnaire de `RuntimeValue`. Chaque entrée est empruntée pour la durée de l'invocation. Le contexte possède les valeurs qu'il crée avec `Json`, `Own`, `List` et `Map`. Les sorties rendues par le nœud restent empruntées jusqu'à leur acquisition par le moteur. Le contexte est libéré à la fin de l'invocation, y compris lors d'une exception ou d'une annulation.

`Own` transfère exactement une fois la propriété d'une ressource `IDisposable` au contexte. Une ressource déjà possédée doit être partagée en conservant sa `RuntimeValue`, sans appeler une seconde fois `Own` sur le même objet. Un nœud ne doit pas libérer les entrées qu'il emprunte, ni conserver leur accès après son retour sans acquérir explicitement une référence.

Une copie JSON est indépendante. La conservation d'une valeur native partage sa ressource sous-jacente : le moteur ne copie pas un tenseur entier à chaque connexion. Un nœud qui modifie un tenseur doit donc créer sa propre copie lorsque l'algorithme nécessite l'isolation de ses consommateurs. Les conteneurs possèdent des références sur leurs enfants et les libèrent déterministement.

Une liste JSON littérale et une liste de valeurs d'exécution sont distinctes. Les règles de mapping, répétition du dernier élément et concaténation des sorties s'appliquent aux listes d'exécution ; un tableau JSON littéral reste une seule valeur.

## Résultats et frontière HTTP

`EngineService.ExecuteValuesAsync` rend un `OwnedExecutionResult` à libérer avec `using`. Les résultats demandés conservent leurs ressources après la fin du calcul et de la mémoïsation du job. La dernière référence libère la ressource native. Aucun pointeur ou handle n'est envoyé au Desktop.

L'API `ExecuteAsync` existante projette ce même résultat vers le contrat JSON du socle. Une sortie native non sérialisable doit produire un diagnostic explicite et être libérée ; elle ne doit pas être sérialisée par réflexion. Les futurs nœuds de sortie fourniront des métadonnées et aperçus compatibles avec les contrats UI de ComfyUI.

Les implémentations `INode` du premier socle restent utilisables via un adaptateur JSON. Les nœuds qui transmettent des valeurs natives doivent employer le contrat natif, notamment le commutateur lazy.

## Intégration TorchSharp

Une vue tensorielle possède son propre wrapper natif et conserve le storage libtorch. Elle peut survivre au wrapper parent si sa propriété a été transférée correctement. Le comptage des références du moteur concerne les wrappers ; la propriété des storages entre vues reste celle de libtorch.

Un wrapper confié au contexte ne doit pas rester simultanément possédé par un `DisposeScope` TorchSharp. Les nœuds doivent détacher les sorties du scope de calcul avant leur transfert au contexte, puis laisser le contexte et le résultat posséder leur durée de vie. Les scopes continuent de libérer les temporaires de calcul.

Les tests de comptage de références vérifient les contrats de propriété. Les tests de pipelines TorchSharp vérifient également les wrappers, vues et calculs réels. Aucun de ces contrôles ne constitue une preuve de compatibilité d'une famille de modèles, de stabilité VRAM à long terme ou d'une politique d'offload complète.
