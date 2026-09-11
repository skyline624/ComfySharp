# Valeurs d'exécution et ressources natives

Le prompt HTTP et le document restent des données JSON. Les valeurs intermédiaires d'un graphe peuvent en revanche contenir des tenseurs, modèles, images ou structures composites. Leur durée de vie appartient au moteur et au Host, indépendamment de la sérialisation du document et du GC.

## Contrat des nœuds

`IRuntimeNode` reçoit un `RuntimeNodeContext` et un dictionnaire de `RuntimeValue`, puis retourne `NodeExecutionOutput(Result, Ui)`. `Result` contient les slots internes ; `Ui`, facultatif, contient uniquement des données JSON d'affichage. Chaque entrée est empruntée pour la durée de l'invocation. Le contexte possède les valeurs qu'il crée avec `Json`, `Own`, `List` et `Map`. Les sorties restent empruntées jusqu'à leur acquisition par le moteur. Le contexte est libéré à la fin de l'invocation, y compris lors d'une exception ou d'une annulation.

`Own` transfère exactement une fois la propriété d'une ressource `IDisposable` au contexte. Une ressource déjà possédée doit être partagée en conservant sa `RuntimeValue`, sans appeler une seconde fois `Own` sur le même objet. Un nœud ne doit pas libérer les entrées qu'il emprunte, ni conserver leur accès après son retour sans acquérir explicitement une référence.

Une copie JSON est indépendante. La conservation d'une valeur native partage sa ressource sous-jacente : le moteur ne copie pas un tenseur entier à chaque connexion. Un nœud qui modifie un tenseur doit donc créer sa propre copie lorsque l'algorithme nécessite l'isolation de ses consommateurs. Les conteneurs possèdent des références sur leurs enfants et les libèrent déterministement.

Une liste JSON littérale et une liste de valeurs d'exécution sont distinctes. Les règles de mapping, répétition du dernier élément et concaténation des sorties s'appliquent aux listes d'exécution ; un tableau JSON littéral reste une seule valeur.

## Résultats et frontière HTTP

`EngineService.ExecuteValuesAsync` rend un `OwnedExecutionResult` à libérer avec `using`. Les résultats demandés conservent leurs ressources après la fin du calcul et de la mémoïsation du job. La dernière référence libère la ressource native. Aucun pointeur ou handle n'est envoyé au Desktop.

L'API interne `ExecuteAsync` projette ce même résultat vers le contrat JSON du socle. Une sortie native non sérialisable produit un diagnostic explicite et est libérée ; elle ne doit pas être sérialisée par réflexion.

Le Host utilise `ExecuteUiAsync`, qui libère les ressources natives avant de rendre des snapshots UI et métadonnées gérés. Les sorties d'affichage des ancêtres sont conservées, pas seulement celles des cibles finales. Les exécutions mappées concatènent les tableaux UI dans l'ordre, en conservant les clés du premier retour. Les valeurs UI natives cachées dans un `JsonValue`, les tableaux mal formés et les nombres non JSON sont rejetés. Le Host enregistre l'historique avant de publier le terminal ; un échec de libération ne peut pas être annoncé comme succès.

`POST /prompt` sélectionne uniquement les vrais `OUTPUT_NODE` et refuse tout type de nœud absent, même hors des dépendances choisies. Les cibles internes explicites restent disponibles aux outils via le moteur C#, sans devenir des sorties HTTP. `PreviewAny` produit une UI `{"text":["…"]}` et un slot STRING indépendant. Voir [les détails](NATIVE_NODE_HOST.md).

Les implémentations `INode` du premier socle restent utilisables via un adaptateur JSON. Les nœuds qui transmettent des valeurs natives doivent employer le contrat natif, notamment le commutateur lazy.

## Intégration TorchSharp

Une vue tensorielle possède son propre wrapper natif et conserve le storage libtorch. Elle peut survivre au wrapper parent si sa propriété a été transférée correctement. Le comptage des références du moteur concerne les wrappers ; la propriété des storages entre vues reste celle de libtorch.

Un wrapper confié au contexte ne doit pas rester simultanément possédé par un `DisposeScope` TorchSharp. Les nœuds inscrivent le wrapper dans le contexte puis le détachent immédiatement du scope ; un échec du transfert laisse ainsi le scope libérer l'allocation. Le contexte et le résultat possèdent ensuite sa durée de vie. Les scopes continuent de libérer les temporaires de calcul.

Les tests de comptage de références vérifient les contrats de propriété. Les tests de pipelines TorchSharp vérifient également les wrappers, vues et calculs réels. Aucun de ces contrôles ne constitue une preuve de compatibilité d'une famille de modèles, de stabilité VRAM à long terme ou d'une politique d'offload complète.
