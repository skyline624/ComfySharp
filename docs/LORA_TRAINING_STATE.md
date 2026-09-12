# État natif en sortie d'entraînement

`LoraTrainingState.Capture` transforme les adapters ordinaires entraînables en
un état `LORA_MODEL` indépendant, sans fichier intermédiaire. Les poids de base
ne font pas partie de cet état. `TrainingNodeValues.CaptureAdapters` publie ses
tenseurs dans une map du moteur, compatible avec `SaveLoRA`.

Le nommage reprend les facteurs `lora_up.weight`, `lora_down.weight`, `alpha`,
les différences de poids `diff` et de biais `diff_b`. Les 686 cibles SD ordinaires
produisent 1 250 tenseurs. Le paramètre de sortie accepte `bf16` et `fp32`, les
deux choix de [TrainLoraNode dans la source figée](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_extras/nodes_train.py#L1034).
Cette conversion finale n'ajoute pas l'entraînement en précision mixte : les
paramètres entraînables actuels restent Float32.

La capture conserve le périphérique et détache les gradients. Elle possède une
copie indépendante des paramètres ; une mise à jour ou la destruction des
adapters d'origine ne modifie pas cette copie. Le code appelant doit sérialiser
les mises à jour pendant la capture. Le moteur retient ensuite des handles
natifs vers ce stockage, sans recopier ses données pour chaque consommateur.
Les scopes natifs intermédiaires sont libérés par tenseur ; une annulation ou
un échec de publication libère les allocations sans prendre possession des
paramètres d'entraînement. Le budget de snapshot par défaut reste 512 Mio.

Le [collecteur indépendant](../labs/save-lora-source/collect_state.py) extrait
et exécute uniquement la boucle finale `to(lora_dtype).detach()` depuis le
commit de référence. Les tests .NET comparent exactement les octets Float32 et
BFloat16, notamment le zéro signé et deux cas d'arrondi à mi-distance.
La fixture est verrouillée par SHA-256. Les autres tests couvrent les 1 250
tenseurs, une mise à jour SGD, la conservation après destruction du producteur,
la sauvegarde via `SaveLoRA` et les erreurs de propriété/budget/annulation.
Voir [la preuve](qualification/lora-training-state.json).

Ce pont ne constitue pas un `TrainLoraNode` public. Il ne porte pas encore la
reprise depuis un adapter existant, les modes bypass, le checkpointing, l'offload,
la précision mixte ni `LoraModelLoader`. Ces capacités restent dans le périmètre
de la V1 ; aucun nœud, workflow complet ou modèle n'est déclaré qualifié par ces
tests synthétiques. Aucun modèle n'est téléchargé ou copié pour ce travail.
