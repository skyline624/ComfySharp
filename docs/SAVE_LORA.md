# SaveLoRA : sauvegarde native des adapters

`SaveLoRA` est enregistré dans le Host et décrit dans son catalogue. Il reçoit
un `LORA_MODEL`, représenté par une map de tenseurs natifs possédés par le moteur,
et écrit un fichier safetensors dans le répertoire de sortie ComfySharp.
Il ne retourne ni valeur ni aperçu. Un `MODEL` patché ou un objet JSON ordinaire
ne remplace pas un `LORA_MODEL`.

Le contrat suit [SaveLoRA à la révision figée](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_extras/nodes_train.py#L1364) :
entrées `lora`, `prefix` et `steps` facultatif, préfixe par défaut
`loras/ComfyUI_trained_lora`, nœud de sortie expérimental, zéro sortie.
Le helper de nommage partagé reproduit aussi le compteur source : un fichier
`adapter_20_steps_00001_.safetensors` fait avancer le prochain compteur à 21.
Ce compteur suit les noms présents, pas un compteur SQL.

`SafeTensorWriter` conserve les dimensions, dtype et octets, y compris NaN/Inf.
Les dtypes pris en charge sont BOOL, U8, I8, I16, I32, I64, F16, BF16, F32 et F64.
Les tenseurs contigus sont copiés en mémoire CPU avant toute écriture ; le
producteur doit sérialiser les mises à jour pendant la capture du snapshot.
Le writer ne ferme pas le flux fourni et libère ses copies en cas d'erreur ou
d'annulation. Il refuse explicitement les autres dtypes, les vues non contiguës,
les rangs supérieurs à 16 et les snapshots dépassant son budget par défaut de
512 Mio. Ces limites restent à qualifier/étendre pour le catalogue complet.

Le stockage écrit dans un fichier temporaire adjacent puis le publie après
vérification de l'annulation. Une erreur avant publication conserve une éventuelle
destination et retire le temporaire. La politique de confinement des chemins
reste celle du stockage local ; elle ne verrouille pas l'arborescence contre
des modifications concurrentes par un autre processus.

Les tests vérifient le contrat, les noms, dix dtypes, les formes vides/scalaires,
les snapshots, la propriété native, les rejets, l'annulation et le stockage.
Un test HTTP du Host a relu puis exporté l'adapter SD1.5 de 1 250 tenseurs
entraîné lors de la campagne précédente. Les clés, dtypes, formes et hashes
des données sont identiques. Un décodeur indépendant confirme ces résultats.
Voir [la preuve et ses hashes](qualification/save-lora.json).

**Le parcours d'entraînement public reste incomplet.** Le producteur `LORA_MODEL`
du test Host est réservé aux tests, et n'est pas enregistré dans l'application.
`TrainLoraNode`, `LoraModelLoader` et le parcours complet de réutilisation restent
à raccorder. Ce travail ne rejoue ni entraînement ni génération et ne qualifie
aucune famille de modèles. La matrice conserve `realWorkflow=false` et les
plateformes non qualifiées. Aucun checkpoint n'est copié ou téléchargé.
