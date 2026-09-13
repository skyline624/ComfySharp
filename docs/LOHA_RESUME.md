# Reprise des paramètres LoHa

`SdTrainableAdapterSet` recharge maintenant les quatre facteurs LoHa, les deux
cores Tucker optionnels et alpha à partir d'un fichier ou d'un snapshot natif.
Les géométries sont vérifiées avant allocation ; le budget compte les tailles
réelles des facteurs, même lorsque les rangs des deux côtés diffèrent. Les
sources Float32, Float16, BFloat16 et Float64 sont converties en feuilles Float32.
Les sources restent indépendantes des nouvelles feuilles entraînables.

Le contrat est celui de la [fabrique d'entraînement figée](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_extras/nodes_train.py)
et de [LohaDiff](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/weight_adapter/loha.py) :

- La fabrique s'arrête au premier provider qui reconnaît un adapter : LoRA
  précède LoHa. Cette priorité diffère de celle du chargeur d'inférence.
- Alpha est lu dans `module.weight.alpha`. Le champ habituel sauvegardé
  `module.alpha` est ignoré et la valeur par défaut est 1, comme dans la source.
- Les différences de normalisation et de biais sont réinitialisées à zéro.
- Le format existant détermine le type du paramètre repris. L'option d'algorithme
  choisit seulement les nouveaux adapters des couches sans facteurs existants.
- Reprendre LoHa n'exécute aucun constructeur Linear et ne consomme pas de RNG.
  Les couches LoRA conservent leurs tirages de construction source.
- DoRA n'entre pas dans `LohaDiff`. Alpha n'a pas de gradient source. Le bypass
  d'entraînement reste explicitement refusé si une couche utilise LoHa.

La référence `loha-resume.reference.json.gz` provient du laboratoire séparé
`labs/lora-training-source/loha_resume.py`. Quatre scénarios couvrent les deux
topologies SD réduites, des couches LoRA/LoHa reprises, des couches fraîches,
Tucker, les dtypes et les priorités. Les valeurs reprises sont comparées
exactement ; les autres paramètres utilisent le profil fixé `atol=rtol=3e-5`.
L'état final du RNG est comparé exactement par SHA-256. La fixture est statique,
compressée et verrouillée par hash ; les tests .NET n'exécutent pas Python.

Des tests supplémentaires effectuent une mise à jour sur les cibles reprises,
puis un rechargement en inférence avec prédiction exactement identique. Ils
contrôlent gradients, absence de modification du modèle de base, budgets,
annulation et libération après une entrée invalide.

```text
dotnet run -c Release --project tools/ComfySharp.RuntimeProbe -- sd-all-adapter-train --algorithm LoHa --checkpoint <existant.safetensors> --resume <adapter-existant.safetensors> --report <nouveau.json> --device cpu
```

Ce diagnostic lit les fichiers existants, poursuit deux mises à jour puis
recharge les facteurs en mémoire. Il n'écrit aucun poids sans `--adapter`.
Le compteur extrait du nom de fichier est une fonction distincte déjà testée ;
ce diagnostic ne reproduit pas encore le nœud public complet.

Voir [les preuves et limites](qualification/loha-resume.json). La reprise des
facteurs ne signifie pas restauration d'un état complet d'optimiseur : la
source initialise un nouvel optimiseur. Le nœud complet, les datasets réels,
la précision mixte, l'offload, les autres algorithmes et la qualification des
plateformes demeurent requis. Les restrictions du backward Tucker source sur
les rangs `i` et la géométrie sont conservées explicitement.
