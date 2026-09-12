# Application d'un adapter en mémoire

`LoraModelLoader` est enregistré dans le Host. En mode ordinaire (`bypass=false`),
il applique une map native `LORA_MODEL` à une copie d'un U-Net SD Float32.
Une force nulle conserve le modèle original sans lire l'adapter. Les forces
négatives sont prises en charge ; la force et les entrées restent validées.

`ILoraTensorSource` permet au même code d'inspection et de chargement de lire
un fichier safetensors ou un état natif. Les formats, alias, priorités, différences
additives et limites ne sont pas réimplémentés dans le nœud. `NativeLoraTensorSource`
capture les valeurs sur CPU avec leur dtype et leurs dimensions ; ses lectures
possèdent un stockage indépendant. Le plan ne peut être chargé qu'avec la source
qui l'a produit. La destruction du producteur ou de la source ne détruit pas
les patches déjà chargés. Aucun fichier intermédiaire n'est créé par ce chemin.

Les clés non utilisées et les alias masqués apparaissent dans le diagnostic
`comfysharp_lora`. Cette sortie de diagnostic est une extension ComfySharp.
Le budget de snapshot est de 512 Mio ; le nœud autorise jusqu'à 4 Gio de poids
modifiés, en plus des poids et allocations déjà présents. Ces budgets ne sont
pas une garantie de mémoire disponible.

Le contrat suit [LoraModelLoader dans la source figée](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_extras/nodes_train.py#L1313).
**Le mode bypass reste à porter et reçoit un diagnostic explicite.** La source
[sépare les adapters de calcul des patches ordinaires](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/sd.py#L138) :
les facteurs LoRA sont appliqués dans les opérations de forward, tandis que les
différences de poids/biais passent par les patches ordinaires. Une modification
des poids n'est donc pas utilisée pour simuler ce mode. Les autres architectures,
la quantification et les champs descriptifs de schéma non encore représentés
dans le contrat commun restent à compléter.

Les nouveaux tests comparent les sept écritures de facteurs et les différences
de biais par fichier et par mémoire. Ils vérifient F32/F16/BF16, les vues
non contiguës, l'indépendance des lectures, l'identité des plans et les échecs.
Quatre parcours sur un U-Net réduit comparent exactement la prédiction d'un
adapter modifié par SGD, capturé en fp32/bf16 puis chargé avec force positive
ou négative. Les poids originaux restent inchangés.

Ces scénarios sont synthétiques. Le producteur public `TrainLoraNode`, le bypass
et le workflow complet restent ouverts ; aucune famille préentraînée n'est
qualifiée par ce raccordement. Voir [les résultats](qualification/lora-memory-loading.json).
