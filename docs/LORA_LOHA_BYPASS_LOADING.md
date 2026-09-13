# Chargement LoRA et LoHa en mode bypass

Le mode d'inspection `Bypass` valide désormais les opérateurs LoRA et LoHa,
en complément de LoKr. Le mode ordinaire conserve ses contrôles de reconstruction
et reste le mode par défaut. Un plan inspecté pour le bypass ne permet pas de
modifier les poids par le chemin ordinaire.

Pour LoRA, la validation suit la chaîne `down → mid → up` : canaux intermédiaires,
facteur intermédiaire matriciel ou spatial, matrices aplaties selon le noyau du
module et couches linéaires ou convolutions 1D/2D/3D. Les opérations natives
acceptent les noyaux spatiaux successifs de la source. Le reshape des matrices
en convolution utilise `view`, comme dans la définition figée.

Pour LoHa, le produit de Hadamard est toujours reconstruit à chaque appel, puis
mis à l'échelle avant l'opérateur. Il peut être utilisé par une couche linéaire
ou une convolution 1D/2D/3D. Les cœurs Tucker d'inférence suivent toujours
l'équation source à quatre dimensions : ils ne deviennent pas arbitrairement
des cœurs 1D ou 3D.

Les trois providers de bypass conservent leur échelle DoRA sans l'utiliser.
L'addition native autorise la diffusion des dimensions de taille 1, conformément
à l'injection source. Les incompatibilités spatiales qui dépendent des entrées
restent vérifiées à l'exécution ; elles ne sont pas transformées en succès.

Les sources de référence sont les définitions
[LoRA](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/weight_adapter/lora.py),
[LoHa](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/weight_adapter/loha.py)
et [l'injection de bypass](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/weight_adapter/bypass.py).

## Preuves

`labs/lora-training-source/lora_loha_bypass_loading.py` exécute les classes figées
dans un laboratoire séparé. La fixture contient 96 cas : 72 calculs réussis et
24 erreurs attendues. Les facteurs F32/F16/BF16 sont convertis vers les activations
Float32, comme dans la source. Les tests .NET chargent des safetensors synthétiques
et des snapshots en mémoire, vérifient les références, les budgets, l'annulation,
la conservation des données après libération des producteurs et les erreurs.

Les mêmes tests contrôlent les refus ordinaires sur les géométries incompatibles,
sans l'échelle DoRA ignorée pour que le refus ne repose pas sur ce champ. Deux
tests du nœud public `LoraModelLoader` utilisent un U-Net réduit : chaîne LoRA
avec facteur intermédiaire matriciel et diffusion des canaux LoHa. Les modèles
restent utilisables après libération et mutation des facteurs d'origine, et la
base est inchangée. Le budget de poids reconstruits est nul ; LoHa construit
toutefois son produit temporaire pendant chaque forward, comme la source.

Les contrôles SD1.5 utilisent les fichiers de checkpoint et d'adapters déjà
présents dans le dossier partagé. Le diagnostic inspecte séparément les modes
ordinaire et bypass, libère ses sources puis répète les prédictions de bypass.
Les écarts entre modes sont des observations, pas une acceptation de précision
du modèle entier. Les résultats figurent dans [la preuve](qualification/lora-loha-bypass-loading.json).

Les 22 150 octets de référence compressée sont synthétiques. Les petits fichiers
safetensors créés par les tests sont temporaires et supprimés ensuite. Aucun
nouveau modèle préentraîné n'est nécessaire.

## Travail restant

Les activations en précision mixte, les modèles hors SD, les GPU et les workflows
d'entraînement sur images restent à qualifier. `reshape_weight`, les injections
sur modèles quantifiés et les algorithmes supplémentaires restent ouverts.
LoHa conserve l'absence de bypass d'entraînement dans la source ; aucun chemin
d'inférence n'est présenté comme un entraînement LoHa en bypass. Les profils
numériques restent `atol=rtol=3e-5`, sans augmentation ni test ignoré.
