# Fondations d'entraînement OFT

`TrainableOftPatch` porte `OFTDiff.__call__`, `h` et `g` depuis la
[source figée](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/weight_adapter/oft.py).
Les blocs Float32 et le redimensionnement optionnel possèdent leur propre stockage,
des propriétaires retenus et une libération déterministe. Ils fonctionnent avec
les optimisateurs C#, les snapshots `LORA_MODEL` et l'export safetensors.

La transformation de Cayley conserve la normalisation globale de la matrice
antisymétrique, y compris les contraintes négatives. L'alpha exporté est une
feuille activée par le contrat d'entraînement, mais la contrainte est capturée
comme un nombre à la construction : alpha ne reçoit pas de gradient et sa
mutation ultérieure ne change pas la contrainte. Les appels natifs gardent
l'ordre du calcul et l'inversion Float32 amont.

La modification des poids accepte F32/F16/BF16, avec feuilles F32. Le bypass
actuel accepte les activations F32 et reproduit `g(base_out + zeros_like(base_out))`.
Le type de module est explicite : une projection linéaire avec sortie de rang
trois reste linéaire. Les sorties convolutionnelles 1D/2D/3D sont transposées,
regroupées en blocs et remises dans leur disposition initiale. Le redimensionnement
ordinaire utilise la diffusion des dimensions ; celui du bypass est aplati,
conformément à la source. Les poids et bypass peuvent donc différer, notamment
en présence d'un biais, qui est transformé par le bypass.

## Références et vérifications

Le collecteur indépendant `labs/lora-training-source/oft_training.py` exécute
les définitions AST amont sous PyTorch 2.10 CPU. Ses 192 cas couvrent les quatre
rangs de poids, les sorties convolutionnelles et linéaires, quatre contraintes,
trois formes de redimensionnement, les dispositions non contiguës, les gradients
d'entrée et de paramètres, une mise à jour SGD et la mutation d'alpha. Les
tolérances absolue et relative sont verrouillées à `3e-5` dans la fixture avant
la comparaison C#. Les tests distribués lisent cette fixture sans Python.

Cinq autres tests couvrent les erreurs, l'annulation, les durées de vie, deux
exports/snapshots F32/BF16 et deux entraînements de U-Net réduit. Le bypass OFT
ne matérialise pas les poids modifiés et s'exécute avec un budget correspondant nul.
Le redimensionnement conserve des copies indépendantes après mutation ou
libération des propriétaires initiaux.

Le diagnostic `sd-oft-train --checkpoint EXISTING.safetensors --report NEW.json`
utilise explicitement le SD1.5 partagé, sans téléchargement ni export de poids.
Il entraîne `out.2.weight` et
`input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight` pendant deux pas SGD
à `0.01`, dans chaque mode, sur CPU Float32 avec 16 threads. Le checkpoint est
identifié par SHA-256 dans la [preuve](qualification/oft-training.json).
Les deux blocs reçoivent des gradients finis et non nuls ; les pertes passent
de `1.1856256` à `1.184864` en modification des poids, et de `1.1856257` à
`1.1848655` en bypass. Les poids de base restent inchangés. La reconstruction
explicite des feuilles depuis leur snapshot reproduit exactement la prédiction
entraînée. Cette opération ne passe pas par le chargeur d'inférence ni par la
fabrique de reprise ; elle ne les valide pas.

## Périmètre restant

La fabrique complète OFT, la reprise avec ordre des fournisseurs et le chargement
d'inférence restent à porter. `OFTAdapter.calculate_weight` applique notamment
la force deux fois dans son chemin ordinaire, ignore le redimensionnement et
ne traite une contrainte que si elle est positive : il ne faut pas le remplacer
par l'opérateur d'entraînement. DoRA nécessite également son propre chemin.
Les activations de bypass en précision réduite, les transferts, CUDA/MPS,
le nœud public `TrainLoraNode` et les workflows sur images restent ouverts.
Ce jalon ne qualifie ni une famille complète, ni une plateforme, ni la V1.
