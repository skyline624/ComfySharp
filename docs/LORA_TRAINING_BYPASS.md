# Entraînement SD par adapters de calcul

`SdLoraTrainingOptions.BypassMode` sélectionne maintenant le calcul des facteurs
LoRA dans les couches linéaires et Conv2d, jusqu'au débruiteur et à l'objectif
d'entraînement. Le mode ordinaire reste la valeur par défaut. Les différences
de normalisation et de biais suivent toujours le chemin de modification des poids.

Le coefficient alpha conserve son graphe de gradients : aucune lecture scalaire
ne le détache pendant le calcul. Le port suit `LoraDiff.h` et
`_setup_lora_adapters_bypass` du backend figé. Les paramètres appartiennent à
l'appelant ; le calcul retient leurs propriétaires et libère ses ressources
temporaires. Les poids de base restent gelés. Le budget des poids modifiés
s'applique aux différences, pas aux facteurs bypass ; activations et état de
l'optimiseur consomment toujours de la mémoire supplémentaire.

Deux scénarios source couvrent les graphes SD1 et SD2 à largeur réduite : six
cibles LoRA (temps, entrée, attention Q, projection spatiale, réduction et sortie),
une normalisation et un biais. Ils comparent les sorties, pertes, gradients de
tous les paramètres, y compris alpha, et deux mises à jour SGD. Le laboratoire
exécute les classes figées `LoraDiff`, `BiasDiff` et `BypassForwardHook`, avec
le véritable `_forward` réduit. Les tolérances absolue et relative sont fixées
à `3e-5` avant comparaison. Les tests .NET lisent la fixture sans Python.

Le snapshot natif entraîné est ensuite rechargé par le lecteur LoRA commun et
utilisé en inférence bypass, après destruction de ses producteurs. Les tests de
lots couvrent les modes standard, multirésolution et buckets avec accumulation,
ainsi que l'annulation avant la première mise à jour. **11 tests ciblés passent,
dont sept nouveaux ; les 935 tests ordinaires d'inférence passent localement.**

Cette tranche couvre les nouveaux adapters à deux facteurs en Float32 sur le
U-Net SD existant. La reprise des variantes d'adapters, la précision mixte,
le checkpointing, l'offload, les autres architectures et le nœud public
`TrainLoraNode` restent à compléter. Le succès d'un scénario synthétique ne
qualifie ni un entraînement sur images ni une famille complète.

Le diagnostic sur le checkpoint SD1.5 partagé exécute deux mises à jour SGD
sur **686 cibles et 1 250 paramètres**, dont 282 coefficients alpha. Tous les
gradients sont finis ; 280 coefficients alpha ont un gradient non nul à la seconde
étape. Les poids de base restent inchangés et le snapshot rechargé en mémoire
reproduit exactement la prédiction entraînée. Les entrées sont synthétiques
(latent 8 × 8 et contexte brut) ; aucun modèle ou adapter n'est écrit ni copié.
Cette exécution ne compare pas les gradients préentraînés à la source.

```text
dotnet run -c Release --project tools/ComfySharp.RuntimeProbe -- sd-all-adapter-train --checkpoint <checkpoint.safetensors> --report <nouveau-rapport.json> --device cpu --bypass true
```

Voir [les preuves](qualification/lora-training-bypass.json), le
[collecteur source](../labs/lora-training-source/bypass.py) et
[le chargement bypass en inférence](LORA_BYPASS.md).
