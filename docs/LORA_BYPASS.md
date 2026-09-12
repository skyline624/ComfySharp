# LoRA appliqué pendant le calcul

`LoraModelLoader` accepte `bypass=true` pour les U-Net SD Float32. Les facteurs
sont appliqués aux activations dans les couches linéaires et Conv2d :
`sortie_base + up(mid(down(entrée))) * force * alpha/rang`, avec `mid` facultatif
et un facteur alpha/rang égal à 1 lorsque alpha est absent. Les différences
additives de poids ou de biais restent des patches ordinaires.

Le port adapte `LoRAAdapter.h`, `WeightAdapterBase.g/bypass_forward` et la
séparation des patches dans `load_bypass_lora_for_models` du
[backend figé](https://github.com/comfy-org/ComfyUI/tree/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a).
Les projections Q/K/V sans biais, les convolutions de réduction spatiale et les
projections linéaires SD2 passent par ce chemin. Avec un facteur intermédiaire,
stride et padding s'appliquent à celui-ci ; sinon ils s'appliquent à down.
Le champ DoRA est ignoré par le calcul bypass de cette source ; le test dédié
reproduit ce comportement. Il ne s'agit pas d'une qualification générale DoRA.

Les facteurs possèdent leurs ressources natives, indépendamment du fichier,
de la map productrice et des autres propriétaires du modèle. Retain, transfert
de périphérique et destruction conservent cette propriété. Un nouveau groupe
de facteurs remplace le groupe bypass précédent, conformément à la source ;
une application contenant seulement des différences conserve le groupe présent.
Les facteurs seuls ne consomment pas le budget des poids modifiés. Leurs snapshots
et les activations consomment néanmoins de la mémoire ; les différences restent
soumises au budget ordinaire. Aucun fichier intermédiaire n'est nécessaire.

Six fixtures comparent sorties et gradients aux fonctions Python figées exécutées
dans un laboratoire séparé. Les tolérances absolue et relative sont fixées à
`8e-6` avant validation. Huit autres tests couvrent les couches du modèle réduit,
le remplacement des groupes, les différences, la répétabilité, l'annulation et
les ressources. Un test supplémentaire traverse le nœud public avec budget de
poids modifiés nul. Les **928 tests ordinaires d'inférence passent localement**,
sans échec ni test ignoré ; les tests CLIP stock constituent une suite séparée.

Un diagnostic utilise le checkpoint SD1.5 et un adapter déjà entraîné depuis
le stockage partagé, sans copie ni téléchargement. Il charge **686 cibles**,
répète trois prédictions identiques et vérifie que l'adapter a un effet sans
modifier le modèle de base. L'écart absolu maximal observé par rapport au mode
ordinaire est `3.933906555175781e-6`. Cette mesure n'est pas un critère d'acceptation
numérique du modèle. Les entrées sont un latent 8 × 8 et un contexte synthétiques ;
aucune génération ni nouvel entraînement n'est réalisé par ce diagnostic.

Commande reproductible avec des fichiers existants et un nouveau rapport :

```powershell
dotnet run -c Release --project tools/ComfySharp.RuntimeProbe -- sd-lora-bypass --checkpoint <checkpoint.safetensors> --adapter <adapter.safetensors> --report <nouveau-rapport.json>
```

Voir [la preuve et les hashes](qualification/lora-bypass.json) et
[le laboratoire source](../labs/lora-bypass-source/README.md).
Le [chemin d'entraînement bypass avec alpha entraînable](LORA_TRAINING_BYPASS.md)
est maintenant raccordé à la boucle SD Float32. Restent ouverts : précision mixte,
quantification, Conv1d/Conv3d, autres architectures, nœud d'entraînement complet,
workflows publics et qualification des plateformes. **Aucune famille n'est
déclarée entièrement compatible par ces résultats.**
