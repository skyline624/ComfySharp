# Entraînement LoHa : fondations SD

`TrainableLohaPatch` possède quatre facteurs Float32, les deux cores Tucker
optionnels et alpha. Il s'intègre au chemin d'entraînement ordinaire du U-Net,
aux quatre optimisateurs existants, au snapshot `LORA_MODEL` et à l'export
safetensors. Les clés `hada_w1_a`, `hada_w1_b`, `hada_w2_a`, `hada_w2_b`,
`hada_t1`, `hada_t2` et `alpha` sont conservées ; aucun adapter n'est converti
en simple différence de poids lors de la sauvegarde.

Le contrat vient de [LohaDiff, HadaWeight et HadaWeightTucker de la source
figée](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/weight_adapter/loha.py).
Deux particularités sont reproduites explicitement :

- La fabrique active `requires_grad` sur alpha, mais le backward source ne
  retourne aucun gradient pour ce paramètre. Les optimiseurs conservent cette
  absence, y compris AdamW : alpha ne reçoit pas de décroissance de poids.
- Le backward Tucker calcule le gradient de chaque facteur `a` avec le tenseur
  intermédiaire de l'autre côté. Une composition naïve d'einsum aurait les mêmes
  sorties et des gradients différents. Le port utilise des termes de valeur
  nulle et des coupures de gradient pour reproduire ce Jacobien de premier ordre
  avec les opérations natives. Les autres gradients restent ceux de la source.

Les rangs des deux axes `i` doivent correspondre pour le backward Tucker source ;
les autres rangs peuvent différer. Les géométries incompatibles sont refusées
avant allocation des paramètres. Les dérivées d'ordre supérieur ne sont pas
qualifiées. Le `LohaDiff` entraînable source n'implémente pas `h()` : demander
le bypass d'entraînement produit une erreur explicite. Le bypass du chargeur
LoHa d'inférence est une [capacité distincte maintenant portée pour SD](LOHA_INFERENCE.md).

`SdTrainableAdapterSet(..., algorithm: "LoHa")` crée les adapters de toutes les
couches SD et conserve les différences de normalisation/biais. Les tirages
`normal_(tensor, 0.1)` utilisent une **moyenne de 0,1 et un écart-type de 1**, comme
la source ; le quatrième facteur utilise une moyenne de 0,01. Le deuxième est
nul. Aucun constructeur Linear ne consomme de RNG supplémentaire pour LoHa.
Les générateurs privés et le budget couvrent les **686 cibles et 1 814 paramètres**.
La reprise existante reste celle de LoRA ; les fichiers LoHa ne sont pas encore
admis par les fabriques de reprise. Le chargeur d'inférence accepte leurs facteurs.

Les collecteurs Python sont limités au laboratoire séparé. Seize références
matrice, Conv2d et Tucker 1D/2D comparent toutes les sorties, tous les gradients
et 32 mises à jour SGD/Adam/AdamW/RMSprop. Le profil fixé est `atol=rtol=3e-5`.
Une autre référence couvre tous les paramètres de la fabrique SD et l'état RNG,
avec comparaison numérique des facteurs et exacte de l'état RNG. Deux topologies
SD, la propriété des snapshots, le budget, l'annulation et l'export sont testés.
Les fichiers distribués sont des fixtures statiques ; les tests .NET n'exécutent
aucun Python. Voir [les résultats et limites](qualification/loha-training.json).

Le diagnostic suivant lit un checkpoint à son emplacement existant :

```text
dotnet run -c Release --project tools/ComfySharp.RuntimeProbe -- sd-all-adapter-train --algorithm LoHa --checkpoint <checkpoint.safetensors> --report <nouveau-rapport.json> --device cpu
```

Sur le vrai SD1.5, il effectue deux mises à jour sur toutes les cibles : 1 532
gradients finis et 282 gradients alpha absents à chaque étape, avec base inchangée.
Les entrées sont synthétiques et petites (latent 8 × 8, contexte brut). Aucun
fichier de poids n'est écrit sans l'option explicite `--adapter`. Le rechargement
en mémoire est vérifié après les mises à jour. Ce diagnostic ne qualifie ni une famille de modèles,
ni la qualité sur images, ni les gradients préentraînés par comparaison à ComfyUI.

Restent requis : reprise LoHa et autres géométries/précisions d'inférence,
intégration au nœud public `TrainLoraNode`, précision mixte, offload, datasets
réels et preuves CPU/GPU sur les plateformes cibles. Cette étape ne ferme pas
le lot 9 ni le catalogue complet de la V1.
