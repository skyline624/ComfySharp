# Bypass LoKr : opérateurs et gradients

Le diagnostic de rechargement initial décrit ci-dessous a été expliqué et corrigé
par [la distinction des modes natifs `mm`/`bmm`](LOKR_LINEAR_DISPATCH.md).
Le rechargement SD1.5 est maintenant exact en mode inférence ; l'écart avec
l'entraînement reste publié. Le catalogue et les plateformes ne sont pas qualifiés.

Les chemins `LoKrAdapter.h` et `LokrDiff.h` sont portés en opérations natives
groupées pour les couches linéaires et les convolutions 1D, 2D et 3D. Ils divisent
les canaux d'entrée en groupes, appliquent le second côté, puis mélangent les
groupes avec le premier côté. Ils ne construisent pas la matrice complète de
Kronecker. Le U-Net SD les utilise pour l'inférence et l'entraînement ; les
différences de normalisation et de biais suivent toujours le chemin ordinaire.

La [source figée](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/weight_adapter/lokr.py)
conserve des différences entre les deux chemins. À l'entraînement, chaque côté
décomposé applique son alpha/rang avant les opérateurs ; les paramètres gardent
leurs gradients. En inférence, le rang du premier côté décomposé est prioritaire,
alpha ne s'applique qu'une fois et DoRA est ignoré. Deux côtés directs avec alpha
zéro font échouer la division du bypass d'inférence, conformément à la source.

Le bypass d'inférence accepte un second facteur B portant les dimensions spatiales
pour les convolutions décomposées. La reconstruction ordinaire par `mm` échoue sur
ce format, comme dans la source. Les côtés matriciels et Tucker suivent leurs
opérateurs propres, sans transposition ajoutée pour rendre un cas invalide valide.
Les contrôles de forme de sortie signalent explicitement les géométries incompatibles.

## Preuves et limites

Le laboratoire séparé `labs/lora-training-source/lokr_bypass.py` exécute les deux
définitions AST figées. Ses 125 configurations donnent 85 calculs d'inférence
réussis et 40 erreurs source, ainsi que 85 calculs d'entraînement réussis et
40 erreurs de construction ou d'exécution source. Les erreurs des deux chemins
ne concernent pas forcément les mêmes configurations. Les facteurs F32/F16/BF16
sont convertis en Float32 ; les activations et paramètres entraînables de ce
profil sont Float32.

Les 250 cas .NET correspondants vérifient sorties, gradients d'entrée, gradients
des paramètres, une mise à jour SGD et les erreurs explicites. Deux autres tests
couvrent un facteur spatial chargé pour bypass et un parcours U-Net avec cibles
linéaire/convolution, gradients, rechargement et base inchangée. Ce dernier fixe
le budget des poids reconstruits à zéro pour contrôler l'absence de reconstruction.
Les tolérances des références restent `atol=rtol=3e-5`. Les snapshots retenus et
transférés survivent à leurs sources ; les compteurs de tenseurs reviennent à leur
valeur initiale après succès ou erreur. Les tests distribués n'exécutent pas Python.

Le diagnostic initial sur le vrai SD1.5 CPU atteint deux pas d'entraînement, mais son
contrôle d'égalité exacte après rechargement a échoué. Ce parcours ne constitue
donc pas une qualification réussie. Les hashes, mesures et contrôles de mode
autograd sont conservés dans [la preuve](qualification/lokr-bypass.json) ;
l'assertion exacte et les profils numériques existants ne sont pas relâchés.
La simple proximité numérique ne clôt pas cette investigation.

Le contrôle sans autograd, sur les mêmes propriétaires et paramètres, garde
l'écart maximal `2,413988e-6`. Le contrôle qui désactive temporairement
`requires_grad` sur les mêmes feuilles retrouve exactement l'inférence, puis
restaure tous les indicateurs. Il sert uniquement au diagnostic après échec.
Deux contrôles Python séparés sur cinq géométries chacun ne reproduisent pas
cet écart. Les contrôles linéaires ajoutés ensuite l'isolent et le reproduisent
dans la source ; voir le suivi en tête de page. Les anciennes observations restent
conservées et ne prouvent pas une parité numérique complète avec la source.

```text
dotnet run -c Release --project tools/ComfySharp.RuntimeProbe -- sd-all-adapter-train --algorithm LoKr --rank 7 --checkpoint <checkpoint-existant.safetensors> --report <nouveau.json> --device cpu --bypass true
```

Le chargeur de fichiers conserve sa validation stricte du nombre d'éléments du
poids cible. Certaines formes matricielles que `h` peut exécuter directement
restent donc refusées à cette frontière lorsque leur reconstruction n'a pas la
taille du poids cible ; une sélection et validation spécifiques au mode bypass
restent à qualifier. Les activations en précision mixte, modules hors SD,
workflows d'entraînement sur images, nœud public complet et plateformes GPU
restent également ouverts. Aucun modèle n'a été téléchargé, copié ou sauvegardé
pour ces vérifications ; le checkpoint partagé est relu directement.
