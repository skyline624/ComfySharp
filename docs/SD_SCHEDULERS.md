# Les neuf schedulers SD

`KSampler` accepte les neuf identifiants du backend figé avec les samplers
Euler et Heun sans churn. Les paramètres restent ceux des workflows ComfyUI :
`scheduler`, `steps`, `denoise`, `seed` et `cfg`.

| Scheduler | Calcul porté |
| --- | --- |
| `simple` | Indices descendants de la table SD, espacement réel puis troncature |
| `sgm_uniform` | Temps uniformes, omission du dernier temps avant ajout de zéro |
| `karras` | Interpolation puissance de sigma, rho par défaut égal à 7 |
| `exponential` | Interpolation uniforme dans le logarithme de sigma |
| `ddim_uniform` | Parcours des indices depuis 1, pas entier, puis inversion |
| `beta` | Quantiles Beta(0.6, 0.6), arrondi au pair, suppression des indices répétés |
| `normal` | Temps uniformes entre les extrémités de la table discrète |
| `linear_quadratic` | Partie linéaire puis quadratique, seuil par défaut 0.025 |
| `kl_optimal` | Interpolation des arctangentes de sigma |

Les fonctions suivent [`comfy/samplers.py`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/samplers.py)
et les générateurs Karras/exponentiel de
[`sampling.py`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/k_diffusion/sampling.py).
Cette version utilise la table SD discrète par défaut à 1 000 temps ; elle
ne prétend pas couvrir les lois de bruit des autres architectures.

Le nombre d'intervalles peut différer de `steps`. Pour sept étapes demandées,
DDIM produit huit intervalles ; pour 1 001 étapes demandées, il en produit
999 et beta 848. Ces longueurs proviennent de la source, pas d'une correction
de ComfySharp. À débruitage complet, tout le planning est conservé. Pour
`0 < denoise <= 0.9999`, le planning est calculé avec `int(steps/denoise)`,
puis ses derniers `steps+1` éléments sont gardés, ou tous s'il est plus court.
`denoise=0` conserve le latent brut selon le contrat existant.

Le quantile bêta est calculé en C# par une série binomiale intégrée sur
`[0,0.5]`, normalisée par symétrie, puis une dichotomie bornée. La médiane
exacte reste `0.5`, ce qui préserve l'arrondi au pair de l'indice `499.5`.
Ni SciPy ni une bibliothèque Python ne sont utilisés par l'application.

Les [71 cas source](../tests/ComfySharp.Inference.Tests/Fixtures/sd-schedulers.reference.json)
comparent les valeurs et les longueurs des plannings, les découpages partiels,
les plateaux et les demandes allant jusqu'à 10 000 étapes. Les bornes
absolue et relative `1e-6` ont été fixées avant comparaison. La source
`kl_optimal` à une étape produit une valeur non finie : ComfySharp lève
une erreur explicite, sans inventer de planning de remplacement.

Le [collecteur indépendant](../labs/sd-schedulers-source/reference.py) exécute
les fonctions figées avec PyTorch 2.10 CPU et SciPy pour le quantile de
référence. Les tests .NET lisent uniquement le JSON. Les anciens corpus et
tolérances restent inchangés ; le test de durée de vie vérifie aussi la
libération des tenseurs lors de requêtes invalides.

```text
dotnet run --no-build -c Release --project src/ComfySharp.Desktop -- --inference-device cuda:0 --models-dir <dossier-modeles-partage> --workflow docs/workflows/sd15-heun-beta.api.json
```

Le [workflow Heun/beta](workflows/sd15-heun-beta.api.json) fournit un exemple.
La [preuve de campagne](qualification/sd-schedulers.json) distingue les
comparaisons de plannings des générations réelles. Les limites du Host restent
SD1.5 Float32, un élément, au plus 512 pixels par dimension, 1 à 100 étapes
demandées et 10 000 étapes pour le planning étendu. Les masques, les autres
samplers, les options avancées des schedulers et les architectures restantes
font toujours partie du travail à réaliser. La comparaison numérique des
générations complètes et les validations Linux/CUDA et macOS/MPS restent ouvertes.
