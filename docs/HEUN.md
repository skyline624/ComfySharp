# Heun sans churn

`KSampler` accepte désormais `sampler_name=heun` avec les [neuf schedulers SD](SD_SCHEDULERS.md),
en plus du parcours Euler existant. Les mêmes paramètres `seed`, `steps`,
`cfg` et `denoise` sont utilisés. Un `denoise` nul conserve le latent brut.

Le port suit la fonction [`sample_heun` figée](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/k_diffusion/sampling.py)
avec son défaut `s_churn=0`. Chaque intervalle non terminal évalue le débruiteur
sur l'état courant, prédit un état au sigma suivant, puis réévalue le
débruiteur et moyenne les deux dérivées. L'intervalle final utilise Euler
pour éviter de diviser par le sigma terminal nul. Vingt étapes correspondent
à 39 évaluations du débruiteur guidé ; CFG peut lui-même appeler deux fois
le réseau par évaluation.

Les opérations conservent l'ordre tensoriel Float32 de la source. Le sampler
retient son débruiteur indépendamment de ses parents et libère les états
intermédiaires à chaque étape. L'annulation est vérifiée avant et après les
évaluations natives ; elle ne peut pas interrompre une opération CUDA en
cours. L'échec de la seconde évaluation libère aussi l'état prédit.

```text
dotnet run --no-build -c Release --project src/ComfySharp.Desktop -- --inference-device cuda:0 --models-dir <dossier-modeles-partage> --workflow docs/workflows/sd15-heun-karras.api.json
```

Le [workflow fourni](workflows/sd15-heun-karras.api.json) charge le checkpoint
SD1.5 partagé, encode le texte, calcule 20 étapes Heun/Karras et affiche le
PNG sauvegardé. La [preuve Windows/CUDA](qualification/heun.json) rapporte
un parcours natif 512 × 512 en 18,3 secondes sur RTX 3090, Float32, TF32
désactivé, avec CLIP, U-Net et VAE sur `cuda:0`. Aucun poids n'est copié
ou téléchargé par l'application.

Les [quatre cas de référence](../tests/ComfySharp.Inference.Tests/Fixtures/heun.reference.json)
proviennent de la vraie fonction source exécutée dans le laboratoire
PyTorch 2.10 CPU. Un débruiteur analytique explicite isole la trajectoire ODE :
les tests comparent la sortie et chaque entrée des évaluations intermédiaires
avec des bornes absolue et relative fixées à `1e-6`. Ils couvrent un intervalle,
plusieurs intervalles, des sigmas égaux et de petits sigmas. Il ne s'agit pas
d'une référence de réseau neuronal. Les tests .NET ne lancent pas Python.
Le [collecteur](../labs/heun-source/reference.py) et ses sources sont identifiés.

La campagne locale passe 46 tests Inference ciblés, dont cinq nouveaux tests
Heun et les références Euler existantes. Le Host passe 310 tests et conserve
un échec de privilège Windows de création de liens symboliques. Les builds
Release CPU et CUDA passent sans avertissement. La CI exécute aussi les
contrats Heun et de débruitage partiel avant les suites numériques déjà en
échec sur Linux ; elle conserve ces échecs et leurs seuils.

Ce jalon ne qualifie pas la famille SD1.5 complète : la comparaison Heun
préentraînée avec ComfyUI, les variantes avec churn, les autres samplers,
les masques, les batches de génération supérieurs à un, les dimensions
supérieures à 512 et la qualification Linux/CUDA et macOS/MPS restent à faire.
