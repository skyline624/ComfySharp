# DPM++ 2M

`KSampler` accepte `sampler_name=dpmpp_2m`, avec les neuf schedulers SD déjà
portés. Le mode applique la fonction
[`sample_dpmpp_2m` du backend figé](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/k_diffusion/sampling.py).
Il conserve une prédiction précédente pour corriger les étapes intermédiaires.
La première et la dernière étape utilisent la branche sans cette correction.
Le dernier intervalle mène à sigma zéro ; les logarithmes, exponentielles
et `expm1` gardent l'ordre arithmétique de la source.

La prédiction précédente partage son stockage avec un alias possédé par le
sampler. Elle est libérée dès son remplacement, ainsi que l'ancien état
latent. Annulation ou exception libèrent l'historique et les intermédiaires.
Les calculs natifs en cours ne peuvent pas être interrompus ; les vérifications
d'annulation se font à leurs frontières. Chaque appel possède son historique,
sans état de trajectoire conservé entre deux générations.

Les plannings peuvent avoir des sigmas répétés. Le port ne les supprime pas :
`[1,1,0]` et `[2,1,1,0]` restent valides dans les références. En revanche,
`[1,1,0.5,0]` rend le rapport entre intervalles indéfini dans la source.
ComfySharp détecte une trajectoire non finie et renvoie une erreur structurée
par le moteur, au lieu de sauvegarder un résultat invalide ou de changer
silencieusement les sigmas. Cette vérification synchronise le résultat
natif à chaque étape.

```text
dotnet run --no-build -c Release --project src/ComfySharp.Desktop -- --inference-device cuda:0 --models-dir <dossier-modeles-partage> --workflow docs/workflows/sd15-dpmpp2m-karras.api.json
```

Le [workflow fourni](workflows/sd15-dpmpp2m-karras.api.json) utilise 20 étapes,
Karras, CFG 7 et la graine 0. Le sampler accepte également un latent produit
par `LoadImage` puis `VAEEncode`, avec débruitage partiel. `denoise=0` conserve
le latent brut selon le contrat commun.

Les [six cas source](../tests/ComfySharp.Inference.Tests/Fixtures/dpmpp-2m.reference.json)
proviennent de la véritable fonction Python figée dans un laboratoire séparé
PyTorch 2.10 CPU. Cinq sorties finies et tous leurs appels intermédiaires sont
comparés, avec bornes absolue et relative fixées à `1e-6` ; un cas non fini
doit être diagnostiqué. Le débruiteur analytique de ce corpus sert uniquement
à isoler la trajectoire, sans prétendre simuler un modèle de génération.
Les tests .NET consomment le JSON et n'utilisent pas Python.

La [preuve de qualification partielle](qualification/dpmpp-2m.json) enregistre
les tests, les sources et les parcours natifs Windows/CUDA : génération
SD1.5 et image-vers-image à `denoise=0.7`. Les vrais poids restent dans le
dossier partagé. Le test de modèle réduit vérifie aussi la guidance et la
durée de vie après libération des propriétaires du modèle.

Le périmètre actuel reste SD1.5 Float32, CPU ou CUDA, un élément, au plus
512 pixels par dimension et 1 à 100 étapes demandées. Les variantes SDE,
CFG++, les autres architectures et la comparaison numérique
préentraînée avec ComfyUI restent à porter ou qualifier. Ce mode ne suffit
pas à déclarer la famille SD1.5 ou la V1 complète.

Les [masques SD1.5](SD_INPAINT.md) sont désormais raccordés à ce sampler,
avec preuve d'exécution CUDA distincte de cette première campagne.
