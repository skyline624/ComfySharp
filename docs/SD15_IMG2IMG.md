# Réencodage d'image et débruitage partiel SD1.5

`VAEEncode` convertit un tenseur `IMAGE` en `LATENT` brut avec les poids du VAE
chargé. Le nœud conserve les identifiants et les entrées `pixels`/`vae` de
[la source figée](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/nodes.py).
Il est disponible dans le Host, le compilateur de documents et le canvas natif.
Le registre contient 43 types, dont la portée complète reste à qualifier.

`KSampler` accepte désormais `denoise` entre 0 et 1 pour Euler/Karras. Le port
reprend `KSampler.set_steps` et `Sampler.max_denoise` de
[samplers.py](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/samplers.py) :

- Au-dessus de 0,9999, construire le planning normal.
- Entre 0 et ce seuil, construire `int(steps / denoise)` étapes et conserver les
  `steps + 1` derniers sigmas, y compris le zéro terminal.
- Déduire la règle de bruit maximal du premier sigma et du maximum du modèle,
  avec la comparaison relative amont de 1e-5.
- À zéro, restituer les valeurs du latent brut sans bruit, scaling ni appel au
  réseau de débruitage. Les producteurs du graphe peuvent néanmoins être exécutés.

Les métadonnées inconnues du latent sont conservées ; les métadonnées de ratio
spatial/temporel sont retirées à la sortie comme dans le parcours existant.
Le budget actuel refuse explicitement les plannings étendus au-delà de 10 000
étapes. L'exécution reste limitée à 1–100 étapes, une image et Euler/Karras.
Les masques et conditionnements régionaux restent à porter.

Le [workflow d'exemple](workflows/sd15-reencode-refine.api.json) génère une image
de pommes rouges, la décode, la réencode avec `VAEEncode`, puis la transforme
avec le prompt « a photograph of a green apple on a wooden table » à 50 % de
débruitage, dix étapes et seed 1. Il utilise le checkpoint du dossier partagé.

Après construction du Host CUDA selon [SD15_WORKFLOW.md](SD15_WORKFLOW.md) :

```text
dotnet run --no-build -c Release --project src/ComfySharp.Desktop -- --inference-device cuda:0 --models-dir <dossier-modeles-partage> --workflow docs/workflows/sd15-reencode-refine.api.json
```

Le workflow part d'une image produite par le premier sampler. Le chargement d'un
fichier externe par `LoadImage` reste à porter. `VAEEncode` accepte actuellement
une image de 32 à 519 pixels par dimension, recadrée au centre vers un multiple
de huit de 32 à 512 ; les trois premiers canaux sont utilisés. Les réseaux et
les calculs restent sur le périphérique sélectionné, avec transferts explicites
aux frontières des nœuds média. Aucune copie de checkpoint sur disque n'est créée.

Les [références de planning](../tests/ComfySharp.Inference.Tests/Fixtures/sd15-denoise.reference.json)
proviennent de huit appels des fonctions Python figées dans un laboratoire
séparé PyTorch 2.10 CPU. Le [collecteur](../labs/sd15-denoise-source/reference.py)
lit ces déclarations avec `git show` au commit imposé et enregistre les hashes
des sources. Les tests .NET consomment uniquement le JSON. Leur précision
absolue et relative est fixée à 1e-6. Ces cas vérifient le planning, pas la
qualification numérique de l'image générée ou de toute la famille.

La [preuve du parcours natif Windows/CUDA](qualification/sd15-img2img.json) conserve
le hash des poids relus sur place, le prompt, les sources testées, le hash du PNG
et les résultats des tests. Le parcours a réussi en 20,4 secondes. L'image reste
principalement rouge malgré le nouveau prompt ; cet essai démontre l'exécution
du graphe, sans déclarer une fidélité sémantique ou une parité numérique complète.
