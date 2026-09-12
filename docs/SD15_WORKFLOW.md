# Parcours SD1.5 dans Desktop et Host

Les six nœuds `CheckpointLoaderSimple`, `CLIPTextEncode`, `EmptyLatentImage`,
`KSampler`, `VAEEncode` et `VAEDecode` raccordent les composants SD1.5 au moteur. `SaveImage`
enregistre le résultat et l'éditeur récupère le PNG par l'API locale pour afficher
son aperçu. Le catalogue du Host contient désormais 43 types enregistrés.
L'enregistrement d'un type ne signifie pas que toutes ses variantes sont portées.

Après compilation Release, lancer l'éditeur depuis la racine du dépôt :

```text
dotnet run --no-build -c Release --project src/ComfySharp.Desktop -- --models-dir <dossier-modeles-partage> --workflow docs/workflows/sd15-euler-karras.api.json
```

Le dossier indiqué contient un sous-dossier `checkpoints`. L'exemple attend
`checkpoints/v1-5-pruned-emaonly.safetensors`. Modifier `ckpt_name` dans le document
pour un autre checkpoint stock SD1.5 compatible. Les noms des sous-dossiers
emploient `/`. Aucun téléchargement, copie de poids ou cache disque de modèles
n'est créé. Le Host peut aussi recevoir `--models-dir` directement, ou hériter
de `COMFYSHARP_MODELS_DIR`. Les liens symboliques et jonctions dans le catalogue
des checkpoints sont refusés ; les fichiers sont lus à leur emplacement partagé.

Le prompt API est importé comme document éditable. Cliquer sur **Queue** lance
les calculs et affiche l'image dans `SaveImage`. La sauvegarde du document crée
un workflow JSON distinct du prompt API d'origine. `--data-dir <dossier>` permet
de choisir les réglages et sorties de ComfySharp séparément des modèles.

Le premier parcours est limité à un checkpoint monolithique stock SD1.5,
prédiction EPS, CPU ou CUDA/Float32, Euler sans churn, Karras, `denoise` de 0 à 1, un batch de
une image de 32 à 512 pixels par dimension et 1 à 100 étapes. Le Host utilise
au plus 16 threads CPU par défaut ; `--cpu-threads` le configure directement.
Les masques, batch-index noise, conditionnements régionaux ou programmés, autres
samplers/schedulers et autres architectures produisent des diagnostics explicites.
Les identifiants de leurs choix amont restent dans les schémas pour préserver les
documents. La qualification numérique CUDA, les autres plateformes et la qualification complète restent ouvertes.
Le [réencodage et débruitage partiel](SD15_IMG2IMG.md) décrit les règles des sigmas
et le budget des plannings étendus ; la commande CLI `sd15-generate` conserve son
parcours de génération à débruitage complet.

Pour utiliser le GPU NVIDIA sous Windows, construire le Host avec son bundle CUDA
verrouillé, puis sélectionner ce périphérique au lancement de l'éditeur :

```text
dotnet restore src/ComfySharp.Host -p:NativeBackend=cuda --locked-mode
dotnet build src/ComfySharp.Host -c Release -p:NativeBackend=cuda --no-restore
dotnet run --no-build -c Release --project src/ComfySharp.Desktop -- --inference-device cuda:0 --models-dir <dossier-modeles-partage> --workflow docs/workflows/sd15-euler-karras.api.json
```

Le Host place CLIP, U-Net et VAE sur le GPU choisi. Le bruit initial et les sigmas
sont construits sur CPU puis transférés ; l'image décodée revient sur CPU pour
les nœuds média. Les checkpoints restent lus à leur emplacement partagé, avec
chargement temporaire en RAM puis transfert des poids en VRAM. Aucun cache disque
de modèle n'est créé. Le budget des poids ne couvre pas les activations GPU.

Le mode Float32 désactive explicitement TF32 pour les multiplications matricielles
et convolutions CUDA avant le chargement. Ces réglages libtorch sont globaux au
processus ; les extensions ne doivent pas les modifier pendant une exécution.
Le CPU reste le défaut. Un bundle CUDA absent ou un GPU indisponible produit
une erreur explicite, sans repli silencieux. Seul `cuda:0` est exposé actuellement.
La [preuve Windows/RTX 3090](qualification/sd15-cuda.json) distingue exécution
réelle, répétabilité et comparaison numérique restant à qualifier.

Le chargeur inspecte les trois composants avant allocation, avec un budget de
poids de 16 Gio. Les tenseurs auxiliaires extérieurs aux réseaux sont signalés
dans `comfysharp_model`; les paramètres de réseau inconnus restent refusés.
Les modèles chargés et leurs graphes possèdent des durées de vie indépendantes.
Les latents aux frontières des nœuds sont dans l'espace VAE brut : `KSampler`
applique la conversion d'entrée puis de sortie ; `VAEDecode` ne divise pas une
seconde fois par 0,18215. Le conditionnement garde la structure `[tensor, metadata]`.

Les schémas dérivent de [nodes.py figé](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/nodes.py)
et des [samplers figés](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/samplers.py).
Leur portée complète reste celle du plan de migration ; ces limites intermédiaires
ne retirent aucune capacité de la V1.

Un diagnostic opt-in ouvre la vraie fenêtre Avalonia, importe et compile le
workflow, lance son Host, attend le travail puis vérifie le bitmap d'aperçu 512×512 :

```text
dotnet run --no-build -c Release --project src/ComfySharp.Desktop -- --models-dir <dossier-modeles-partage> --workflow docs/workflows/sd15-euler-karras.api.json --data-dir <dossier-diagnostic> --sd15-smoke-report <nouveau-rapport.json>
```

Le diagnostic ferme sa propre fenêtre et renvoie 0 uniquement si ce parcours
réussit. Le rapport contient le document, le prompt compilé, l'historique,
les dimensions de l'aperçu et le hash du PNG. Il conserve les sorties dans le
dossier de diagnostic et refuse d'écraser un rapport existant. Ce scénario demande
des poids locaux et ne fait pas partie des tests CI sans modèle. Il n'annonce
jamais une qualification complète de la famille.
