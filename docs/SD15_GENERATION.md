# Génération SD1.5 avec un checkpoint local

La commande `ComfySharp.RuntimeProbe sd15-generate` raccorde le chargeur SD1.5
aux trois réseaux réels : CLIP-L, U-Net et VAE classique. Elle accepte un
checkpoint monolithique safetensors au layout SD1.5, sans télécharger ni copier
les poids. Le diagnostic reste utilisable séparément du [parcours Host et éditeur](SD15_WORKFLOW.md).

Construire selon le README, puis inspecter le fichier partagé :

```text
dotnet run --no-build -c Release --project tools/ComfySharp.RuntimeProbe -- sd15-generate --checkpoint <checkpoint.safetensors>
```

L'inspection vérifie les schémas des trois composants et annonce les octets de
poids résidents et temporaires. Elle ne charge pas les tenseurs natifs. Les clés
inconnues sont refusées par défaut. Pour un checkpoint contenant aussi des tables
de diffusion ou un état d'entraînement, `--report-outside-components` autorise les
tenseurs hors des trois espaces de noms et les liste dans le rapport. Les poids
inconnus à l'intérieur des réseaux restent refusés ; aucun réseau n'est substitué.

Pour calculer une image, fournir un nouveau fichier PNG dans un dossier existant :

```text
dotnet run --no-build -c Release --project tools/ComfySharp.RuntimeProbe -- sd15-generate --checkpoint <checkpoint.safetensors> --report-outside-components --execute --output <nouveau.png> --prompt "a photograph of a red apple on a wooden table" --negative "" --width 512 --height 512 --steps 20 --cfg 7 --seed 0 --threads 8 --weight-budget-mib 16384
```

Le calcul utilise CPU/Float32 par défaut, le bruit gaussien natif avec générateur par appel,
le conditionnement CLIP ComfyUI, le débruiteur EPS, CFG séparé, Euler sans churn et
Karras rho 7. Le latent initial utilise la règle de débruitage maximal ; le latent
final est divisé par 0,18215 avant décodage. Le fichier PNG contient les paramètres
dans `comfysharp.sd15`, sans prétendre être un document graphique ComfyUI.

Sous Windows/NVIDIA, `--device cuda:0` sélectionne le calcul des trois réseaux
sur GPU avec TF32 désactivé. Construire d'abord la variante CUDA du diagnostic :

```text
dotnet restore tools/ComfySharp.RuntimeProbe -p:NativeBackend=cuda --locked-mode
dotnet build tools/ComfySharp.RuntimeProbe -c Release -p:NativeBackend=cuda --no-restore
dotnet tools/ComfySharp.RuntimeProbe/bin/native/win-x64/cuda/Release/net10.0/ComfySharp.RuntimeProbe.dll sd15-generate --device cuda:0 --checkpoint <checkpoint.safetensors> --report-outside-components --execute --output <nouveau.png>
```

Le diagnostic refuse un GPU indisponible. Il conserve le bruit initial et les
sigmas CPU avant transfert pour pouvoir les comparer exactement ; les captures
CUDA sont explicitement transférées sur CPU pour leur écriture. Le rapport
enregistre le périphérique et la politique TF32. Les valeurs CPU et GPU ne sont
pas supposées identiques : voir les [mesures CUDA](qualification/sd15-cuda.json).

Les limites de cette commande sont une image, des dimensions multiples de huit
entre 32 et 512, 1 à 100 étapes et 4 096 caractères par texte. Le budget explicite
contrôle les poids uniquement ; il ne garantit pas le pic mémoire des activations.
Ctrl+C demande une annulation entre opérations natives. Un PNG existant est refusé,
y compris s'il apparaît après la validation des arguments.

Les étapes sont écrites sur stderr et le résultat JSON sur stdout. Une génération
réussie rapporte le SHA-256 du checkpoint lu, celui du PNG, les dimensions,
paramètres, durée et pic mémoire du processus. Le checkpoint est ouvert une fois
pour l'inspection, le hash et le chargement. Aucun cache de poids sur disque n'est
créé ; les graphes et tenseurs possèdent des ressources explicitement libérées.

L'option `--trace-dir <nouveau-dossier>` capture huit tenseurs Float32 : les deux
conditionnements, le bruit, les sigmas, le latent initial, le latent final, le
latent VAE et l'image. Le rapport JSON contient leurs formes, tailles et hashes.
Ces captures servent au [laboratoire de comparaison](../labs/sd15-pretrained-source/README.md),
avec les poids partagés et les déclarations ComfyUI figées. Un dossier de traces
existant est refusé ; les fichiers intermédiaires d'un calcul échoué ne constituent
pas une preuve de réussite.

Une sortie `status=ok` prouve l'exécution demandée, pas la parité numérique avec
ComfyUI. `familyQualified=false` reste explicite : comparaison avec références,
autres workflows, CUDA et qualification des plateformes
restent nécessaires. Le diagnostic réduit antérieur et ses tolérances restent
inchangés dans [SD15_PIPELINE.md](SD15_PIPELINE.md).

La [première comparaison préentraînée](qualification/sd15-pretrained-comparison.json) retrouve exactement les huit captures des fonctions ComfyUI figées, sur le cas documenté 512 × 512 / 20 étapes / seed 0 / CFG 7, en CPU Windows. Les autres cas et plateformes restent à qualifier.
