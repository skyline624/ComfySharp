# Génération SD1.5 avec un checkpoint local

La commande `ComfySharp.RuntimeProbe sd15-generate` raccorde le chargeur SD1.5
aux trois réseaux réels : CLIP-L, U-Net et VAE classique. Elle accepte un
checkpoint monolithique safetensors au layout SD1.5, sans télécharger ni copier
les poids. Le diagnostic est distinct des nœuds du Host et de l'éditeur.

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

Le calcul utilise CPU/Float32, le bruit gaussien natif avec générateur par appel,
le conditionnement CLIP ComfyUI, le débruiteur EPS, CFG séparé, Euler sans churn et
Karras rho 7. Le latent initial utilise la règle de débruitage maximal ; le latent
final est divisé par 0,18215 avant décodage. Le fichier PNG contient les paramètres
dans `comfysharp.sd15`, sans prétendre être un document graphique ComfyUI.

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

Une sortie `status=ok` prouve l'exécution demandée, pas la parité numérique avec
ComfyUI. `familyQualified=false` reste explicite : comparaison avec références,
autres workflows, intégration Host/Desktop, CUDA et qualification des plateformes
restent nécessaires. Le diagnostic réduit antérieur et ses tolérances restent
inchangés dans [SD15_PIPELINE.md](SD15_PIPELINE.md).
