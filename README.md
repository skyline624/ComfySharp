# ComfySharp

Port indépendant de [ComfyUI](https://github.com/comfy-org/ComfyUI) en C#/.NET 10, avec interface native Avalonia et calcul natif TorchSharp/libtorch. Licence GPLv3. Ce projet n'est pas une version officielle de Comfy-Org.

**État : développement initial 0.1.0-dev. La génération par modèles d'images, vidéo, audio et 3D n'est pas encore disponible.** Le périmètre final reste le catalogue local complet, y compris l'entraînement intégré. Voir [l'objectif](docs/OBJECTIF.md), [le plan approuvé](docs/MIGRATION.md), [l'avancement réel](docs/STATUS.md) et [la matrice](docs/capabilities/README.md).

Le socle contient un éditeur C#/XAML avec canvas de nœuds, un Host .NET séparé, des documents JSON conservés sans perte et un moteur avec valeurs natives. Son registre fournit 37 identifiants : 18 utilitaires, cinq générateurs de sigmas, six traitements SIGMAS, quatre primitives IMAGE, ImageBatch, PreviewAny, SaveImage et PreviewImage. Les [aperçus PNG natifs](docs/BITMAP_PREVIEWS.md) offrent une navigation dans les lots et conservent le workflow soumis dans les métadonnées. Les graphes CLIP-L/G, U-Net SD1/SD2 et VAE classique sont implémentés dans la bibliothèque d'inférence ; leur qualification et leur assemblage en workflows de génération restent en cours.

## Compiler et lancer

Installer le SDK .NET **10.0.300**, puis depuis ce dépôt :

```sh
dotnet restore ComfySharp.slnx --locked-mode
dotnet build ComfySharp.slnx --no-restore
dotnet test ComfySharp.slnx --no-build --no-restore
dotnet run --no-build --project src/ComfySharp.Desktop
```

Desktop recherche le Host compilé et le lance sur une adresse loopback. Pour un autre emplacement, définir `COMFYSHARP_HOST_PATH` vers l'apphost ou le `.dll` du Host. Un arrêt du Host préserve les documents ; le redémarrage ne resoumet aucun travail.

Serveur seul :

```sh
dotnet run --project src/ComfySharp.Host -- --urls http://127.0.0.1:8189
```

`GET /health`, `/object_info`, `/queue`, `/history` et `POST /prompt` permettent d'inspecter et d'exécuter le sous-ensemble courant. Relier le résultat à `PreviewAny` : le serveur sélectionne les vrais nœuds de sortie. `partial_execution_targets`, facultatif, filtre uniquement ces sorties. L'historique et les événements UI contiennent les textes de prévisualisation ; les tenseurs restent dans le Host. Les formats d'API encore incomplets sont indiqués dans [STATUS.md](docs/STATUS.md).

Exemple exécutable : ouvrir [text-length.workflow.json](examples/text-length.workflow.json) dans l'éditeur, ou soumettre [text-length.prompt.json](examples/text-length.prompt.json) au Host. Le texte « Bonjour 🌍 » contient neuf caractères Unicode.

[StringFormat](docs/STRING_FORMAT.md) permet aussi de composer des textes à partir de connexions nommées. Le profil actuel prend en charge les substitutions simples, `!s`, la largeur, la précision et les alignements de chaînes Unicode. Les autres fonctions du formatage Python restent à porter et produisent des diagnostics explicites.

[StringContains et StringCompare](docs/TEXT_COMPARISON.md) recherchent et comparent du texte Unicode valide. Leur mode insensible à la casse utilise les règles de minuscules de Python figées, y compris le sigma grec contextuel et l'expansion de `İ`.

[CaseConverter](docs/CASE_CONVERTER.md) propose les majuscules, les minuscules et les deux modes de capitalisation de ComfyUI, avec les tables Unicode de CPython figées. Il prend en charge les expansions de caractères et le contexte original des chaînes Unicode valides.

Les [primitives IMAGE](docs/IMAGE_PRIMITIVES.md) créent des images unies, inversent les couleurs en conservant l'alpha, répètent un lot et en extraient une tranche. Ce premier profil travaille sur CPU en Float32 ; l'éditeur peut exécuter la chaîne et afficher les valeurs du tenseur. [ImageBatch](docs/IMAGE_BATCH.md) ajoute la réunion de lots avec padding alpha, recadrage et redimensionnement bilinéaire. Les codecs et l'export de fichiers restent à porter.

[sigma-preview.workflow.json](examples/sigma-preview.workflow.json) et son [prompt](examples/sigma-preview.prompt.json) exécutent `KarrasScheduler → SplitSigmas → PreviewAny` avec de vrais tenseurs CPU. Les deux sorties sont `tensor([3., 2.])` et `tensor([2., 1., 0.])`. Aucun poids de modèle n'est nécessaire à ce calcul. La prévisualisation native est en texte brut ; Markdown et d'autres représentations restent à porter. Voir [les contrats et limites](docs/NATIVE_NODE_HOST.md).

Le SDK sert à construire le projet ; les distributions autonomes futures n'exigeront pas son installation. Aucun Python, pip, Node ou frontend JavaScript n'est utilisé par le produit. NuGet télécharge les bibliothèques de compilation ; le produit ne télécharge aucun modèle et n'appelle aucun service distant.

## Vérifier le runtime natif

```sh
dotnet run --project tools/ComfySharp.RuntimeProbe -- --device cpu --repeat 25
```

Windows/NVIDIA, avec la variante native CUDA 12.8 (plusieurs Go) :

```sh
dotnet run --project tools/ComfySharp.RuntimeProbe -p:NativeBackend=cuda -- --device cuda --repeat 25
```

La sélection native isole automatiquement les sorties de compilation et verrous par plateforme et backend. Elle désigne les bibliothèques distribuées ; le paramètre `--device` du probe désigne le calcul. Les générateurs SIGMAS actuels créent des tenseurs CPU, y compris dans un Host distribué avec CUDA. Un backend indisponible retourne une erreur. Les preuves locales concernent actuellement Windows CPU et une RTX 3090 ; Linux/NVIDIA et macOS/MPS attendent leur qualification matérielle. Les tests CI CPU ne remplacent pas celle-ci.

## Repères

L'inspecteur vérifie un fichier safetensors sans charger les tenseurs ni initialiser libtorch :

```sh
dotnet run --project tools/ComfySharp.ModelInspect -- chemin/vers/modele.safetensors
```

Ajouter `--sha256` pour lire le fichier entier et calculer son empreinte, ou `--tensors 10` pour afficher au plus dix noms/formes de tenseurs. Le résultat par défaut contient seulement des agrégats. Un en-tête valide n'établit pas la compatibilité du modèle. Le lecteur tensoriel prend en charge dix dtypes CPU ; les types non pris en charge produisent une erreur explicite.

Le diagnostic CLIP produit des IDs, poids et séquences SD1/SDXL avec ses ressources embarquées, sans modèle ni libtorch :

```sh
dotnet run --project tools/ComfySharp.Tokenize -- --text "a (cat:1.5)" --profile sdxl
```

Il accepte aussi `--file` pour un fichier UTF-8 ou `--stdin`, et `--disable-weights`. Les profils disponibles sont `sd1-l`, `sdxl-l`, `sdxl-g` et `sdxl`. Les [règles de tokenisation et leurs références](docs/CLIP_TOKENIZATION.md) sont figées ; ce diagnostic ne calcule aucun embedding d'encodeur.

Le [diagnostic expérimental des encodeurs](docs/CLIP_ENCODERS.md) charge un checkpoint safetensors explicitement choisi et calcule le conditioning CLIP en CPU/F32. Son inspection des métadonnées fonctionne sans initialiser libtorch. La qualification numérique aux dimensions complètes est en cours ; aucune compatibilité SD1/SDXL n'est encore annoncée.

- [Architecture](docs/ARCHITECTURE.md), [backlog](docs/BACKLOG.md), [validation numérique](docs/NUMERICAL_VALIDATION.md).
- [Contrat des valeurs natives](docs/RUNTIME_VALUES.md) et [réconciliation des nœuds](docs/audit/05-RECONCILIATION-NOEUDS.md).
- [Générateurs de niveaux de bruit et corpus de référence](docs/SIGMA_SCHEDULES.md).
- [U-Net, VAE et frontières de diffusion SD](docs/SD_COMPONENTS.md), [références source indépendantes](docs/qualification/sd-source-86812a9.md).
- [Audits source](docs/audit/02-MOTEUR-WORKFLOWS.md) avec liens vers les révisions amont figées.
- [Notices](THIRD_PARTY_NOTICES.md), versions centralisées et verrous par projet/runtime.
- `tools/publish.ps1` prépare un dossier portable de développement non signé ; une V1 exige la fermeture de toute la matrice.

Modèles, entrées, sorties et données personnelles sont exclus de Git. La base ComfySharp utilise son propre répertoire de données ; elle ne migre pas en place la base ComfyUI.
