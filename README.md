# ComfySharp

Port indépendant de [ComfyUI](https://github.com/comfy-org/ComfyUI) en C#/.NET 10, avec interface native Avalonia et calcul natif TorchSharp/libtorch. Licence GPLv3. Ce projet n'est pas une version officielle de Comfy-Org.

**État : développement initial 0.1.0-dev. La génération d'images, vidéo, audio et 3D n'est pas encore disponible.** Le périmètre final reste le catalogue local complet, y compris l'entraînement intégré. Voir [l'objectif](docs/OBJECTIF.md), [le plan approuvé](docs/MIGRATION.md), [l'avancement réel](docs/STATUS.md) et [la matrice](docs/capabilities/README.md).

Le socle contient un éditeur C#/XAML avec canvas de nœuds, un Host .NET séparé, des documents JSON conservés sans perte, un moteur de graphes avec valeurs natives et 13 nœuds utilitaires réellement exécutables. Les lecteurs de poids et probes CPU/CUDA sont des fondations techniques, pas une implémentation des modèles.

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

`GET /health`, `/object_info`, `/queue`, `/history` et `POST /prompt` permettent d'inspecter et d'exécuter le sous-ensemble courant. Les nœuds utilitaires ont besoin de `partial_execution_targets` explicites : aucun faux nœud de sortie n'a été ajouté. Les formats d'API encore incomplets sont indiqués dans [STATUS.md](docs/STATUS.md).

Exemple exécutable : ouvrir [text-length.workflow.json](examples/text-length.workflow.json) dans l'éditeur, ou soumettre [text-length.prompt.json](examples/text-length.prompt.json) au Host. Le texte « Bonjour 🌍 » contient neuf caractères Unicode.

Le SDK sert à construire le projet ; les distributions autonomes futures n'exigeront pas son installation. Aucun Python, pip, Node ou frontend JavaScript n'est utilisé par le produit. NuGet télécharge les bibliothèques de compilation ; le produit ne télécharge aucun modèle et n'appelle aucun service distant.

## Vérifier le runtime natif

```sh
dotnet run --project tools/ComfySharp.RuntimeProbe -- --device cpu --repeat 25
```

Windows/NVIDIA, avec la variante native CUDA 12.8 (plusieurs Go) et un dossier d'artefacts distinct :

```sh
dotnet run --project tools/ComfySharp.RuntimeProbe -p:NativeBackend=cuda --artifacts-path artifacts/cuda -- --device cuda --repeat 25
```

Un backend indisponible retourne une erreur, jamais une réussite simulée. Les preuves locales concernent actuellement Windows CPU et une RTX 3090 ; Linux/NVIDIA et macOS/MPS attendent leur qualification matérielle. Les tests CI CPU ne remplacent pas celle-ci.

## Repères

L'inspecteur vérifie un fichier safetensors sans charger les tenseurs ni initialiser libtorch :

```sh
dotnet run --project tools/ComfySharp.ModelInspect -- chemin/vers/modele.safetensors
```

Ajouter `--sha256` pour lire le fichier entier et calculer son empreinte, ou `--tensors 10` pour afficher au plus dix noms/formes de tenseurs. Le résultat par défaut contient seulement des agrégats. Un en-tête valide n'établit pas la compatibilité du modèle. Le lecteur tensoriel prend en charge dix dtypes CPU ; les types non pris en charge produisent une erreur explicite.

- [Architecture](docs/ARCHITECTURE.md), [backlog](docs/BACKLOG.md), [validation numérique](docs/NUMERICAL_VALIDATION.md).
- [Contrat des valeurs natives](docs/RUNTIME_VALUES.md) et [réconciliation des nœuds](docs/audit/05-RECONCILIATION-NOEUDS.md).
- [Générateurs de niveaux de bruit et corpus de référence](docs/SIGMA_SCHEDULES.md).
- [Audits source](docs/audit/02-MOTEUR-WORKFLOWS.md) avec liens vers les révisions amont figées.
- [Notices](THIRD_PARTY_NOTICES.md), versions centralisées et verrous par projet/runtime.
- `tools/publish.ps1` prépare un dossier portable de développement non signé ; une V1 exige la fermeture de toute la matrice.

Modèles, entrées, sorties et données personnelles sont exclus de Git. La base ComfySharp utilise son propre répertoire de données ; elle ne migre pas en place la base ComfyUI.
