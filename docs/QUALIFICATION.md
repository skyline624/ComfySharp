# Qualification initiale

Campagne du 11 septembre 2026, sources de la première tranche 0.1.0-dev. Ces résultats qualifient seulement les fonctions décrites, pas une architecture de modèle ni la V1.

| Environnement | Résultat observé |
|---|---|
| Windows x64, SDK .NET 10.0.300 | Compilation Host, Core, Workflow, Inference, Desktop et tests |
| TorchSharp 0.107.0.0 / libtorch CPU 2.10.0 | Opérations natives, lecteur F32, RNG par générateur, Euler élémentaire, gradients et mise à jour SGD |
| NVIDIA GeForce RTX 3090, 24 576 MiB, pilote 610.88, libtorch CUDA 12.8 / 2.10.0 | 25 répétitions GPU réelles, résultats ci-dessous |
| Linux x64 CPU | Tests et probe natif exécutés avec succès sur le runner Ubuntu 24.04 |
| macOS 14 ARM64 CPU | Tests et probe CPU réussis après préchargement de la bibliothèque OpenMP fournie |
| Linux NVIDIA et macOS MPS | Non qualifiés |

Commande de probe : `dotnet run --project tools/ComfySharp.RuntimeProbe -p:NativeBackend=cuda --artifacts-path artifacts/cuda -- --device cuda --repeat 25`. Pour CPU : omettre la propriété CUDA et employer `--device cpu`. Le test vérifie l'emplacement GPU réel des tenseurs et synchronise CUDA avant observation mémoire.

| Assertion | CPU | RTX 3090 CUDA |
|---|---:|---:|
| Gradient de x² en x=2 | 4 | 4 |
| Paramètre après SGD, lr=0,1 | 1,6 | 1,6 |
| Somme de convolution de référence | 16 | 16 |
| Somme d'attention SDPA de référence | 4 | 4 |
| Somme de multiplication matricielle | 54 | 54 |
| Vue encore valide après libération du wrapper parent | Réussite | Réussite |
| Bruit CPU natif, même seed, générateurs indépendants | Identique | Identique |
| Transfert aller/retour du bruit CPU | Identique | Identique après transfert GPU |

Mémoire privée observée sur 25 itérations avec dispose scopes : CPU 129 875 968 → 130 621 440 octets ; CUDA 1 621 782 528 → 1 624 412 160 octets. Aucun GC forcé. **Ce relevé court ne prouve ni une absence de fuite, ni un plateau VRAM.** Les campagnes longues, changements de modèles et erreurs natives restent nécessaires.

La première tranche du lecteur safetensors teste offsets, doublons, chevauchements, formes/overflow, limites d'allocation, métadonnées, données tronquées et annulation après allocation avec libération. Cette première campagne ne couvrait que le chargement F32 ; l'extension aux autres types est décrite plus bas. Aucune performance de génération, qualité d'image ou fidélité de modèle n'a été mesurée ; aucun poids n'a été téléchargé pour cette campagne.

Tests de contrat supplémentaires : validation avant file, annulation d'un ancien job sans toucher son successeur, WebSocket initial, SQLite/settings/prune, conservation du document, édition/undo et démarrage de Desktop avec son Host. Les totaux finaux de tests et résultats CI sont publiés avec la révision correspondante.

Les verrous `packages.<runtime>.cpu.lock.json` existent pour les deux projets natifs sur les trois runtimes ; le probe a aussi `packages.win-x64.cuda.lock.json`. Les sept restaurations `--locked-mode` ont réussi. Une restauration croisée n'est pas une qualification de la plateforme cible.

Après revue et corrections, les suites locales comptent **164 tests réussis, aucun ignoré** : Core 47, Host/Catalog 71, Inference 23, Workflow 17, Desktop Headless 6. Elles couvrent notamment les régressions d'annulation HTTP chunked, de conservation des valeurs lazy, de clés JSON dupliquées, d'annulation après allocation native, de sauvegarde asynchrone et d'IDs numériques équivalents.

Une construction Windows autonome de Desktop et Host a aussi réussi son `--smoke-test` avec `PATH` limité à `Windows/System32`, sans override du chemin Host : lancement du Host voisin, health/catalogue, exécution réelle d'un nœud texte, lecture du résultat, fermeture. Cela prouve ce parcours local du socle ; une installation sur machine propre et la qualification V1 restent à faire. La publication des binaires attend la clôture des notices et de l'inventaire de leurs composants.

Le premier commit public [`b0a299c`](https://github.com/skyline624/ComfySharp/commit/b0a299c460b01de5bc486ec06d4e5a550dec338b) a été cloné dans un dossier indépendant, sans lien vers le dépôt parent. Ce clone a restauré les dépendances verrouillées, compilé la solution sans avertissement, passé les 164 tests et exécuté le parcours Desktop → Host → résultat texte. Son état Git est resté propre.

La [campagne CI `c790c79`](https://github.com/skyline624/ComfySharp/actions/runs/34601071274) a passé les 164 tests, le contrôle du catalogue et 25 répétitions du probe CPU sur Windows Server 2025 et Ubuntu 24.04. Windows Server en CI complète la vérification locale Windows 11 ; il ne remplace pas les tests d'installation sur la plateforme cible. macOS 14 a restauré et compilé, puis échoué sur six tests natifs : `libtorch_cpu.dylib` recherchait OpenMP au chemin absolu `/opt/homebrew/opt/libomp/lib/libomp.dylib`.

Dans la [campagne de diagnostic `295ad14`](https://github.com/skyline624/ComfySharp/actions/runs/34601732349), le préchargement explicite de la copie fournie de `libomp.dylib`, suivi du bridge, a réussi sur macOS 14.8.9 ARM64 sans installation Homebrew. Cette expérience ne qualifie que le chargement des bibliothèques ; les tests de calcul restent obligatoires après intégration. Le bridge NuGet porte une cible minimale macOS 15 dans ses métadonnées : ce point reste à traiter pour une distribution officiellement qualifiée macOS 14, même si le chargeur a accepté ce binaire pendant l'expérience.

La [construction des archives du premier commit](https://github.com/skyline624/ComfySharp/actions/runs/34602207188) a réussi sur les trois runners : restaurations verrouillées, publication autonome du Host et de Desktop, sommes SHA-256, archives `tar.gz` et contrôle des permissions exécutables Unix. Aucun binaire n'a été téléversé. Ce contrôle ne lance pas les applications sur Linux/macOS et ne contient aucun moteur de modèle ; il ne vaut pas qualification d'installation ou de génération.

La correction [`27245b8`](https://github.com/skyline624/ComfySharp/commit/27245b8ba9f150500fc84ac56395922ca5c727ca) précharge OpenMP puis le bridge depuis les seuls répertoires natifs de l'application. Elle conserve leurs handles pour la durée du processus et mémorise les échecs. Elle ne modifie aucun binaire natif et n'installe rien via Homebrew. Après revue, la [campagne CI correspondante](https://github.com/skyline624/ComfySharp/actions/runs/34602491042) a réussi sur **Windows, Ubuntu 24.04 et macOS 14** : 177 tests par OS, aucun ignoré (Core 47, Host/Catalog 71, Inference 36, Workflow 17, Desktop Headless 6), contrôle du catalogue et 25 répétitions du probe CPU. Le défaut de chargement initial est donc corrigé pour ces opérations. MPS, les familles de modèles et la qualification complète du bridge pour macOS 14 restent ouverts.

Sur le runner macOS, `Process.PrivateMemorySize64` a renvoyé zéro à chaque observation. Ces valeurs ne constituent pas une mesure exploitable de la consommation mémoire ; elles ne signifient pas que les calculs n'ont alloué aucune mémoire. Un collecteur adapté devra remplacer cette observation avant les campagnes de ressources macOS.

La tranche [`52c79fe`](https://github.com/skyline624/ComfySharp/commit/52c79fec7c63d830117aa7ea9a5298ed77da9fbf) passe **250 tests par OS** dans la [campagne CI `34606696624`](https://github.com/skyline624/ComfySharp/actions/runs/34606696624) : Core 67, Host/Catalog 90, Inference 70, Workflow 17 et Desktop Headless 6. Les restaurations verrouillées et compilations Release réussissent sur Windows Server 2025, Ubuntu 24.04 et macOS 14 ARM64. La compilation locale Windows des 17 projets ne produit aucun avertissement ni erreur.

Cette même campagne lance une **fenêtre native Desktop sur chaque OS**, supervise un Host distinct, compile un document vers prompt et vérifie son résultat scalaire avant fermeture. Linux utilise X11 avec Xvfb ; ce test ne couvre pas un bureau Wayland natif. Le contrôle des 1 160 entrées et de leurs documents de preuve, l'inspection safetensors en processus neuf sans distribution native libtorch et les 25 répétitions du probe CPU passent également sur les trois OS. Le lot 0 satisfait donc ses critères de dépôt indépendant, compilation et communication Desktop/Host ; cela ne qualifie pas une famille, une installation propre finale, CUDA Linux ou MPS.

Les tests supplémentaires couvrent les valeurs natives partagées du moteur, leurs vues, les sorties multiples, les branches lazy, les allocations asynchrones, les erreurs et l'annulation. Cinq cas utilisent de vrais tenseurs TorchSharp dans le moteur. Le lecteur matérialise sur CPU F32, F16, BF16, F64, I8, U8, I16, I32, I64 et BOOL ; les bits demi-précision, y compris NaN et sous-normaux, sont conservés. U16/U32/U64 restent lisibles comme métadonnées avec refus explicite de matérialisation. Le cas de remplacement d'un fichier ouvert est spécifique à Unix et passe sur Linux/macOS : son retour anticipé sous Windows ne prouve pas ce comportement. Aucun test n'est marqué ignoré par xUnit ; cette distinction de plateforme reste explicite.

`ComfySharp.ModelInspect` vérifie en processus séparé une fixture synthétique de 218 octets sans distribution native libtorch. Son SHA-256 attendu est `2d8ae1aabaeaad341c2fc30cfd9af025506805fb6e83a70c3be61dbec952c47a`. Le hachage utilise le même fichier ouvert que la lecture des métadonnées. Les contrôles de CLI couvrent aussi les limites d'affichage et les arguments invalides. L'inspection de métadonnées ne qualifie ni l'architecture ni les poids d'un modèle.

Le référentiel contient désormais 1 160 entrées et 940 identités de nœuds conservées : 652 candidats locaux, 275 candidats distants exclus et 13 déclarations ou références non enregistrées dans ces listes. Les 663 fiches de schéma comprennent 533 résolutions statiques et 130 partielles, avec 281 champs explicitement indéterminés. Les contrôles .NET vérifient les correspondances, les sources figées et les contradictions de complétude ; les trois marqueurs de complétude restent faux. Le contrôle `--release` échoue comme attendu, avec 883 exigences encore ouvertes. Ces nombres décrivent la couverture du référentiel, pas celle des fonctionnalités portées.
