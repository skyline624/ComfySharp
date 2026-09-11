# Qualification initiale

Campagne du 11 septembre 2026, sources de la première tranche 0.1.0-dev. Ces résultats qualifient seulement les fonctions décrites, pas une architecture de modèle ni la V1.

| Environnement | Résultat observé |
|---|---|
| Windows x64, SDK .NET 10.0.300 | Compilation Host, Core, Workflow, Inference, Desktop et tests |
| TorchSharp 0.107.0.0 / libtorch CPU 2.10.0 | Opérations natives, lecteur F32, RNG par générateur, Euler élémentaire, gradients et mise à jour SGD |
| NVIDIA GeForce RTX 3090, 24 576 MiB, pilote 610.88, libtorch CUDA 12.8 / 2.10.0 | 25 répétitions GPU réelles, résultats ci-dessous |
| Linux x64 CPU et macOS arm64 CPU | Verrous de dépendances restaurés depuis Windows ; aucune exécution locale sur ces OS |
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

Le lecteur safetensors teste offsets, doublons, chevauchements, formes/overflow, limites d'allocation, métadonnées, données tronquées et annulation après allocation avec libération. Seul le chargement F32 est annoncé. Aucune performance de génération, qualité d'image ou fidélité de modèle n'a été mesurée ; aucun poids n'a été téléchargé pour cette campagne.

Tests de contrat supplémentaires : validation avant file, annulation d'un ancien job sans toucher son successeur, WebSocket initial, SQLite/settings/prune, conservation du document, édition/undo et démarrage de Desktop avec son Host. Les totaux finaux de tests et résultats CI sont publiés avec la révision correspondante.

Les verrous `packages.<runtime>.cpu.lock.json` existent pour les deux projets natifs sur les trois runtimes ; le probe a aussi `packages.win-x64.cuda.lock.json`. Les sept restaurations `--locked-mode` ont réussi. Une restauration croisée n'est pas une qualification de la plateforme cible.

Après revue et corrections, les suites locales comptent **164 tests réussis, aucun ignoré** : Core 47, Host/Catalog 71, Inference 23, Workflow 17, Desktop Headless 6. Elles couvrent notamment les régressions d'annulation HTTP chunked, de conservation des valeurs lazy, de clés JSON dupliquées, d'annulation après allocation native, de sauvegarde asynchrone et d'IDs numériques équivalents.

Une construction Windows autonome de Desktop et Host a aussi réussi son `--smoke-test` avec `PATH` limité à `Windows/System32`, sans override du chemin Host : lancement du Host voisin, health/catalogue, exécution réelle d'un nœud texte, lecture du résultat, fermeture. Cela prouve ce parcours local du socle ; une installation sur machine propre et la qualification V1 restent à faire. La publication des binaires attend la clôture des notices et de l'inventaire de leurs composants.
