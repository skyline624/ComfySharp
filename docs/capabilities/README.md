# Matrice de migration

Le [manifeste](manifest.json) contient **1 171 entrées**, dont 942 lignes de nœuds : 664 locales, 277 distantes exclues et une référence custom. La [réconciliation](../audit/05-RECONCILIATION-NOEUDS.md) distingue 652 candidats locaux intégrés, 275 candidats distants, treize autres déclarations et deux doublons de déclaration conservés. Les autres entrées couvrent configurations de modèles, formats latents, samplers, schedulers, adapters, optimisateurs, losses, quantification, trois profils de tokenisation textuelle CLIP, trois profils d'encodeurs CLIP et cinq composants SD détaillés après l'inventaire initial. Les lignes de nœuds ne sont pas un décompte d'inscriptions actives. `catalogueComplete` reste **false**.

Chaque ligne garde son identifiant, sa source publique figée, sa catégorie, son périmètre, un identifiant de scénario et des statuts indépendants. [node-registration.json](node-registration.json) conserve l'ordre des modules, les conditions et les inscriptions ; [node-schemas.json](node-schemas.json) décrit 663 contrats locaux/de référence, leurs héritages, paramètres et 281 expressions non résolues. Les fixtures et preuves matérielles restent à compléter. Une case `false` signifie absence de validation, même si une primitive apparentée a passé des tests.

Vingt-cinq identifiants sont présents dans le registre C# : treize utilitaires, cinq générateurs et six traitements SIGMAS, puis PreviewAny. Ce dernier reste partiellement porté ; ses formats non pris en charge sont explicites. Les preuves de ces [primitives et sorties UI](../NATIVE_NODE_HOST.md) ne valident aucun modèle image, vidéo, audio ou 3D. La primitive Euler ne valide pas à elle seule le sampler d'un workflow complet. Le probe CUDA ne valide pas le catalogue GPU.

Les trois entrées `text-tokenizer` suivent les profils SD1 CLIP-L et SDXL CLIP-L/G, avec BPE, poids et séquences testés dans le [profil de dépendances figé](../CLIP_TOKENIZATION.md). Elles ne modifient pas les statuts des nœuds CLIPTextEncode ou des familles SD1/SDXL. Les cases de workflows réels et de qualification matérielle restent ouvertes.

Les cinq entrées `model-component` décrivent les [composants SD](../SD_COMPONENTS.md) : U-Net SD1, U-Net SD2, VAE image classique, frontière discrete/EPS/V et CFG par image entière. Leur implémentation reste partielle ; les tests locaux couvrent le CPU/F32 en inférence, avec des graphes et paramètres synthétiques à largeur réduite. Le profil `sd-components-native210-cpu-f32-v1` sépare ces références des dimensions complètes, des vrais poids et des workflows génératifs. Toutes leurs cases de plateformes et de workflows réels restent `false`, et leurs listes de preuves restent vides. Aucune famille ni aucun nœud modèle n'est annoncé compatible.

```sh
dotnet run --project tools/ComfySharp.Catalog -- docs/capabilities/manifest.json --node-evidence docs/capabilities/node-registration.json docs/capabilities/node-schemas.json
dotnet run --project tools/ComfySharp.Catalog -- docs/capabilities/manifest.json --release
```

Le second contrôle doit actuellement échouer : il interdit une déclaration V1 avec des entrées obligatoires non qualifiées ou un inventaire incomplet. Le premier détecte les doublons, sources non figées et incohérences entre manifeste et registre.

Pour terminer le lot 1 : finaliser les schémas et fournisseurs dynamiques ; détailler les variantes, encodeurs/VAE/widgets, formats historiques et surfaces locales ; compléter les dépendances ; affecter de vrais scénarios et poids sous licence avec SHA-256 ; verrouiller les tolérances **avant** l'acceptation numérique. L'énumération statique des candidats intégrés ne prouve pas les imports, la finalisation des schémas ou l'exécution d'un modèle.
