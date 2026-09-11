# Matrice de migration

Le [manifeste](manifest.json) contient **1 106 entrées statiques** : 888 déclarations de nœuds (622 locales, 265 distantes exclues, une déclaration de test), configurations de modèles, formats latents, samplers, schedulers, adapters, optimisateurs, losses et formats de quantification. Les lignes de nœuds ne sont pas un décompte d'inscriptions actives. `catalogueComplete` reste **false**.

Chaque ligne garde son identifiant, sa source publique figée, sa catégorie, son périmètre, un identifiant de scénario et des statuts indépendants. Les paramètres détaillés, fixtures de référence et preuves matérielles restent à compléter. Une case `false` signifie absence de validation, même si une primitive apparentée a passé des tests.

Treize nœuds utilitaires sont présents dans le registre C#. Aucun modèle image, vidéo, audio ou 3D n'est annoncé compatible. La primitive Euler ne valide pas à elle seule le sampler d'un workflow complet. Le probe CUDA ne valide pas le catalogue GPU.

```sh
dotnet run --project tools/ComfySharp.Catalog -- docs/capabilities/manifest.json
dotnet run --project tools/ComfySharp.Catalog -- docs/capabilities/manifest.json --release
```

Le second contrôle doit actuellement échouer : il interdit une déclaration V1 avec des entrées obligatoires non qualifiées ou un inventaire incomplet. Le premier détecte les doublons, sources non figées et incohérences entre manifeste et registre.

Pour terminer le lot 1 : réconcilier toutes les inscriptions dynamiques/conditionnelles sans dépendre d'imports optionnels réussis ; détailler encodeurs/VAE/widgets, formats historiques et surfaces locales ; inventorier paramètres/dépendances/variantes ; affecter de vrais scénarios et poids sous licence avec SHA-256 ; verrouiller les tolérances **avant** l'acceptation numérique.
