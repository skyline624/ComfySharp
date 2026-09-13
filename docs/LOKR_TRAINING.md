# Fondations d'entraînement LoKr

`TrainableLokrPatch` porte la reconstruction de `LokrDiff`, ses paramètres
entraînables, les gradients natifs et l'export de ses facteurs. Il s'intègre
aux optimisateurs existants et au chemin de reconstruction du U-Net SD.
La [source figée](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/weight_adapter/lokr.py)
reste la référence fonctionnelle.

Les côtés directs ont priorité sur les côtés décomposés lorsqu'ils coexistent.
Les paramètres décomposés restent enregistrés, avec un gradient absent s'ils ne
sont pas utilisés. Chaque côté reconstruit applique indépendamment son facteur
`alpha / rang` : reconstruire les deux côtés applique donc alpha deux fois.
Les côtés directs ignorent alpha, même lorsque la fabrique active son gradient.
Les clés `lokr_w1`, `lokr_w2`, `lokr_w1_a`, `lokr_w1_b`, `lokr_w2_a`,
`lokr_w2_b`, `lokr_t2` et `alpha` sont conservées dans les snapshots et fichiers.

La reconstruction Tucker utilise l'einsum source à quatre dimensions. Dans le
cas multicanal collecté, son résultat non contigu fait échouer `torch.kron`
dans la source. Le port conserve cette erreur ; il n'ajoute pas silencieusement
une copie contiguë. Le cas Tucker monocanal collecté s'exécute et ses gradients
sont comparés. La présence de ces cas ne qualifie pas toutes les géométries Tucker.

## Fabrique et vérifications

`SdTrainableAdapterSet(..., algorithm: "LoKr")` factorise séparément les canaux
d'entrée et de sortie selon l'algorithme amont. Le premier côté est nul, le
second suit l'initialisation Kaiming source. Aucun constructeur Linear ne
consomme de tirage supplémentaire. Les différences de normalisation et biais
restent présentes : **686 cibles et 1 250 paramètres** par topologie réduite.
Le budget tient compte des véritables facteurs, de leurs dimensions spatiales
et d'alpha.

Les 52 tests comprennent :

- 36 scénarios de calcul avec deux mises à jour, couvrant quatre optimisateurs,
  matrices, poids Conv1d/2d/3d et Tucker monocanal ; quatre cas d'erreur source.
- Quatre contrôles de sauvegarde, conservation des dtypes et propriété des données.
- Quatre fabriques SD1/SD2 aux rangs 2 et 7, avec tous les paramètres comparés et
  états RNG exacts ; un contrôle de 72 factorisations de dimensions.
- Deux parcours du U-Net réduit avec gradients et base inchangée, puis un contrôle
  d'erreurs, d'annulation et de libération native.

La fixture de fabrique stocke les vrais octets Float32 source, dédupliqués par
SHA-256 et compressés, pour réduire son espace disque. Aucun poids de modèle
préentraîné n'est inclus. Les profils numériques restent `3e-5` absolu et relatif.

## Travail restant

Le [chargement LoKr d'inférence et DoRA](LOKR_INFERENCE.md) dispose maintenant de
références source et d'un diagnostic SD1.5 CPU entraîné puis rechargé en mémoire.
La reprise d'entraînement de fichiers LoKr, le bypass, les workflows complets,
les autres précisions et les plateformes restent à vérifier ou porter.
Le bypass d'entraînement est refusé explicitement tant que son chemin
opérationnel n'est pas porté ; une reconstruction des poids ne le remplace pas.
Le nœud `TrainLoraNode` public reste incomplet. Aucune famille ni plateforme n'est
qualifiée par les tests réduits. Voir [la preuve](qualification/lokr-training.json).
