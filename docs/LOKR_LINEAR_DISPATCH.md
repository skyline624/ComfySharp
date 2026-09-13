# LoKr : comparer le rechargement dans le même mode de calcul

Les [géométries propres au bypass LoKr](LOKR_BYPASS_LOADING.md) sont maintenant
inspectées selon leurs opérateurs. Cette évolution ne change pas la distinction
entre les modes de calcul expliquée ici.

Le diagnostic initial supposait que la prédiction d'entraînement et la prédiction
d'inférence devaient être identiques bit à bit après une sauvegarde sans perte.
Cette hypothèse est fausse pour certains bypass linéaires de la source ComfyUI.
La vérification compare désormais exactement deux évaluations en mode inférence :
les paramètres d'entraînement figés temporairement avant capture, puis les
paramètres rechargés. Elle conserve séparément le hash d'entraînement et son écart.
Les échecs initiaux restent archivés ; aucune tolérance n'a été augmentée.

## Origine de l'écart

Le [collecteur indépendant](../labs/lora-training-source/lokr_linear_dispatch.py)
exécute `LokrDiff.h` et `LoKrAdapter.h` au commit ComfyUI
`1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`, avec PyTorch CPU 2.10.0.
Sur cinq géométries linéaires, les feuilles entraînables et les feuilles figées
produisent des valeurs proches mais différentes. Figer les mêmes feuilles,
sans changer leur stockage ni leur contenu, retrouve exactement l'inférence.
Les dix anciens contrôles de convolution restent des résultats négatifs utiles.

Le profileur source observe `aten::mm` lorsque le premier facteur est entraînable,
et `aten::bmm` lorsqu'il est figé, pour le même opérateur linéaire sur une vue
transposée à plusieurs dimensions. Le choix utilise les indicateurs `requires_grad`
même quand autograd est désactivé. Ce comportement est explicite dans
[`should_fold` de PyTorch 2.10.0](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/LinearAlgebra.cpp#L1787).
Les cinq nouveaux tests .NET vérifient les références et l'égalité exacte avec
les formes natives `mm` et `bmm` correspondantes. Ils ne forcent pas tous les
processeurs à produire le même écart bit à bit.

Sur le vrai SD1.5, la capture de 282 frontières de bypass situe la première
divergence dans `input_blocks.1.1.transformer_blocks.0.attn1.to_q` : entrée et
sortie de base identiques, un élément différent après bypass, écart maximal
`1,4551915228366852e-11`. Les divergences se propagent ensuite dans le réseau.
Les captures conservent les hashes finaux du parcours sans instrumentation.

## Vérification corrigée

`FrozenAdapterEvaluation` appartient à l'outil de diagnostic. Il désactive les
indicateurs des feuilles uniquement pendant une évaluation synchrone après
l'entraînement, puis restaure leurs valeurs initiales et le mode autograd, même
en cas d'annulation. Deux tests vérifient les indicateurs mixtes, les valeurs
inchangées, le retour normal, l'exception et la libération des tenseurs.
Le code d'entraînement et les opérateurs LoKr ne sont pas modifiés pour forcer
une égalité entre modes.

Le checkpoint SD1.5 partagé, de hash connu, passe deux mises à jour SGD Float32
sur CPU, avec 686 cibles et 1 250 tenseurs de paramètres. Le rechargement en mémoire
est exact en mode inférence ; la base reste inchangée. Les pertes sont
`1,1856256` et `1,1707273`. Le hash d'entraînement reste différent du hash
d'inférence, avec un écart maximal de `2,413988e-6`, explicitement publié dans
[la preuve](qualification/lokr-linear-dispatch.json).

L'ancien test U-Net réduit utilisait aussi l'égalité entre modes et échouait dans
la CI Windows 34743858674. Il conserve une égalité exacte de rechargement dans
le même mode, ainsi que la comparaison numérique entre modes avec les tolérances
`atol=rtol=3e-5` déjà utilisées par les références. Aucun test n'est ignoré.

Cette preuve concerne une prédiction brute réduite avec de vrais poids, pas un
workflow d'entraînement sur images ni une famille complète. Le chargement des
géométries spécifiques au bypass, les précisions mixtes et les plateformes GPU
restent à qualifier. L'export sur disque est adapté au même mode de comparaison,
mais aucun nouveau fichier de poids n'a été écrit pour ce diagnostic.
