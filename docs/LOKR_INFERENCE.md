# Chargement et réutilisation LoKr

Le chargeur commun accepte les facteurs LoKr directs, décomposés et Tucker 2D,
alpha et DoRA, depuis un fichier safetensors ou des tenseurs natifs. Les facteurs
F16/BF16 deviennent des snapshots Float32 possédant leurs données. Le plan inspecte
les métadonnées et contrôle le budget avant lecture des facteurs ; il reste lié
à sa source. Les snapshots survivent au fichier et aux paramètres d'entraînement.

Les contrats proviennent du [LoKrAdapter figé](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/weight_adapter/lokr.py)
et du [chargeur commun](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/lora.py).
LoKr remplace LoHa et LoRA pour un même alias. Les alias suivants, différences de
normalisation, puis différences explicites conservent leurs priorités source.
Les clés reconnues mais remplacées restent comptées comme consommées.

## Différences avec l'entraînement

Les côtés directs ont priorité sur leurs facteurs décomposés. Les facteurs
inactifs restent conservés et budgétés. La reconstruction d'inférence applique
alpha une seule fois ; si les deux côtés sont décomposés, le rang du second
côté remplace celui du premier. Deux côtés directs ignorent alpha.
Le [chemin d'entraînement](LOKR_TRAINING.md) applique alpha séparément à chaque
côté décomposé. Un fichier entraîné puis rechargé peut donc changer de résultat
selon ses facteurs, comme dans la source. Cette distinction reste explicite.

L'inférence ajoute des axes spatiaux au premier côté seulement si le second est
4D. Les poids 3D et 5D suivent le padding des axes de `torch.kron`, puis le reshape
de destination ; cela diffère aussi de `LokrDiff`. DoRA reproduit la normalisation
sur le poids original pour l'axe de sortie et sur le poids modifié pour l'autre axe.

Dans les cas Tucker multicanaux collectés, le résultat non contigu d'einsum fait
échouer `torch.kron`. ComfyUI journalise cette erreur et retourne le poids original.
ComfySharp propage l'erreur pour que l'adapter non appliqué ne soit pas annoncé
comme réussi. Il n'insère pas une copie contiguë modifiant ce comportement de
calcul. Cette différence de signalement est documentée ; ces cas ne sont pas
comptés comme des reconstructions opérationnelles.

## Preuves

Le laboratoire séparé `labs/lora-training-source/lokr_inference.py` exécute l'AST
figé ; seule la conversion de device est adaptée au CPU. Les tests .NET embarquent
le résultat statique et son SHA-256, sans Python. Les 155 tests comprennent
135 reconstructions réussies, 15 erreurs source, quatre scénarios de priorité et
un contrôle des erreurs/ressources. Ils couvrent aussi la fermeture de la source,
la mutation des facteurs d'origine, le transfert, les owners retenus, les budgets
et l'annulation. Les tolérances sont verrouillées à `atol=rtol=3e-5`.
Les deux parcours U-Net réduits d'entraînement vérifient maintenant le rechargement.

Un diagnostic avec les vrais poids SD1.5 sur Windows CPU a effectué deux pas SGD
avec le paramètre de factorisation/rang 7, 686 cibles et 1 250 paramètres.
Il a produit 968 gradients finis à chaque pas ; les 282 alpha directs n'ont pas de
gradient, conformément à la source. Les pertes sont 1,1856256 puis 1,1707271.
960 paramètres ont changé ; la base est inchangée et la prédiction après
rechargement en mémoire est identique octet par octet. Le budget des paramètres
est 137 523 272 octets. Aucun checkpoint n'a été copié et aucun adapter n'a été
sauvegardé lors de ce diagnostic. Voir [la preuve détaillée](qualification/lokr-inference.json).

```text
dotnet run -c Release --project tools/ComfySharp.RuntimeProbe -- sd-all-adapter-train --algorithm LoKr --rank 7 --checkpoint <checkpoint-existant.safetensors> --report <nouveau.json> --device cpu
```

La [reprise d'entraînement LoKr](LOKR_RESUME.md) possède maintenant un chemin
distinct et ses propres références. Ce parcours utilise des entrées de prédiction
synthétiques réduites avec de vrais poids. Il ne valide pas un entraînement complet
sur images, les gradients du modèle préentraîné contre ComfyUI, le bypass, les autres
précisions ou CUDA/MPS. Les 155 tests ne qualifient aucune famille complète.
Le nœud public `TrainLoraNode`, les workflows réels et les plateformes restent
requis pour la V1. Le [bypass LoKr](LOKR_BYPASS.md) dispose depuis d'opérateurs
et de tests dédiés ; le [rechargement SD1.5](LOKR_LINEAR_DISPATCH.md) est exact
dans le même mode inférence, sans qualifier encore la famille complète.
