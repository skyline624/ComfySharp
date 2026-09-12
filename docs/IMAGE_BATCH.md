# ImageBatch

`ImageBatch` réunit deux lots de tenseurs IMAGE. Son identifiant, ses entrées `image1` et `image2`, sa sortie IMAGE, ses alias et l'indication source `deprecated=true` sont conservés. L'éditeur affiche **Batch Images (DEPRECATED)**, comme le backend figé ; les workflows existants peuvent conserver ce nœud.

Le profil initial est CPU/Float32, NHWC RGB ou RGBA, avec des dimensions strictement positives. Le batch des deux entrées peut différer. Si les canaux diffèrent, le côté RGB reçoit un canal alpha égal à 1 ; l'alpha du côté RGBA reste présent et participe au redimensionnement. Si les dimensions spatiales diffèrent, seule la seconde entrée est recadrée au centre et redimensionnée à la hauteur et à la largeur de la première. L'interpolation est bilinéaire, avec `align_corners=false` et `antialias=false`.

Le recadrage conserve l'ordre des expressions en double et l'arrondi au pair de Python. Les marges sont retirées symétriquement. Pour certains rapports d'aspect extrêmes, le recadrage produit une dimension nulle : le calcul échoue et libère ses ressources, comme le chemin source. Il n'applique pas de correction implicite de la taille. Les valeurs ne sont pas clampées entre 0 et 1.

## Ressources

Les deux entrées sont empruntées et ne sont pas modifiées. La concaténation produit un stockage indépendant, dont la propriété est transférée au contexte du nœud. Les calculs sont réalisés sous `no_grad`, sans changer le mode du code appelant. L'annulation est contrôlée avant l'accès aux entrées et aux frontières des opérations natives synchrones.

Le plafond configurable de 512 MiB par appel additionne les payloads de l'éventuel padding, de l'éventuelle seconde image redimensionnée et de la concaténation finale. Il exclut les entrées empruntées, les temporaires internes des kernels, l'allocateur et ses caches. Les tailles sont calculées avec détection des débordements. Ce plafond local ne constitue pas une borne de RAM totale ni une option du schéma ComfyUI.

## Validation et portée

Les premiers contrôles locaux passent : 18 cas d'opérations et cinq cas du vrai nœud. Ils vérifient notamment les lots distincts, l'ajout et l'interpolation de l'alpha, les coordonnées de demi-pixel, les arrondis du recadrage, l'absence d'antialias, les vues non contiguës, la propriété de la sortie, les budgets et la libération après erreur. Ces exemples comportementaux sont distincts du corpus source numérique.

Le [protocole source](../labs/image-batch-source/README.md) fixe 19 cas avant collecte : quatre sans redimensionnement à comparer sur les octets Float32 exacts, quatorze bilinéaires avec `abs(actual-source) <= 1e-6 + 1e-6*abs(source)` et un échec de recadrage vide. Cette marge est limitée aux entrées et dimensions bornées du protocole ; elle ne valide pas tout le domaine numérique du nœud. Les 19 comparaisons passent ; l'écart absolu maximal observé dans les cas bilinéaires est zéro sur cette machine, sans changer la règle de tolérance. La [campagne finale locale](qualification/image-batch-integration.md) passe 2 362 tests, dont 53 nouveaux, et le smoke natif Desktop/Host.

La route d'éditeur utilise deux nœuds EmptyImage, ImageBatch, ImageFromBatch et PreviewAny. Ce dernier affiche le tenseur sous forme de texte. Aucun codec ni fichier image, modèle préentraîné, autre dtype ou GPU n'est qualifié par cette tranche. Les validations des plateformes requises restent ouvertes.

## Provenance

Le port suit [ImageBatch](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/nodes.py#L1955) et [common_upscale](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/utils.py#L1105) du backend figé. Le laboratoire de référence exécute leurs vrais corps dans un environnement Python séparé. L'application et les tests .NET distribués restent indépendants de cet environnement.
