# Source Contains/Compare et minuscules Unicode — 5466f97

La collecte conserve **40 cas des vrais nœuds source** et **68 empreintes exhaustives par plan Unicode/contexte**, sous CPython 3.12.10 et Unicode 15.0.0. Le [laboratoire](../../labs/text-comparison-source/README.md) a été publié au commit [`5466f97312d8dbdbe5c656a6c31061895afc9af3`](https://github.com/skyline624/ComfySharp/commit/5466f97312d8dbdbe5c656a6c31061895afc9af3) avant la collecte. Cette preuve ne compare encore aucun résultat C#.

L'artefact `reference.json` mesure **280652 octets**, SHA-256 **`12935c33d4632a79e499a7c24b052d88036aee8323e45ea92d923443b2b9e04f`**. La [preuve JSON](text-comparison-source-5466f97.json) conserve ses pins, ceux du protocole et des sources, les résultats par cas et les 68 digests. Il s'agit d'un **audit des artefacts par l'auteur du collecteur**, avec 561 contrôles stdlib ; ce n'est pas une revue personnelle indépendante. Aucun corps source, builtin `lower`, .NET ou calcul natif n'a été rejoué pour cet audit.

## Provenance et lancement

Les trois fichiers du laboratoire correspondent exactement aux blobs du commit admis. Le protocole SHA `0d00fa0c7721b07b2fc12381f60ed364b286963553577fcdb6672c028e18aa92`, les trois helpers Prefix publiés et les dix blobs du backend [`1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`](https://github.com/comfy-org/ComfyUI/tree/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a) ont été revérifiés, ainsi que les lignes, AST et segments des **75 déclarations**. Les corps StringContains/StringCompare, les classes de schéma, le binder, le mapper et NodeOutput sont réels ; seul le cache/serveur du laboratoire est adapté. Aucune table C# ou sortie produit ne calcule les attentes.

Root rapporte deux tentatives préalables arrêtées avant tout corps de nœud : l'alias Store de l'exécutable ne pouvait pas être lu pour son hash, puis le lancement direct Appx a été refusé. La collecte réussie utilise un environnement virtuel stdlib créé avec `--without-pip`, toujours sous Python 3.12.10/Unicode 15.0.0. Le protocole n'a pas changé et la destination extérieure était neuve. Ces échecs de lancement ne constituent pas deux calculs de corpus supplémentaires.

L'exécutable enregistré est le **lanceur de cet environnement virtuel**, 274424 octets, SHA `0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14`. Ce hash n'atteste pas la DLL effective de l'interpréteur ni toutes ses bibliothèques dynamiques. Aucun chemin de machine n'est publié ici.

## Quarante cas, avec leurs frontières

| Partition prospective | Cas | Observation |
|---|---:|---|
| Comparables | 36 | 35 cas de prédicats, soit 39 booléens avec le mapping, et un chemin blocker |
| Erreurs de type au mapper | 2 | TypeError sur texte null ; AttributeError sur motif numérique en mode lower |
| Modes Compare invalides, source-only | 2 | Retour collecté avec `outputs=[]`, sans booléen ni TypeError observé |

Les 24 cas Contains et 16 Compare donnent ainsi **38 retours et deux exceptions**. Les modes invalides sont introduits volontairement sous la frontière de validation globale COMBO. Leur retour vide n'autorise pas leur admission par le produit et n'est pas remplacé par `false`. La possibilité prospective d'une erreur de fusion n'a pas été sélectionnée par cette exécution ; le protocole est resté intact.

Les schémas sont stables entre cas. La metadata V3 de Compare expose `mode` comme `COMBO`, avec `multiselect=false` et les options ordonnées `Starts With`, `Ends With`, `Equal`. Les deux nœuds gardent `case_sensitive` BOOLEAN requis, default vrai/advanced vrai, les STRING multilignes, les sorties BOOLEAN ordinaires et le module source. Le défaut de widget ne constitue pas une injection des entrées HTTP manquantes.

Les observations couvrent casse sensible/insensible, vides, astral, NUL/newline, expansion de İ, sigma final/médian, U+0345 et plusieurs caractères ignorables. Elles distinguent lower de casefold et conservent la différence composé/décomposé. Les arguments complets sont traités séparément avant comparaison. Deux cas mappés produisent trois résultats chacun avec repeat-last.

Le chemin blocker conserve le message `Execution Blocked: stop`, rend un blocker silencieux et n'appelle ni le corps ni le binder. Au total, le run observé de chaque cas conserve **43 appels aux corps source et 43 retours du binder**. Leurs arguments concordent avec les colonnes réellement acquises et leur expansion conjointe. Les 40 snapshots prompt/cache sont inchangés.

Chaque cas est exécuté off/on/off dans des namespaces frais. Les 40 encodages centraux ont été rehashés et correspondent aux **120 hashes enregistrés**. Les deux runs sans observation ne sont pas conservés en brut : leur égalité reste une attestation du collecteur, tandis que l'encodage central est directement vérifiable.

## Empreintes du builtin lower

Le volet distinct `helperLower` utilise exclusivement le builtin CPython. Il visite les **1 112 064 scalaires Unicode valides**, de 0 à 0x10FFFF sans les surrogates, en ordre croissant. Le plan 0 contient 63488 scalaires ; les seize autres en contiennent 65536 chacun. Quatre constructions sont évaluées :

1. `chr(cp)`
2. `chr(cp) + 'Σ'`
3. `'A' + chr(cp) + 'Σ'`
4. `'AΣ' + chr(cp) + 'A'`

Cela donne **17 × 4 = 68 digests**, **4 448 256 entrées par passage** et **13 344 768 appels au builtin** sur les trois répétitions. Pour chaque entrée et sortie, le framing est `uint32LE(cp) || uint32LE(longueur UTF8) || octets UTF8 stricts`, sans BOM, séparateur ou normalisation ; le hash SHA-256 repart de zéro par plan/mode.

L'audit a reconstruit les **68 digests d'entrée** et leurs comptes/tailles sans appeler `lower`. Les **204 hashes de sortie enregistrés** sont identiques par triplet. Les sorties individuelles ne sont pas conservées et leurs digests n'ont pas été recalculés par cet audit : ils proviennent de la collecte réelle. Il n'y a pas de fichier à plusieurs millions de lignes.

Les trois constructions contextuelles sondent les propriétés qui interviennent autour du sigma ; les cas de nœuds complètent les séquences à plusieurs ignorables. Ce corpus ne parcourt pas toutes les chaînes Unicode ni tous les contextes possibles. Il ne qualifie pas encore le helper C#, ses tables dérivées, les nœuds du port, les routes HTTP/Desktop ou une plateforme de calcul.
