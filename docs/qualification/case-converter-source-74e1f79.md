# CaseConverter : collecte source 74e1f79

La première collecte source réussie contient **40 cas de nœud et 85 plans de digest**. Elle a été exécutée après publication du [laboratoire au commit 74e1f790f0c22ddd91d1230322ed3abc44b0143a](https://github.com/skyline624/ComfySharp/commit/74e1f790f0c22ddd91d1230322ed3abc44b0143a), avec protocole inchangé. Le runner root confirme exit0 et aucun échec de lancement préalable pour cette collecte. L'artefact final `reference.json` fait **268 611 octets**, SHA256 `649e592edbb38b02220f9f7b92fda634cb6e8a3c1eea1bc5bde95e2165f98e7b`.

Il s'agit d'un **audit des artefacts par l'auteur du collecteur**, pas d'un audit de collecte par une autre personne. Les 765 contrôles utilisent uniquement la bibliothèque standard, sans importer le collecteur, rejouer une déclaration source, appeler une fonction de casse ou invoquer le produit .NET. Les [pins, comptes et observations](case-converter-source-74e1f79.json) permettent une vérification séparée.

## Source et frontières observées

Les 10 blobs et 74 déclarations AST sont vérifiés face à ComfyUI `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`. La vraie [classe CaseConverter](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_extras/nodes_string.py#L106), son Schema, le binder, le mapper et NodeOutput sont exécutés sans remplacement de corps. Les trois helpers Prefix et les trois fichiers d'origine TextComparison sont épinglés à leurs commits publiés ; les trois fichiers du nouveau laboratoire correspondent exactement au commit admis.

L'object_info observé possède les entrées requises `string` STRING multiline puis `mode` COMBO V3. Ses options ordonnées sont **UPPERCASE, lowercase, Capitalize, Title Case**, sans défaut explicite de mode. Une sortie STRING ordinaire est déclarée. L'attribution du module suit l'ordre source GET_SCHEMA puis RELATIVE_PYTHON_MODULE ; elle ne remplace pas l'état initial du schéma.

| Partition prospective conservée | Cas | Résultat observé |
|---|---:|---|
| Comparables | 28 | 27 cas de texte produisent 32 chaînes après mapping ; un blocker se propage sans appeler le corps |
| Erreurs typées source | 8 | AttributeError au stade mapper : null ou entier pour chacun des quatre modes reconnus |
| Modes invalides source-only | 4 | Une sortie contenant le texte inchangé pour chaque cas |

Au total, **32 cas retournent et huit lèvent**, avec **44 appels du vrai corps et 44 retours du vrai binder** dans les passages observés. Les quatre modes invalides atteignent le repli réel ; ils ne produisent ni une exception ni zéro sortie. Ce résultat demeure distinct de l'admission COMBO d'un moteur local. De même, les huit AttributeError ne prescrivent pas une égalité avec les diagnostics typés locaux ou les coercitions STRING du moteur.

Les six textes fixes de chaque mode couvrent vide, casse ASCII mixte, première position nonlettre, apostrophes/chiffres, expansions, digraphs titlecase, sigma et U+0345, suites de marques, astral et NUL. Deux cas utilisent des liens et caches réels pour le mapping repeat-last. Dans le cas d'ordre inversé, prompt, acquisition et binder conservent `mode, string`, tandis que les arguments du corps sont observés dans l'ordre de sa signature `string, mode` ; ces ordres ne sont pas confondus.

Chaque cas effectue trois passages frais **off/on/off**. Le profil du passage central observe les véritables objets code du corps et du binder. Les 40 encodages centraux ont été rehashés et correspondent aux 120 empreintes neutres enregistrées. Les 40 snapshots prompt/cache restent identiques. Les deux passages non observés ne sont pas conservés en brut : leur égalité reste une attestation du collecteur contrôlée par ces hashes, sans replay pendant l'audit.

## Digests des builtins

Le runtime observé est **CPython 3.12.10, Unicode 15.0.0, Windows**. Les 17 plans couvrent tous les scalaires valides de 0 à 0x10FFFF en excluant les surrogates : 63 488 dans le plan zéro et 65 536 dans chacun des seize autres, soit **1 112 064 scalaires**. Les cinq recettes exactes sont :

1. `chr(cp).upper()`
2. `chr(cp).capitalize()`
3. `('A' + chr(cp) + 'Σ').capitalize()`
4. `('A' + chr(cp) + 'Σ').title()`
5. `('AΣ' + chr(cp) + 'A').title()`

Les **85 plans** représentent **5 560 320 entrées par passage et 16 680 960 applications builtin sur trois passages réels**. Le cadre binaire est, séparément pour l'entrée et la sortie, `uint32LE(cp) || uint32LE(longueur UTF-8 stricte) || octets UTF-8`. Aucun BOM, séparateur, terminateur ou normalisation n'intervient.

L'audit recalcule les 85 empreintes d'entrées, leurs comptes et leurs tailles sans fonction de casse. Les 255 empreintes de sorties enregistrées concordent par triplets. Les lignes de sorties exhaustives ne sont pas conservées et leurs hashes n'ont donc pas été recalculés pendant cet audit. Les 68 plans lower historiques restent une régression distincte, non recollectée et non ajoutée au compte de cette campagne. Ces cinq constructions n'épuisent pas toutes les chaînes Unicode ni tous les contextes.

## Provenance et limites

L'exécutable enregistré, `python.exe` de 274 424 octets avec SHA256 `0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14`, est le **lanceur du venv stdlib**. Son hash n'atteste pas la DLL réelle de l'interpréteur ni toutes ses bibliothèques dynamiques. Aucun chemin machine privé n'est publié. La sortie finale n'est écrite qu'après les cas, les trois traversées de chaque plan et les recontrôles des fichiers, helpers, sources et HEAD.

La collecte lit les vrais builtins, sans table générée, extracteur ou résultat C# pour fabriquer des attentes. Le mapper n'est pas une validation HTTP/frontend globale. Les surrogates CLR isolés, limites de ressources, annulation et propriété des valeurs demeurent des contrats locaux. Cette preuve décrit la source collectée ; elle n'annonce aucun résultat du futur comparateur C#, aucune qualification d'autres OS ni de composants natifs.
