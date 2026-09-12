# StringFormat — première collecte source 52fe346

Le laboratoire prospectif a été [publié au commit 52fe346c74c2db14b6c73aec804c215674550f4d](https://github.com/skyline624/ComfySharp/tree/52fe346c74c2db14b6c73aec804c215674550f4d/labs/string-format-source) avant sa première exécution par root. La collecte Python 3.12.10 sous Windows termine avec **44 cas : 32 retours et 12 erreurs prescrites**. Son fichier brut mesure **408 527 octets**, SHA-256 `a6c1aa98c333851195d0bff97f3e1305835634a1cb922dcc102dabe5e07871e0`.

Cette preuve est un **audit des artefacts par l’auteur du collecteur**. Elle ne se présente pas comme une revue indépendante par une autre personne. Aucun import du collecteur, replay source, test .NET, calcul natif ou smoke Desktop n’a été exécuté pendant cet audit. Aucun résultat C# n’est qualifié ici.

## Provenance et portée exacte

Les **506 contrôles de lecture** vérifient le protocole publié, les trois fichiers du nouveau laboratoire, les six fichiers helpers publiés aux commits Names `4312bda` et Prefix `ae241a0`, et les **dix blobs Git / 73 déclarations AST** du backend figé `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`. Chaque déclaration conserve ligne, empreinte AST et empreinte du segment source. Les pins et les résultats par cas sont dans la [preuve JSON](string-format-source-52fe346.json).

Le collecteur appelle le vrai [StringFormat source](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_extras/nodes_string.py#L9), dont le corps exécute `f_string.format(**values)`. Les Schema, TemplateNames, acquisition, mapper, binder et NodeOutput sont issus de la source. **Aucun corps-écho ni formateur de remplacement n’est utilisé.** Le builtin appartient à l’interpréteur déclaré Python 3.12.10 ; cette preuve ne prétend pas disposer d’un hash du binaire interpréteur chargé.

Les 44 métadonnées source sont identiques : `StringFormat`, `Format Text`, groupe `values` Names a–z/min=0 avant `f_string`, sortie STRING ordinaire et InputIsList=false. La description source annonçant les possibilités complètes de Python reste une observation de métadonnées source ; elle n’est pas une promesse du futur profil produit partiel.

| Partition prospective | Cas | Résultat source observé |
|---|---:|---|
| Comparable dans le profil texte | 24 | 24 retours, 26 chaînes après mapping |
| Erreur source prescrite | 12 | 6 ValueError, 4 KeyError, 2 IndexError |
| Hors profil produit | 8 | 8 retours conservés comme observations source |

La présence de trois lignes mappées dans un cas explique les **34 chaînes retournées**, les **46 appels du vrai corps** et les **46 retours observés du binder**. Les erreurs ont lieu dans la phase `mapper`, où le corps est invoqué ; aucune sortie factice n’est ajoutée aux cas en erreur.

## Répétitions, snapshots et ordre des erreurs

Chaque cas s’exécute dans des namespaces frais, off/on/off. L’audit recalcule exactement les **44 hashes centraux** et retrouve les trois empreintes identiques enregistrées par cas, soit 132 empreintes. Les deux résultats bruts off ne sont pas conservés : leur égalité est attestée par le contrôle du collecteur, tandis que le hash central est recalculable depuis l’artefact. Les 44 couples de snapshots entrée/cache sont inchangés, avec réencodage préservant types et ordre.

Les profils observent les arguments nommés du vrai StringFormat et les retours réels de `build_nested_inputs`. Les valeurs et formats concordent entre les deux frontières ; l’ordre racine du dictionnaire du binder est enregistré séparément de ses membres. Aucun callback de blocage n’est émis dans ce corpus de formatage.

Les cas de priorité confirment la distinction entre parsing structurel et traitement sémantique : un nom absent précède une conversion inconnue ou un spec non pris en charge par le futur produit ; une conversion malformée ou une accolade non fermée est détectée avant le lookup. Un champ absent antérieur précède aussi une erreur d’accolade située plus loin dans le format. Enfin, la conversion inconnue `!q` précède la résolution d’un argument absent dans un spec imbriqué. Les types et messages source sont conservés, sans exiger que les textes de diagnostics C# leur soient identiques.

## Observations utiles au profil partiel

Le centrage impair donne `*abc**` ; le remplissage astral donne `😀xyz😀😀`. La précision zéro rend une chaîne vide, et la précision de deux code points conserve `😀é`. La séquence combinante reste mesurée en code points, pas en graphèmes. Les conversions `!s` précèdent réellement largeur et précision : None produit `****None`, True produit deux espaces puis `Tru` puis deux espaces, et l’entier converti en texte produit `00000-42` avec remplissage à gauche.

Les huit cas hors profil réussissent dans la source : repr/ascii, item, attribut, spec imbriqué, float, conteneur, format numérique entier et largeur à zéro initial. En particulier `{a:05}` appliqué à `x` donne `x0000` ; le refuser localement doit rester un diagnostic de fonctionnalité hors profil, pas une affirmation que Python rejette cette syntaxe. Ces sorties ne deviennent pas des attentes de réussite du port C#.

La collecte adapte uniquement cache et serveur ; elle n’exécute pas les corps producteurs, PromptExecutor, la validation HTTP complète ou le frontend. Les contrôles de budget, annulation, valeurs natives et ownership sont des tests produit distincts. Les futurs comparateurs devront conserver les partitions 24/12/8 et leurs limites ; leur réussite, ainsi que celle d’un trajet Host ou Desktop, devra faire l’objet d’une preuve ultérieure.
