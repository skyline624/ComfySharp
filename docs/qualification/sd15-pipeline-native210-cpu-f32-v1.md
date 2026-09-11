# SD1.5 pipeline — profil prospectif réduit v1

**Protocole prospectif publié avant tout forward du parcours.** Le [protocole figé](../../labs/sd15-pipeline-source/protocol.json), SHA-256 `f0e4f537414c6f8692c852832d040fe3fd5dc23fdc270bb351a16bbba6cd75e7`, fixe quatre cas avant leur collecte. Aucun résultat numérique ni modèle n'est qualifié. Cette tranche couvre uniquement les dimensions réduites ; le collecteur sera revu et publié séparément avant son premier calcul de référence.

## Données figées

- CLIP-L réduit : H16 / intermédiaire32 / 2 couches / 4 têtes / QuickGelu, vocabulaire49408 et positions77 ; **projection présente**. U-Net SD1 réduit : base32 / contexte16 / 4 têtes fixes / projection convolutionnelle. VAE classique : base32. Configurations source complètes dans le JSON.
- **971 poids sélectionnés**, soit 37 CLIP +686 U-Net +248 VAE, 14 534 459 paramètres /58 137 836 octets F32. Recette `sha256-name-lcg-high16-power2-v1` sur le nom canonique local au composant. L'expression par indice est indépendante, pas un LCG récursif. Les digests de schéma portent uniquement sur noms/shapes, jamais sur des sorties antérieures.
- La source utilise l'AST `CLIPTextModel` exact, calcule son pooled projeté auxiliaire, puis le wrapper SD1 sélectionne le pooled non projeté. Le port conditionne le calcul projeté à l'option : cette distinction doit être déclarée. Projection inutilisée par le conditionnement U-Net ; ce corpus **ne démontre pas la projection absente**. Un éventuel `SDClipModel.logit_scale` construit par le wrapper est un paramètre de wrapper inutilisé et déclaré séparément, pas un des37 poids du transformer.
- Quatre textes publics proviennent des chunks source déjà publiés : vide, `a (b (c:2):3)`, `a ((b))`, et le texte exact `packing/length-76`, dont l'espace terminal est conservé. Cinq chunks de77, **385 triplets** IDs/bits F64 du poids/word ID. Les poids restent leurs bits IEEE754 exacts ; aucun arrondi décimal implicite.
- **20 entrées distinctes** : huit ensembles IDs/poids pour les quatre textes et douze tenseurs F32 (bruit, zéro et sigmas). Les cas font **28 références** à ces entrées, réutilisations de textes comprises. Leurs formes, octets et empreintes sont fixés ; aucun résultat antérieur de pipeline n'est repris.

| Cas | Positif / négatif | Sigmas F32 | CFG | Maximum denoise |
|---|---|---|---:|---|
| empty-one-step | empty / empty | 1.5,0 | 1 | false |
| weighted-three-step | nested / implicit | 1.5,0.75,0.25,0 | 3.5 | false |
| two-chunks-separate | length-76 / empty | 1.5,0.5,0 | 3.5 | false |
| maximum-start | implicit / empty | Smax,1.5,0 | 7 | true |

Smax existe déjà dans les trois corpus sampling source publiés : **14.614641189575195**, F32 **0x4169d592**, octets LE **92d56941**, index999. Les trois tables de1000sigmas rehashées en stdlib ont le même SHA `187a5a207118003afdd3af1298e04c99829ac795f0b9d658fe5466ac56e06649`. Le protocole conserve les trois SHA de fichiers donateurs et le JSON pointer. **Aucune nouvelle préparation de Tensor requise** ; une valeur de schedule source existante est réutilisée comme entrée, distincte d'un expected de pipeline.

## Calcul à collecter, non encore exécuté

Batch1, bruit/zéro `[1,4,4,5]`, image finale `[1,32,40,3]`. Le bruit est une entrée déterministe de recette, **pas un bruit gaussien ni une API seed compatible**. Native CPU/F32, attention_basic CLIP/U-Net, normal_attention VAE, EPS/sigma_data1, tables discrètes1000 par défaut, **Separate** et churn0. Négatif vide réellement encodé, jamais substitué par null. Sigmas explicites : aucune promesse de normal_scheduler.

Ordre : tokenizer réel → deux encodages CLIP → `EPS.noise_scaling` (sqrt(1+sigma²) au maximum, sigma sinon, +zéro) → Euler tensor/F32 → identité inverse_noise_scaling EPS → **division une fois par le scalaire double0.18215** → VAE.decode/process_output exact, tampon NCHW puis vue NHWC. Ne pas réécrire les maths, importer .NET comme source, ajouter une `.contiguous()` de calcul ou remplacer une politique sur échec.

Chaque cas répète trois fois off/on/off. Les4cas totalisent8pas et15appels U-Net par passage ;45appels U-Net,24 encodages CLIP et12 décodages VAE pour les3 répétitions. Captures centrales obligatoires :8frontières par cas et4records par pas Euler, **64 records** sur la répétition instrumentée. Les paramètres/entrées sont rehashés avant/après ; les trois sorties finales doivent se répéter exactement dans un même processus. Image clampée seule insuffisante pour l'acceptation.

## Source, runtime et comparaison

Backend `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`. Les15 fichiers source exécutables, listes de symboles et SHA AST ont été dérivés des blobs Git en stdlib ; les deux lambdas VAE sont épinglées séparément. Les helpers existants sont épinglés raw et Git canonical ; leur lecture comme référence ne permet pas d'appeler le générateur/fill CLIP stock, qui emploie une autre recette. Les adapters d'infrastructure nécessaires au wrapper CLIP et au VAE devront être explicités dans la future revue collecteur.

Le C# tokenize les textes puis compare exactement les chunks/bits d'entrée. Le laboratoire torch consomme les chunks du corpus tokenizer source, **jamais ceux du produit**. Profil tokenizer déjà indépendant : Transformers5.14.1/tokenizers0.22.2. Profil numérique : Python3.12.10 /torch2.10.0 CPU /numpy2.2.6 /einops0.8.1, intra/inter-op1, dispatch auto. Pas d'ajout Python au produit.

Pour locks/helpers : admission explicitement CRLF→LF seulement vers le pin Git ; avant/après exécution, les bytes **raw du même fichier** doivent rester identiques. Rapporter modules réellement chargés et SHA séparément des inventaires, capacité observée séparément du mode demandé. Pas de chemins privés dans le manifeste.

Les indices Euler sont exacts. Les captures scalaires `sigma` et `sigmaHat` ont la forme `[]`, le dtype F32 et exactement les bits de `sigmas[index]`, avec churn=0. Elles ne bénéficient pas de la tolérance des activations.

Comparateur prospectif : `abs(actual-reference) <= 3e-5 + 3e-5*abs(reference)`, structure/inputs exacts, non-finis refusés. MaxAbsolute/maxToleranceFraction/nombre horsborne/première frontière divergente obligatoires. Un échec n'autorise ni relâchement du profil, ni écrasement de fixtures, ni extrapolation entre OS. Le manifeste final n'est publié qu'après les contrôles d'intégrité. Les sorties restent à collecter et à auditer séparément.

## Vérifications de cette préparation

JSON valide, 971 noms/shapes déclarés via digests par composant,4cas,20inputs/28bindings,5chunks/385triplets ; Smax issu des3tables dont les payload hashes ont été recalculés ; sigmas monotones/positifs avant zéro et booléens maximum conformes à la règle source. Noise/zero/sigma et chunks ont uniquement été sérialisés/recréés par calcul stdlib pour calculer leurs hashes. Aucun import Torch/numpy, création Tensor, forward, poids privé ou sortie attendue de pipeline.
