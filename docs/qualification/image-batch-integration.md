# ImageBatch : intégration locale et référence source

La campagne locale Windows CPU passe **2 362 tests, dont 53 nouveaux**, sans échec ni test ignoré. Le build Release de la solution termine sans avertissement ni erreur. Les [preuves structurées](image-batch-integration.json) conservent les empreintes, les décomptes et les résultats par cas.

| Projet | PASS | Nouveaux cas |
|---|---:|---:|
| Core | 750 | 0 |
| Desktop | 44 | 1 |
| Host | 266 | 7 |
| Inference | 684 | 42 |
| Tokenization | 540 | 0 |
| Workflow | 78 | 3 |

Les 42 nouveaux cas Inference comprennent 18 opérations, cinq tests du vrai nœud et 19 comparaisons source. Les six TRX ont été vérifiés par identifiant : 2 362 identités distinctes. Certains anciens cas de fichiers malformés partagent un libellé tronqué ; ces libellés ne servent pas au décompte. Les campagnes ciblées ne sont pas ajoutées au total.

## Source collectée après publication du protocole

Le [protocole et le collecteur](../../labs/image-batch-source/README.md) ont été publiés au commit `5fd5f1e5af171d2dc43d7d37366d7f4f50e976de` avant la première exécution. La collecte utilise les vrais corps `ImageBatch` et `common_upscale` du backend `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`, sélectionnés dans deux blobs et deux déclarations AST exactes. Les dépendances de capture et leurs origines publiques sont épinglées.

L'artefact de 693 068 octets, SHA256 `879044ad5c86b0a31c96129bd680736b63e55b3142819b01a120db8b7863d2e0`, contient 19 cas et trois exécutions fraîches par cas : 54 sorties et trois échecs attendus de recadrage vide. La séquence sans observation/avec observation/sans observation reste identique après exclusion du seul champ d'observation. Le callback observe les variables du vrai `common_upscale`, y compris sa vue spatiale vide avant l'échec natif.

Un audit statique de l'artefact vérifie 606 payloads et 102 624 octets, 108 mutations successives d'entrée, les identités source et les captures avant/après. Il ne rejoue ni le calcul ni les résultats du produit. L'environnement source est CPython 3.12.10, Torch 2.10.0+cpu, avec un thread intra-op et un thread inter-op, sous no_grad. Les bibliothèques natives réellement chargées sont observées une fois après le premier corps ; les noms de packages et les locks ne remplacent pas cette observation.

## Comparaison C# et parcours applicatif

Chaque cas invoque une fois le nœud C# enregistré et compare ses captures aux trois enregistrements source. Les quatre cas sans redimensionnement comparent les octets Float32 exacts. Les quatorze cas bilinéaires conservent la règle prospective `abs(actual-source) <= 1e-6 + 1e-6*abs(source)` ; **l'écart absolu maximal observé est zéro** dans cette campagne locale. La marge n'a pas été modifiée après collecte. Le dix-neuvième cas vérifie un véritable échec natif d'interpolation sur le crop vide, avec nettoyage, sans exiger une égalité des messages Python et .NET.

Les entrées conservent leurs storages, strides et offsets réels. Les captures d'entrée, les métadonnées de sortie et les mutations sont exactes. Les mutations de coin publiées peuvent être hors du crop de la seconde entrée ; leur observation reste bornée. Une mutation de toutes les valeurs des deux entrées vérifie séparément la propriété indépendante de la sortie C#, après l'unique appel du nœud.

Le smoke démarre une vraie fenêtre Desktop et son Host supervisé, compile deux EmptyImage de tailles différentes, ImageBatch, ImageFromBatch et PreviewAny, puis vérifie le texte du tenseur vert extrait de la seconde entrée. Les sept tests Host couvrent aussi le schéma déprécié, les refus avant mise en queue et les workflows 0.4/1. Le smoke à couleurs uniformes vérifie la route applicative ; les coefficients bilinéaires sont vérifiés par le corpus numérique distinct.

Les 23 fichiers de source, tests, ressources et réglages capturés avant le build sont identiques après les tests. Les traces, résultats et empreintes sont associés à cette campagne ; il ne s'agit pas d'une attestation des DLL chargées par les processus .NET.

## Limites

Le [contrat ImageBatch](../IMAGE_BATCH.md) reste partiel : CPU/Float32 NHWC RGB/RGBA, plafond configurable de payloads, annulation aux frontières natives et preview textuelle. Cette campagne n'ajoute ni codec, GPU, autre dtype, poids préentraîné, nouvelle famille de modèles ni qualification Linux/macOS d'ImageBatch. Les contrôles numériques SD/CLIP déjà ouverts restent inchangés. Python sert uniquement au laboratoire de référence séparé ; l'application et les tests distribués sont C#/.NET.
