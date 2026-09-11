# Premiers diagnostics U-Net aux dimensions complètes

Au commit `2ee03aa03599d52891048420c68043876774c8db`, le RuntimeProbe a exécuté
les U-Net SD1.5 et SD2 avec leurs largeurs complètes, des paramètres synthétiques
et un latent de seulement `16×16`. Les deux processus Windows CPU/F32 terminent
avec succès et donnent chacun trois hashes de sortie identiques.

Il s'agit d'un diagnostic d'exécution. Aucune comparaison numérique avec le
modèle source aux dimensions complètes, aucun checkpoint préentraîné, aucune
génération d'image et aucun GPU ne sont validés par ces résultats.

## Mesures locales

Le [relevé détaillé](sd-stock-diagnostic-2ee03aa.json) contient les configurations,
les 1 372 identités de paramètres, les hashes des entrées/sorties, les temps et
les observations mémoire. Les deux processus ont été exécutés successivement
après constat d'environ 42 Gio de RAM libre, avec un budget explicite de 6 Gio
par processus et un thread intra/inter-op.

| Mesure | SD1.5 | SD2 |
|---|---:|---:|
| Paramètres nommés | 686 | 686 |
| Valeurs de poids F32 | 859 520 964 | 865 910 724 |
| Temps de remplissage et hash | 3,85 s | 3,94 s |
| Temps par forward, trois répétitions | 1,37–1,48 s | 1,39–1,59 s |
| Pic de working set rapporté | 3,53 Gio | 3,54 Gio |
| Sortie | `[1,4,16,16]` | `[1,4,16,16]` |
| Hashes de sortie identiques | 3 / 3 | 3 / 3 |

Les contextes sont respectivement `[1,77,768]` et `[1,77,1024]`, avec un temps
explicite de `0.125`. Les poids sont remplis directement en mémoire native par
chunks d'au plus 1 Mio ; aucun poids de modèle n'est téléchargé ou écrit.

Un audit indépendant a recalculé les 1 372 hashes de paramètres avec la recette
nommée, sans importer Torch ni allouer de banque de poids, avec seulement
3 Mio de buffers réutilisés. Les huit fichiers d'entrée/sortie ont été rehashés,
leurs dimensions vérifiées et toutes leurs valeurs constatées finies. Les
entrées régénérées sont exactes. Cette vérification des paramètres synthétiques
ne compare pas les sorties du graphe à un oracle source.

Le working set après `Dispose` reste supérieur à son niveau initial : environ
2,20 Gio pour SD1.5 et 2,28 Gio pour SD2. L'origine de cette rétention n'est pas
établie par ce diagnostic. Trois répétitions et une seule banque par processus
ne démontrent pas l'absence de fuite, ni le comportement de chargements et
déchargements successifs. Les pics observés ne sont pas une garantie mémoire.

Le modèle de CPU/.NET est relevé séparément sur la même machine dans le JSON.
Les modules natifs effectivement chargés n'ont pas été capturés dans ces deux
processus stock ; le package libtorch y est seulement déclaré. Ces observations
ne sont pas des preuves de provenance native exclusive.

## CI normale associée

La [CI du même commit](https://github.com/skyline624/ComfySharp/actions/runs/34643684123)
et son [relevé audité](sd-components-2ee03aa.json) apportent des preuves distinctes
de ce diagnostic local. Douze artifacts et 35 TRX ont été vérifiés, sans résultat
individuel ignoré.

| Cible | Premier accès natif | Tests uniques, avec CLIP stock |
|---|---|---|
| Windows x64 | 182 réussis | 1 280 réussis |
| Linux x64 | 174 réussis, 8 échecs | Suite non exécutée |
| macOS arm64 | 182 réussis | 1 278 réussis, 2 échecs CLIP antérieurs |

Les 35 nouveaux tests comprennent 32 tests du diagnostic et de son générateur,
plus trois tests d'observation du runtime. Ils ne sont pas 35 comparaisons de
modèles avec une source. Les diagnostics CLI, le démarrage Desktop/Host et les
probes CPU passent sur Windows et macOS ; les six contrôles suivants sont non
exécutés sur Linux après l'échec du premier accès.

Les traces U-Net sont exactes sur Windows et macOS. Linux conserve huit échecs
U-Net/CFG et 42 valeurs de sortie hors borne. La nouvelle observation CPU
identifie un AMD EPYC 7763 pour Windows et un AMD EPYC 9V74 pour Linux ; leur
capacité ATen effective reste inconnue. Les inventaires de modules contiennent
quatre bibliothèques sur Windows et cinq sur Linux ; l'inventaire macOS est
explicitement indisponible. Les anciens échecs Windows sans ces observations
ne permettent pas d'attribuer son retour au succès à un changement de CPU.

Les références et tolérances existantes restent inchangées. La comparaison
source aux dimensions complètes, les ressources sur la durée, les vrais poids
et les workflows restent des étapes ouvertes de la migration complète.
