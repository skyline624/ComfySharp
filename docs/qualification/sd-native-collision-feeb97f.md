# SD CPU : collision de bibliothèques et variation de la CI

Le [laboratoire de provenance native](https://github.com/skyline624/ComfySharp/actions/runs/34642167261)
du commit `feeb97f3a126f1bf5295cd76a9ce07054cdb615b` a lancé ses six processus.
La précondition d'environnement corrigée passe. Les quatre processus source et
NuGet terminent ; les deux processus demandant les bibliothèques de la wheel
plantent au chargement, avant toute capture de tenseur.

Le message natif signale un double enregistrement du backend `Conjugate`. La
pile contient le libtorch CPU préchargé et un second libtorch CPU provenant du
chemin de sondage NuGet. Aucun inventaire des modules chargés n'a pu être capturé
dans ces deux processus. Leur origine exclusive reste donc **invalide pour une
comparaison numérique**. Il ne s'agit pas d'un écart numérique observé avec la
wheel. Les deux TRX correspondants contiennent un échec du testhost et zéro
résultat de test, sans succès implicite.

Le [relevé de l'expérience](sd-native-collision-feeb97f.json) conserve les phases,
hashes et mesures. Les trois artifacts ont été téléchargés ; dix hashes de
scripts ont été vérifiés contre les blobs du commit exécuté. Les 184 payloads
F32 disponibles ont été rehashés indépendamment, avec 129 comparaisons
recalculées. Un second audit confirme les 86 comparaisons source/NuGet dans les
mêmes modes et l'identité des modules NuGet observés. Le build original et les
références acceptées sont inchangés.

## Contrôles qui ont réellement produit des résultats

L'hôte est un AMD EPYC 7763. La source utilise PyTorch 2.10.0 CPU, Python 3.12.10,
et un thread intra/inter-op ; ses capacités effectives sont AVX2 et DEFAULT selon
le processus. Le produit utilise également un thread intra/inter-op. Ses deux
inventaires montrent exactement les trois bibliothèques cœur NuGet attendues,
le bridge TorchSharp inchangé et une bibliothèque OpenMP attendue.

| Comparaison sur ce même hôte | Captures exactes | Différence absolue maximale de sortie | Valeurs de sortie hors borne |
|---|---:|---:|---:|
| NuGet auto / source auto | 16 / 43 | `4.437565803527832e-5` | 2 / 256 |
| NuGet DEFAULT / source DEFAULT | 43 / 43 | 0 | 0 |

Les 686 identités de paramètres, leurs layouts et les entrées concordent entre
les quatre processus disponibles. Les séquences avec observateur désactivé,
activé puis désactivé donnent chacune trois hashes de sortie identiques.
La première différence en mode automatique reste la GroupNorm
`input_blocks.10.0.in_layers.0.output`. Aucun résultat de ce tableau n'appartient
à une variante wheel du produit.

Les deux tests NuGet contre la référence acceptée passent ici. Leur oracle est
distinct de la source exécutée dans ce diagnostic : ce succès n'annule pas les
deux dépassements mesurés dans le tableau. Le profil et les sorties de référence
restent inchangés ; aucune famille n'est qualifiée.

## CI normale du même commit

La [campagne normale](https://github.com/skyline624/ComfySharp/actions/runs/34642167344)
et son [relevé indépendant](sd-components-feeb97f.json) confirment neuf artifacts
et 28 TRX, sans résultat ignoré. Le commit contient toujours 1 245 tests, avec
les mêmes sources C#, dépendances, fixtures et workflow normal que `b3d380f`.

| Cible | Premier accès natif | Suite complète avec comparaisons CLIP |
|---|---|---|
| Windows x64 | 172 réussis, 10 échecs | Non exécutée |
| Linux x64 | 175 réussis, 7 échecs | Non exécutée |
| macOS arm64 | 182 réussis | 1 243 réussis, 2 échecs CLIP antérieurs |

Les 80 hashes Windows sont identiques à la campagne précédente, avec 68 valeurs
de sortie hors borne. Les 80 hashes macOS sont également inchangés et exacts par
rapport à leur source. Sous Linux, aucun des 80 hashes n'est identique à la
campagne précédente ; 42 valeurs de sortie dépassent la borne. Les étapes
agrégées, CLI, Desktop/Host et probe suivantes n'ont pas été exécutées sur
Windows/Linux. Le diagnostic de provenance et la CI normale sont deux
expériences distinctes, dont on ne peut pas fusionner les identités matérielles.

La prochaine expérience isole les bibliothèques dans des copies du dossier de
test, avec un contrôle de copie conservant NuGet, les mêmes assemblies C# et les
mêmes conditions d'invocation. Elle conserve la vérification stricte des modules
réellement chargés. Une copie ou une variable d'environnement ne constitue pas
à elle seule une preuve de provenance ni une adoption du runtime dans le produit.
