# Comparaison des modes CPU Windows

Les captures des CI `34728064909` (réussite Windows) et `34728727984`
(échec Windows) identifient des hôtes différents avec les mêmes bibliothèques :

| Relevé | Réussite | Échec |
| --- | --- | --- |
| Processeur | AMD EPYC 9V74 | Intel Xeon Platinum 8573C |
| AVX-512 disponible pour .NET | Non | Oui |
| Threads intra/inter-op observés | 1 / 1 | 1 / 1 |
| Mode ATen demandé | Aucun | Aucun |
| Mode ATen réellement observé dans ces anciennes captures | Indisponible | Indisponible |

Les quatre images natives ont les mêmes SHA-256 sur les deux runners :

| Bibliothèque | SHA-256 |
| --- | --- |
| torch_cpu.dll | `c75772dea925c23ea45fbf39502d48d4e87350249e75050d43bea2c12fdd3a1b` |
| c10.dll | `a057ab00ec2edceec9198e87bd417843489f9bb4b937ff2cf22503e699383063` |
| libiomp5md.dll | `d41a71c38f627a95748596820ab380135dcccc4ceecd6fe7d4e247d015bf55a4` |
| LibTorchSharp.dll | `8552d2fd6d045a5a9fd2315300f6f6aac7770433beeb4a743e22d59d77ed1858` |

Ces relevés viennent de `sd15-reduced--square.json` dans les artefacts
`sd-component-traces-win-x64` des deux campagnes. Ils montrent une différence
de matériel et écartent un changement de ces quatre binaires. Ils ne suffisent
pas à attribuer la cause des écarts au dispatch CPU.

## Expérience sur un même hôte

Le workflow `windows-runtime.yml`, déclenché par ses propres changements ou
manuellement, construit le bridge d'identité depuis
les headers libtorch 2.10 du laboratoire CPU épinglé. Chaque profil lance des
processus source et .NET indépendants. Aucun poids préentraîné n'est nécessaire.

| Profil | Contrôles demandés uniquement pour ces processus |
| --- | --- |
| `auto` | Aucun contrôle de dispatch |
| `aten-avx2` | `ATEN_CPU_CAPABILITY=avx2` |
| `math-avx2` | Même contrôle ATen, plus `MKL_ENABLE_INSTRUCTIONS=AVX2`, `ONEDNN_MAX_CPU_ISA=AVX2`, `DNNL_MAX_CPU_ISA=AVX2` |

Le script ne modifie pas l'environnement de l'application distribuée. La lecture
native établit le mode ATen effectif et le build chargé ; les variables MKL et
oneDNN restent des contrôles demandés, sans prétendre observer chaque kernel.

L'oracle est produit par le collecteur source d'entraînement inchangé, dont les
hashes et les fichiers amont sont vérifiés avant exécution. Deux étapes sur les
topologies SD1/SD2 réduites donnent les poids, frontières intermédiaires, pertes,
gradients et paramètres mis à jour. La capture complète conserve les échecs
contre les fixtures historiques, puis compare séparément la source du même hôte
aux seuils originaux. La suite ordinaire s'exécute ensuite sans observation.

Les résultats gardent le matériel, le mode ATen, les bibliothèques chargées, les
verdicts TRX et les différences par capture. Un test échoué, non exécuté ou une
comparaison source hors tolérance fait échouer l'expérience. Les fixtures et
les tolérances distribuées restent inchangées. Aucune sélection automatique de
profil, adoption de bundle, qualification de famille ou de plateforme n'est
déduite de la seule exécution du diagnostic.

## Contrôle local du dispositif

Les trois profils locaux terminent avec trois tests d'entraînement et les
1 038 tests ordinaires réussis chacun, sans test ignoré. Chaque profil reproduit
exactement les 282 captures source et les 686 poids de base des deux topologies.
Les builds et le mode ATen `AVX2` correspondent entre la source et .NET.
La [preuve structurée](windows-runtime-controls.json) conserve les identités et
les hashes du collecteur exécuté. Des gardes supplémentaires sur le nombre de
tests et la cible Windows ont été ajoutées ensuite ; les artefacts locaux
satisfont aussi ces contrôles. La CI vérifiera le collecteur publié.

Ce résultat local ne reproduit pas l'échec observé sur le Xeon de la CI. Le
contrôle initial avait échoué avec un ancien bridge local sans exports
d'identité ; sa recompilation depuis le code actuel a permis la collecte.
Aucune bibliothèque libtorch ni assertion du produit n'a été changée.
