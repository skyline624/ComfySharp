# Composants SD : comparaison CPU sur un même hôte Linux

Au commit `76c726023ab5f0f5868eb3439d895fc79cf53c2c`, le port et la source
ComfyUI correspondent bit pour bit en mode demandé `DEFAULT` pour les huit
sorties U-Net, leurs 72 frontières internes et les six sorties CFG. En AVX2,
des écarts persistent. **Cette expérience ne valide pas les composants** : les
tests contre les références acceptées continuent d'échouer, y compris en
`DEFAULT`. Aucun seuil, profil ou fichier de référence n'a changé.

Le [diagnostic isolé](https://github.com/skyline624/ComfySharp/actions/runs/34637495790)
et la [CI normale](https://github.com/skyline624/ComfySharp/actions/runs/34637495740)
testent le même commit. Leurs résultats sont consignés séparément dans le
[relevé diagnostique](sd-runtime-diagnostic-76c7260.json) et le
[relevé des tests](sd-components-76c7260.json).

## Comparaison contrôlée

Le diagnostic exécute 31 processus indépendants sur un AMD EPYC 7763 : trois
sources, douze suites produit et seize suites U-Net avec différents décalages
du latent. Chaque processus établit son environnement avant le chargement
natif. Tous terminent ; neuf réussissent et 22 échouent. Les 28 rapports TRX,
170 traces et cinq artifacts sont présents. Un contrôle indépendant a vérifié
2 171 payloads de tenseurs et recalculé 2 054 comparaisons produit/source ainsi
que 633 comparaisons source/source, sans erreur d'intégrité.

Le CPU ne possède pas d'instructions AVX512. La source rapporte AVX2 en mode
automatique comme avec l'option `avx2`, et DEFAULT avec l'option `default`.
Les 211 enregistrements de tenseurs source auto/AVX2 sont identiques. Les
sorties et frontières produit auto/AVX2 sont également identiques entre elles.
La capacité effective du produit n'est pas présentée comme une mesure :
l'API publique TorchSharp utilisée n'expose pas ce contrôle.

Le tableau indique les **éléments de sortie U-Net / CFG hors tolérance**.
Chaque cellule compare huit sorties U-Net, 72 frontières et six sorties CFG.
Toutes les frontières internes respectent la borne dans les neuf comparaisons.

| Mode demandé du produit | Source auto | Source avx2 | Source default |
|---|---:|---:|---:|
| auto | 2 / 108 | 2 / 108 | 28 / 191 |
| avx2 | 2 / 108 | 2 / 108 | 28 / 191 |
| default | 49 / 176 | 49 / 176 | 0 / 0 ; 86 tenseurs bit pour bit identiques |

Les deux valeurs U-Net hors borne face à la source AVX2 du même hôte
appartiennent au cas `sd15-reduced/square` ; l'erreur maximale vaut
`4.437565803527832e-5`. Les premières frontières différentes des deux cas
carrés SD1/SD2 sont `down3`, avec une différence maximale de
`5.960464477539063e-8`. Leur embedding temporel et leurs frontières précédentes
sont identiques. L'intervalle comprend un sous-échantillonnage et deux blocs
résiduels : ces traces ne désignent pas encore une primitive responsable.

Les seize décalages natifs de zéro à quinze éléments conservent les mêmes
valeurs, dimensions et strides d'entrée ; seul zéro est aligné sur 64 octets.
Les huit sorties et 72 frontières de chaque exécution restent identiques au
produit en mode automatique. Ce contrôle n'appuie donc pas une correction par
copie du latent. Il ne couvre pas tous les tenseurs, allocateurs ou matériels.

## Identités natives et limites d'interprétation

Les onze bibliothèques de la source sont identiques à celles de la campagne
source acceptée, qui s'exécutait sur un autre CPU et rapportait AVX512. Les
paquets source et produit annoncent tous deux PyTorch/libtorch 2.10, mais leurs
binaires diffèrent :

| `libtorch_cpu.so` | Taille | SHA-256 |
|---|---:|---|
| Source Python | 443 509 856 octets | `4dc5b3a649b61d8f39be127866bc258dda19d74e339f449fe18103db63e8d76f` |
| Produit .NET | 461 211 481 octets | `0f4b3e14ed8468219fefe110605e741afc6bfbec27cc777a224301edb9f24c41` |

`libc10` et OpenMP diffèrent également. La dépendance numérique au mode demandé
est démontrée dans la source. La correspondance exacte en DEFAULT et les
différences en AVX2 orientent l'enquête vers des opérations sensibles au
dispatch, leur build natif ou leurs entrées. Elles ne prouvent pas une erreur
de compilateur, un kernel précis ou l'absence d'un défaut de layout du port.

Dans le [code PyTorch figé](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/DispatchStub.h#L450),
un enregistrement ordinaire de kernel ne sélectionne pas automatiquement une
variante AVX512. Le [sélecteur](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/DispatchStub.cpp#L351)
peut se replier sur AVX2. C'est notamment le cas des enregistrements de
[GroupNorm](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cpu/group_norm_kernel.cpp#L1519)
et [LayerNorm](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cpu/layer_norm_kernel.cpp#L614).
La capacité globale ne prouve donc pas la largeur vectorielle d'une opération.
Elle ne contrôle pas non plus tous les chemins internes BLAS/oneDNN.

## Résultats contre les références acceptées

Dans les trois modes du diagnostic, les tests existants passent pour le
sampling (3/3) et le VAE (8/8), mais échouent pour deux des huit cas U-Net et
les six cas CFG. Les échecs U-Net concernent `sd15-reduced/odd-rectangle` et
`sd2-reduced/batch-shared-time-odd`, différents du cas carré identifié dans
la comparaison avec la nouvelle source du même hôte. Chaque suite de décalage
du latent conserve ces deux échecs. La correspondance locale DEFAULT ne les
remplace pas.

Dans la CI normale, Windows passe 1 245 tests ; macOS en passe 1 243 avec les
deux échecs CLIP antérieurs. Les 93 nouveaux tests SD passent sur ces deux OS
et leurs 80 tenseurs U-Net sont exacts. Linux passe 174 contrôles de premier
accès et en échoue huit, puis n'exécute pas les étapes suivantes. Ses traces
contiennent 42 valeurs hors borne en sortie, aucune aux frontières internes.
Le [contrôle précédent](sd-cpu-investigation-ca41c5f.md) en comptait quatre,
avec cinq tests échoués. Les sources produit, fixtures, dépendances et profil
sont inchangés entre ces deux campagnes ; cette variation ne démontre ni une
correction ni une cause de régression.

La prochaine expérience capture les primitives entre `down2` et `down3`,
avec leurs valeurs d'entrée, paramètres et layouts, puis vérifie que les
observateurs ne changent pas les sorties. Aucun modèle préentraîné, workflow,
graphe aux dimensions complètes ou backend GPU n'est qualifié par ces essais.
