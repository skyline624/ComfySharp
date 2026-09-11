# Localisation de la première divergence U-Net sous Linux

La [capture ciblée](https://github.com/skyline624/ComfySharp/actions/runs/34639647077)
au commit `7549b1ced545e24b38b5f1b485c9b1489bd12d0e` localise la première
différence du cas carré SD1.5 à la sortie d'une GroupNorm. Son entrée, ses poids
et leurs layouts sont identiques. Cette observation distingue une opération
précise ; elle ne prouve pas encore la cause de sa différence numérique.

Les [mesures détaillées](sd-operator-diagnostic-7549b1c.json) conservent les
comparaisons et les hashes. Un contrôle indépendant a rehashé les 184 payloads
des quatre processus et recalculé les 215 comparaisons de tenseurs. Les
686 paramètres et leurs layouts concordent dans les quatre processus.

## Opération observée

Le même Intel Xeon Platinum 8370C exécute quatre processus séparés : source et
produit en mode automatique, puis source et produit en mode demandé DEFAULT.
La source rapporte respectivement AVX512 et DEFAULT, avec un thread intra-op
et un thread inter-op. La capacité demandée du produit est enregistrée sans
être présentée comme une capacité effective mesurée.

Le diagnostic observe 43 tenseurs réels : embedding temporel, frontière
`down2`, dernière convolution de sous-échantillonnage, primitives des deux
blocs résiduels suivants et sortie finale. Les quatre processus donnent chacun
la même sortie avec observateurs désactivés, activés, puis désactivés. Le
produit possède une seule implémentation du calcul résiduel ; les captures ne
recalculent aucune primitive.

En mode automatique, l'embedding temporel, `down2`, le sous-échantillonnage
suivant et l'entrée du bloc résiduel 10 sont exacts. La première différence
apparaît dans `input_blocks.10.0.in_layers.0`, avec cette configuration :

| Élément | Mesure |
|---|---|
| Normalisation | GroupNorm, 32 groupes, epsilon `1e-5`, CPU Float32 |
| Entrée | `[1,128,1,1]`, strides `[128,1,1,1]`, alignée sur 64 octets |
| Entrée et paramètres gamma/bêta | Identiques bit pour bit, strides et alignements identiques |
| Sortie | 6 valeurs différentes sur 128 |
| Différence absolue maximale | `1.1920928955078125e-7` |
| Première valeur différente, index 64 | Produit `0.4833134710788727` ; source `0.4833134412765503` |

Les alignements des entrées initiales du modèle ne sont pas tous identiques
entre Python et .NET ; la conclusion ci-dessus porte sur **l'entrée réelle de
la GroupNorm**, capturée après les opérations précédentes. Ses valeurs et son
layout sont exacts. Les références amont, les équations, les dépendances et les
seuils acceptés restent inchangés.

## Comparaisons et limites

| Produit / source | Tenseurs exacts sur 43 | Différence absolue maximale | Éléments hors borne |
|---|---:|---:|---:|
| auto / auto | 16 | `1.239776611328125e-5` | 0 |
| auto / default | 0 | `1.341104507446289e-5` | 0 |
| default / auto | 0 | `1.9490718841552734e-5` | 0 |
| default / default | 43 | 0 | 0 |

Les comparaisons entre modes différents divergent dès l'embedding temporel ;
elles n'isolent pas la GroupNorm. En mode automatique identique, les opérations
suivantes reçoivent parfois des valeurs déjà différentes : leurs écarts ne
prouvent pas chacun un défaut indépendant.

Les quatre processus et les deux tests du cas carré contre la référence
acceptée réussissent dans cette campagne. Cela ne contredit pas le diagnostic
précédent sur AMD EPYC : le matériel et le dispatch source diffèrent. Les
binaires natifs source et produit restent distincts. Une expérience séparée
comparera les mêmes assemblies C# avec les bibliothèques natives de chaque
distribution, en vérifiant les hashes réellement chargés. Aucun changement de
runtime produit n'est adopté sur la seule observation d'une GroupNorm.

La [CI normale associée](https://github.com/skyline624/ComfySharp/actions/runs/34639647079)
reste en échec. Son [relevé vérifié](sd-components-7549b1c.json) compte
1 245 tests réussis sur Windows, 1 243 réussis et deux échecs CLIP antérieurs
sur macOS. Les 93 tests SD passent sur ces deux OS. Linux passe 177 contrôles
de premier accès et en échoue cinq, puis n'exécute pas les étapes suivantes.
Ses 80 hashes de traces U-Net sont identiques à ceux de la campagne `ca41c5f` ;
quatre valeurs de sortie SD2 dépassent toujours la borne. Cette répétition ne
démontre pas une correction, et aucun résultat de famille, poids préentraîné,
workflow complet ou GPU n'est qualifié.
