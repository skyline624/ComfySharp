# Entraînement : provenance native et ressources de tests

Révision étudiée : `c5b6a5b02c82011037bd49a7a219a5ac966c0295`.
Les références et le seuil `3e-5 + 3e-5 × abs(référence)` restent inchangés.

## Écarts numériques encore ouverts

La [capture Linux détaillée](https://github.com/skyline624/ComfySharp/actions/runs/34714010303)
conserve les échecs des deux cas d'entraînement, avec et sans observation,
y compris avec un seul thread inter-op. Les douze tests de fichiers mixtes et
d'alias passent indépendamment.

Les 686 poids de base correspondent à la source. La première différence observée
apparaît dans la sortie d'une GroupNorm :

| Cas | Frontière à la première mise à jour | Erreur absolue maximale à cette frontière |
| --- | --- | --- |
| SD1 réduit | `input_blocks.10.0.in_layers.0.output` | `2.384185791015625e-7` |
| SD2 réduit | `input_blocks.11.0.in_layers.0.output` | `1.1920928955078125e-7` |

Ces erreurs élémentaires sont sous le seuil ; des écarts s'accumulent ensuite.
Les captures s'arrêtent à la première assertion en échec : elles ne prouvent pas
les étapes ultérieures. Les strides du collecteur source décrivent sa copie
contiguë de diagnostic, pas nécessairement le tenseur original.

Un second problème concerne l'oracle : l'ancienne collecte Linux annonce AVX512,
la collecte du runner courant AVX2. Le code source inchangé dépasse lui-même le
seuil de l'ancien oracle sur six sorties/gradients des deux étapes. Le profil doit
donc identifier la provenance native et le dispatch CPU ; le seul RID ne suffit
pas. Cela n'explique pas à lui seul l'écart C#/source sur le même runner.

Les [copies natives déjà contrôlées](sd-native-aligned-996f8a3.md) avaient obtenu
86 captures C# exactes avec les bibliothèques natives de la source sur le même hôte.
La suite du travail doit qualifier ce bundle pour l'entraînement et les autres
opérations, puis formaliser les profils de référence. Répéter la localisation de
GroupNorm ou augmenter les tolérances ne résout pas ces deux problèmes.

La [CI normale de c5b6a5b](https://github.com/skyline624/ComfySharp/actions/runs/34714010279)
échoue sur les deux comparaisons d'entraînement Windows et Linux. macOS atteint
la suite générale : 864 tests passent, un contrôle de compteur échoue. Le test
CLIP aux dimensions complètes n'est pas atteint dans cette campagne.

## Fuite identifiée dans les tests

Une observation locale avant/après chaque test, en exécution séquentielle
temporaire, reproduit exactement l'échec `expected 2 / actual 1` du test
`Mixed_bias_norm_and_matrix_adapters_preserve_the_base_and_receive_gradients`.
Le test d'affichage `DenseFormatterPreservesLinebreaksDtypeSignedZeroAndLimits`
laisse auparavant son compteur passer de 1 à 2 :
`tensor(...).reshape(...)` ne libérait que la vue finale.

Le stockage intermédiaire est maintenant possédé par un `using` distinct.
Le contrôle suivant passe les 865 tests. La même observation identifie également
trois expressions chaînées dans deux tests de conditioning CLIP : deux vues
intermédiaires de couches et une vue de pooling par profil étaient oubliées.
Elles reçoivent également un propriétaire explicite.

Après les trois corrections, les 865 tests passent en exécution séquentielle
observée (41 s), puis les 865 passent dans la configuration parallèle normale,
sans observateur (37 s). Ces campagnes locales excluent le processus séparé
`ClipStockReferenceTests`, comme la suite ordinaire de CI. L'incrément isolé
de compteur dans `NativeGradientOptimizerAndViewLifetime` n'est pas attribué
par cette enquête et ne permet pas d'affirmer que le compteur global vaut zéro.

Aucune assertion, référence numérique ou fonction du produit n'est modifiée.
La sérialisation des tests et l'observateur restent des instruments locaux,
absents de la configuration normale. La correction ne qualifie pas la précision
multiplateforme et ne constitue pas une mesure exhaustive de fuite de VRAM.
