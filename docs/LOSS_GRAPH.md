# Graphe des pertes d'entraînement

`LossGraphNode` est enregistré dans le Host. Il reçoit un `LOSS_MAP`, dessine
un aperçu RGB de 840 × 520 et publie un PNG temporaire, avec les métadonnées
du prompt et du workflow lorsque leur sauvegarde est activée. Le registre
contient maintenant 51 types ; le catalogue complet reste à porter.

Le port suit [le nœud source figé](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_extras/nodes_train.py#L1408) :

- Abscisse calculée avec le nombre de pertes, sans étirer le dernier point
  jusqu'au bord droit ; ordonnée normalisée entre le minimum et le maximum.
- Axes, libellés, extrema à deux décimales et courbe bleue.
- Erreur pour une séquence vide ou constante, comme dans la source.
- `filename_prefix` conservé dans le schéma mais inutilisé par ce nœud source.
- Aucun slot de sortie ; résultat UI `images` et `animated: [false]`.

Le rendu utilise Skia natif. La géométrie et les chaînes sont comparées aux
commandes de dessin collectées en exécutant la classe source. Les pixels des
traits et des caractères ne sont pas déclarés identiques à Pillow : le moteur
de rasterisation et les polices de repli peuvent différer selon la plateforme.
Le laboratoire utilise des substituts pour les appels de dessin, de police
et de prévisualisation ; il ne fournit pas une image Pillow de référence.

## Contexte caché V3

Les champs déclarés `PROMPT`, `EXTRA_PNGINFO` et `UNIQUE_ID` sont transmis dans
`RuntimeNodeContext.Hidden`, séparément des arguments d'exécution et des listes.
Le contexte est renouvelé à chaque invocation. Les champs non déclarés ne sont
pas exposés. La projection `/object_info` conserve les noms et tableaux V3.
`DYNPROMPT` et les autres champs cachés restent explicitement non pris en charge.
Cette évolution ne constitue pas une implémentation complète du SDK V3.

## Vérifications et limites

Huit cas figés couvrent les coordonnées, textes et erreurs ; deux tests
contrôlent le transfert d'un `LOSS_MAP`, les métadonnées, le fichier temporaire
et l'annulation. Un test HTTP soumet le graphe, attend l'historique et relit
le PNG via `/view`. Trois tests supplémentaires contrôlent le contexte V3.
La [preuve structurée](qualification/loss-graph.json) distingue ces résultats
des suites générales et de leurs échecs.

Un aperçu a aussi été produit depuis les deux pertes du
[véritable essai SD1.5 de reprise LoHa](qualification/loha-resume.json),
0,83137566 et 0,7629695. Le producteur de pertes utilisé pour cette vérification
est réservé au test et lit le rapport existant. Aucun poids n'a été téléchargé,
copié ou réécrit. `TrainLoraNode` public, le workflow d'entraînement complet,
les fenêtres réelles et la qualification multiplateforme restent ouverts.
