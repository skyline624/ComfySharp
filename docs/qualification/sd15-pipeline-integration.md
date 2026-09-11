# SD1.5 réduit — observation d'intégration Windows CPU/F32

Les quatre tests de référence passent dans un processus Windows local. **Les 64 captures source sont bit à bit identiques**, ainsi que les 36 comparaisons de sorties finales répétées. Les 32 frontières écrites auparavant par la vraie CLI sont également identiques aux captures des tests. Les preuves et empreintes sont dans [le JSON associé](sd15-pipeline-integration.json).

La référence indépendante vient de la [collecte source 34656804132](https://github.com/skyline624/ComfySharp/actions/runs/34656804132), au commit `6ad8207ffe46d3d1e57780b9fc4c7288b686638d`. Le test sélectionne uniquement sa fixture Windows, avec les pins du pipeline et du manifeste vérifiés avant lecture. Le [profil prospectif](sd15-pipeline-native210-cpu-f32-v1.md) conserve la borne `3e-5 + 3e-5*abs(source)` ; les scalaires sigma/sigmaHat, leurs formes et indices exigent une égalité exacte.

| Observation | Records | Éléments F32 comparés | Records bit exacts | Hors borne |
|---|---:|---:|---:|---:|
| Captures distinctes : 8 frontières par cas + 4 records par pas Euler | 64 | 28832 | 64 | 0 |
| Trois sorties finales × 3 passages × 4 cas | 36 | 48000 | 36 | 0 |
| Frontières des quatre processus CLI précédents contre tests | 32 | 27536 | 32 | 0 |

L'écart absolu maximal est zéro dans les trois catégories. Les 32 frontières CLI sont déjà comprises dans les 64 frontières/états du corpus ; les 36 comparaisons finales répètent douze sorties sémantiques. Ces lignes ne doivent pas être additionnées pour annoncer une couverture unique plus grande.

Les quatre cas vérifient chacun les mêmes 971 paramètres CLIP/U-Net/VAE contre la source avant et après calcul, ainsi que les entrées bruit/sigmas et les tokens gelés. Les trois passages off/on/off conservent les hashes finaux. Les captures sont sous no-grad et l'état grad de l'appelant est rétabli. Les snapshots sérialisent les valeurs F32 sans modifier la disposition employée par le graphe ; les payloads ont été rehashés avant comparaison. Le zéro interne n'est pas présenté comme un emprunt directement instrumenté.

Le test tracé fournit quatre résultats Passed et quatre fichiers de captures complets. Un premier processus de quatre tests avait également réussi selon root, mais son option de traces n'avait pas été positionnée : il n'apporte aucune preuve de capture supplémentaire. La suite complète ultérieure passe **1363 tests distincts, dont 577 Inference, sans échec ni test non exécuté**. Ses six TRX ont été rehashés et recomptés ; les quatre tests ciblés ne sont pas ajoutés à ce total. Le fichier de référence est resté au SHA gelé `ca62c3ab17b1b95a024618f8630d7e7534401b05e16a47fb27473ff0cff91127`. La [preuve de composition précédente](sd15-pipeline-composition.json) décrit séparément la suite 1351 antérieure au patch diagnostic d'alignement. Les TRX ne constituent pas à eux seuls une attestation de tous les fichiers sources du checkout.

Le produit observé utilise .NET 10.0.10, TorchSharp 0.107.0.0, le package libtorch déclaré 2.10.0 et intra/inter-op1. Les maps attestent les SHA des bibliothèques réellement chargées. La source et le produit emploient des binaires torch_cpu distincts et ont tourné sur des machines différentes. La source rapporte AVX2 ; le dispatch ATen effectif du produit reste inconnu. Les capacités intrinsèques .NET ne le remplacent pas.

Cette preuve concerne seulement les quatre cas réduits synthétiques Windows décrits dans [SD15_PIPELINE](../SD15_PIPELINE.md). Elle n'établit ni acceptation Linux/macOS, ni largeur stock, poids préentraînés, GPU, nœud de génération ou disponibilité V1. Aucun résultat C# n'a servi d'oracle ; aucune borne ou fixture n'a été ajustée.
