# Chargement des géométries LoKr en mode bypass

Le chargeur distingue maintenant `LoraLoadMode.Weights` (par défaut) et
`LoraLoadMode.Bypass`. Pour LoKr, le second valide les dimensions des opérateurs
de `LoKrAdapter.h` au lieu d'exiger que les facteurs reconstruisent un poids
de même taille que la cible. Le mode est conservé dans le plan ; un plan inspecté
pour le bypass ne peut pas être utilisé pour modifier des poids. Il faut refaire
l'inspection dans le mode ordinaire pour cette opération.

Cette distinction est nécessaire pour les facteurs spatiaux, les chaînes Tucker
et les additions avec diffusion des dimensions de taille 1. Les règles viennent
du [LoKr figé](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/weight_adapter/lokr.py)
et de son [injection dans la couche](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/weight_adapter/bypass.py).

## Comportements portés

- Couches linéaires et convolutions 1D, 2D et 3D : canaux d'entrée, groupes et
  compatibilité des canaux intermédiaires vérifiés à l'inspection.
- Facteurs secondaires spatiaux, chaînes de plusieurs convolutions et cœurs
  Tucker matriciels/spatiaux, selon les opérateurs réellement utilisés par `h`.
- Priorité aux côtés directs ; les facteurs inactifs restent conservés.
- Cœur Tucker ignoré dans le bypass linéaire, conformément à la source.
- Échelle DoRA conservée mais non utilisée par le bypass LoKr. Son ancien contrôle
  de diffusion vers le poids reste appliqué en mode ordinaire.
- Addition native avec diffusion des dimensions de taille 1. Les formes de sortie
  peuvent donc différer de celles de la sortie de base lorsque la source le permet.

La validation spatiale finale dépend des dimensions d'entrée, du stride et du
padding. Elle reste dans les opérateurs et l'addition natifs : une incompatibilité
donne une erreur explicite. Le chargeur ne promet pas qu'une chaîne soit valide
pour toutes les tailles d'entrée simplement parce que ses canaux sont compatibles.

`LoraAdapterSet.ApplyBypass` applique un adapter chargé à une entrée et une sortie
de base fournies par l'appelant. `ApplyBypassTo` conserve l'intégration du U-Net.
Le nœud public `LoraModelLoader` sélectionne le mode d'inspection avec son entrée
`bypass`. Les diagnostics d'entraînement font de même pour leur rechargement.

## Preuves

Le collecteur indépendant `labs/lora-training-source/lokr_bypass_loading.py`
exécute les définitions figées et l'addition source. Sa fixture de 7 189 octets
compressés contient 22 scénarios : 15 calculs réussis et sept erreurs source.
Les tests chargent les mêmes facteurs depuis safetensors et depuis la mémoire,
comparent les résultats, contrôlent leur durée de vie et les budgets de facteurs.
Les fichiers de tests sont petits, synthétiques, temporaires et supprimés ensuite.
Aucun poids préentraîné n'est inclus dans les références.

Onze autres cas vérifient que le chargement ordinaire continue de refuser les
géométries qui ne peuvent pas reconstruire le poids cible. Leurs données excluent
l'échelle DoRA ignorée pour vérifier que le refus vient bien de la géométrie.
Un test du nœud public applique une chaîne Tucker à un U-Net réduit, compare son
résultat au chemin natif direct, libère les données d'origine et vérifie que la
base est inchangée. Le budget de poids reconstruits est nul.

Le diagnostic CPU sur le SD1.5 partagé relit les mêmes poids et utilise cette
inspection pour le rechargement en mémoire. Les résultats et hashes sont publiés
dans [la preuve](qualification/lokr-bypass-loading.json). Ce test reste une
prédiction brute sur tenseurs réduits, sans qualification de workflow sur images.

## Limites restantes

Cette qualification de géométrie concerne LoKr. Les branches LoRA et LoHa du
chargeur conservent leurs contrôles existants ; leur couverture des géométries
propres au bypass doit être examinée séparément. Les précisions mixtes, autres
familles de modèles, entraînement complet sur images et GPU restent à qualifier.
Les tolérances des références restent `atol=rtol=3e-5`, sans modification.
