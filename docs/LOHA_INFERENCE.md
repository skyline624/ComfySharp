# Chargement et réutilisation LoHa

Le chargeur de fichiers safetensors et de tenseurs natifs accepte les quatre
matrices `hada_w1_a/b`, `hada_w2_a/b`, les deux cores Tucker 2D optionnels,
alpha et DoRA. Les calculs utilisent Float32 ; les facteurs sauvegardés F16 et
BF16 sont convertis explicitement. Les snapshots possèdent leurs storages,
survivent au fichier et aux paramètres d'entraînement, puis les libèrent sans
attendre le GC. Le budget tient compte des quatre matrices et des cores.

Les contrats adaptés proviennent du [LoHaAdapter figé](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/weight_adapter/loha.py)
et du [chargeur commun](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/lora.py).
La boucle des providers continue après chaque correspondance : **LoHa remplace
LoRA** pour un même préfixe contenant les deux formats. Un alias ultérieur
remplace le précédent ; les différences de normalisation puis les différences
explicites sont appliquées après les adapters. Les clés des formats reconnus
restent comptées comme consommées même si leur résultat est remplacé.

`LohaMath` reproduit le produit de Hadamard, le reshape de destination et le
broadcast des deux reconstructions. DoRA conserve l'asymétrie de normalisation
de la source : poids original sur l'axe de sortie, poids modifié sur l'autre axe.
Le bypass d'inférence linear/Conv2d reconstruit et met à l'échelle la différence
avant l'opération. Il ignore DoRA comme `LoHaAdapter.h` et conserve le gradient
des activations. Les hooks SD réutilisent ces snapshots après clone/transfert.
Le bypass **d'entraînement** LoHa demeure refusé : `LohaDiff.h` n'existe pas
dans la référence. La [reprise des facteurs entraînables](LOHA_RESUME.md) utilise
un contrat séparé, avec priorité LoRA et règles alpha propres à la fabrique.

Le laboratoire séparé `labs/lora-training-source/loha_inference.py` exécute les
définitions AST du commit figé. Seule la conversion vers le device est adaptée
au laboratoire CPU. Il produit 48 cas de calcul et quatre cas de priorité du
chargeur. Les tests C# embarquent ces références statiques, avec hash verrouillé
et `atol=rtol=3e-5` pour les poids, sorties et gradients. Ils n'exécutent pas Python.
Les tests couvrent aussi budgets, facteurs manquants/non finis, cores orphelins,
annulation, indépendance des snapshots et libération des tenseurs.

Sur le checkpoint SD1.5 identifié dans [la preuve](qualification/loha-inference.json),
deux mises à jour SGD produisent un adapter de **18 946 320 octets**, avec
**686 cibles et 1 814 tenseurs**. Le rechargement en mémoire et depuis le fichier
reproduit exactement la prédiction entraînée. Un décodeur indépendant vérifie
tous les hashes des tenseurs ; le chargeur ComfyUI accepte les 282 adapters LoHa,
109 différences de poids et 295 différences de biais. Le checkpoint partagé
n'est ni recopié ni téléchargé. Seul l'adapter issu du test est sauvegardé.

```text
dotnet run -c Release --project tools/ComfySharp.RuntimeProbe -- sd-all-adapter-train --algorithm LoHa --checkpoint <existant.safetensors> --report <nouveau.json> --device cpu --adapter <nouvel-adapter.safetensors>
```

Ce parcours utilise des entrées synthétiques réduites et de vrais poids sur
Windows CPU. Il ne prouve pas la qualité d'entraînement sur images, les gradients
préentraînés contre ComfyUI, les convolutions 1D/3D du bypass, la précision mixte,
les autres familles ni CUDA/MPS. Le nœud public `TrainLoraNode` complet et les
qualifications des plateformes restent requis avant la V1.
