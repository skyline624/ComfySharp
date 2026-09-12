# Reprise des adapters LoRA SD

`SdTrainableAdapterSet` accepte une source `existing` safetensors ou native pour
reprendre les facteurs LoRA des U-Net SD. Le résultat possède ses paramètres ;
le lecteur et les tenseurs d'entrée peuvent être détruits après la construction.
Le budget tient compte du rang de chaque adapter chargé et du rang demandé pour
les adapters créés. Les erreurs et l'annulation libèrent les allocations partielles.

Le port reproduit `_create_weight_adapter` et `_setup_lora_adapters` du backend
[figé](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_extras/nodes_train.py#L697).
Les sept écritures de facteurs suivent la priorité de `LoRAAdapter.load`, sous
les noms canoniques `diffusion_model.<module>`. Les alias généraux du chargeur
d'inférence ne sont pas utilisés par cette fabrique source.

Deux particularités source sont conservées et visibles dans `IgnoredExistingKeys` :

- Les différences de normalisation et de biais sont recréées à zéro, même si
  un export contient `.diff` ou `.diff_b`.
- Alpha est lu sous `<module>.weight.alpha`. La clé habituelle `<module>.alpha`
  n'est pas utilisée par cette fabrique : en son absence, alpha repart à 1.

Les métadonnées DoRA/reshape n'interviennent pas dans le constructeur source
à deux facteurs. Les facteurs intermédiaires LoCon et les algorithmes de reprise
LoHa, LoKr, GLoRA, OFT et BOFT restent à porter ; leurs inscriptions reconnues
reçoivent un diagnostic explicite. Les clés orphelines qui ne déclenchent aucun
chargeur source restent ignorées et signalées.

La fabrique source copie les poids chargés dans des couches CPU Float32 avant
leur transfert. Le port conserve cette conversion et les tirages aléatoires des
constructeurs, même si leurs valeurs initiales sont ensuite remplacées. Quatre
références SD1/SD2 vérifient exactement tous les paramètres et l'état du générateur,
avec facteurs F32/F16/BF16, rangs différents et priorités concurrentes.

`SdTrainingResumeSteps.Parse` reproduit séparément le compteur du nom sélectionné :
`[None]` donne zéro ; sinon la source prend le dernier fragment séparé par `_`
avant le premier `_steps_` et le convertit en entier. Treize références couvrent
noms valides/invalides, signes, chiffres Unicode et valeurs dépassant Int64.
Ce compteur n'est pas encore raccordé au nœud public d'entraînement.

Un diagnostic utilise directement le checkpoint SD1.5 et l'adapter déjà entraîné
du stockage partagé. Il restaure **282 cibles à facteurs** et recrée les différences,
effectue deux mises à jour sur **686 cibles/1 250 paramètres**, vérifie les gradients
finis et la base inchangée, puis recharge le snapshot en mémoire avec prédiction
exacte. Les entrées sont synthétiques (latent 8 × 8 et contexte brut) ; aucune
comparaison de gradients préentraînés à la source ni qualité sur images n'est
déclarée. Aucun modèle ou adapter n'est écrit, téléchargé ou copié.

```text
dotnet run -c Release --project tools/ComfySharp.RuntimeProbe -- sd-all-adapter-train --checkpoint <checkpoint.safetensors> --resume <adapter-existant.safetensors> --report <nouveau-rapport.json> --device cpu --bypass true
```

Le diagnostic ci-dessus exerce la fabrique de poids sans imposer le format de nom
du nœud source ; il ne restaure pas le compteur d'étapes. L'optimiseur repart avec
un état neuf, comme le parcours source. Précision mixte, autres algorithmes, nœud
public, reprise complète et plateformes restent ouverts. Voir [les preuves](qualification/lora-resume.json).
