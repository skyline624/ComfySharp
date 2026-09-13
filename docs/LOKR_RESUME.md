# Reprise d'entraînement LoKr

La fabrique SD reprend maintenant les facteurs LoKr d'un fichier safetensors ou
d'une map de tenseurs natifs. Elle possède des copies Float32 indépendantes,
conserve les facteurs directs, décomposés et Tucker et respecte l'ordre des
paramètres source. Les facteurs F32, F16, BF16 et F64 collectés sont vérifiés.
Le budget préalable comprend tous les paramètres effectivement enregistrés,
y compris les facteurs décomposés rendus inactifs par un côté direct.

La référence est la [fabrique d'entraînement figée](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_extras/nodes_train.py)
avec [LoKrAdapter et LokrDiff](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/weight_adapter/lokr.py).
Elle retient le premier provider reconnu : LoRA, puis LoHa, puis LoKr. Le choix
de l'algorithme demandé ne remplace pas les adapters déjà reconnus ; il sert aux
cibles absentes. Cette priorité diffère du [chargement d'inférence](LOKR_INFERENCE.md).

La clé alpha consultée est `diffusion_model.<module>.weight.alpha`.
La clé exportée ordinaire `<module>.alpha` reste ignorée, avec valeur de reprise 1.
Les différences de normalisation et de biais sont recréées à zéro. DoRA ne fait
pas partie de `LokrDiff`. Les facteurs `lokr_w1_b` sans `lokr_w1_a`, ainsi que
`lokr_t2` sans `lokr_w2_a`, ne sont pas enregistrés par le constructeur source :
ils figurent dans les clés inutilisées du plan. Un couple décomposé présent et
inactif reste en revanche enregistré, avec gradient absent s'il n'est pas utilisé.

La validation des métadonnées refuse une géométrie incomplète ou incompatible
avant allocation des paramètres. Une valeur non finie provoque un échec avec
libération des ressources déjà allouées ; la source empruntée reste lisible.
Les erreurs natives Tucker observées et les différences alpha entre entraînement
et inférence restent celles documentées dans les deux chemins de calcul.

## Vérifications

Le laboratoire séparé `labs/lora-training-source/lokr_resume.py` exécute l'AST
amont et les fabriques complètes de deux topologies SD réduites. Quatre scénarios
mêlent reprise LoRA/LoHa/LoKr, créations LoRA/LoHa/LoKr, facteurs directs et
décomposés, Tucker, facteurs inactifs, alpha, DoRA et différences. Chaque paramètre
des 686 cibles et l'état RNG sont comparés. Les références Float32 sont stockées
en octets dédupliqués par SHA-256 et compressés. Aucun poids préentraîné ni runtime
Python n'est inclus dans les tests .NET.

Les 13 tests ciblés comprennent ces quatre fabriques, quatre parcours U-Net
réduits avec gradients/snapshot/rechargement, et cinq cas d'erreur/annulation.
Le test historique d'algorithme non porté conserve OFT et distingue désormais
le cas LoKr reconnu mais incomplet. Les tolérances restent `atol=rtol=3e-5`.
Les deux côtés décomposés appliquent alpha deux fois à l'entraînement et une seule
fois en inférence ; les tests ne réclament pas une égalité contraire à la source.

Sur le vrai SD1.5 Windows CPU, le diagnostic a entraîné un adapter LoKr, repris
ses facteurs en mémoire et effectué deux pas SGD supplémentaires. Les 282 cibles
LoKr ont conservé leurs 564 facteurs ; 404 différences ont été remises à zéro et
282 alpha à 1. Les pertes après reprise sont 1,1825215 puis 1,1677487. Chaque pas
a 968 gradients finis et 282 alpha directs sans gradient. 961 paramètres ont
changé. La base reste inchangée ; le rechargement final donne exactement la même
prédiction. Les détails et hashes sont dans [la preuve](qualification/lokr-resume.json).

```text
dotnet run -c Release --project tools/ComfySharp.RuntimeProbe -- sd-all-adapter-train --algorithm LoKr --rank 7 --checkpoint <checkpoint-existant.safetensors> --report <nouveau.json> --device cpu --resume-roundtrip true
```

Ce diagnostic relit le checkpoint partagé et reprend l'adapter en mémoire. Il
n'écrit aucun nouveau fichier de poids. Les entrées sont des tenseurs de prédiction
synthétiques réduits ; ce n'est pas une qualification de workflow sur images,
de gradients préentraînés contre Python, du nœud public complet ou des GPU.
Les bypass LoKr d'entraînement et d'inférence, les autres précisions, OFT et la
qualification des plateformes restent nécessaires à la V1.
