# Références source des composants SD — 86812a9

Le [laboratoire source indépendant](../../labs/sd-source/README.md) a réussi sur
Windows, Ubuntu et macOS dans la
[campagne 34632139846](https://github.com/skyline624/ComfySharp/actions/runs/34632139846),
au commit `86812a9dae975a6a59586de87d9d7d3e0bd1ef99`. Le lien entre cette révision
et la campagne a été vérifié auprès de GitHub. Ce résultat porte sur la production
et l'intégrité des références ComfyUI ; la comparaison du port C# est une étape
distincte.

Le [relevé d'intégrité](sd-source-86812a9.json) recense les trois manifests épinglés
dans les tests .NET. Les six scripts du laboratoire et les onze fichiers de la
référence backend ont été comparés aux objets Git exacts, y compris leurs fins de
ligne. Les dix-neuf extractions AST par plateforme ont aussi été reconstituées et
vérifiées. Aucun code produit ne fournit les sorties attendues.

Chaque cible contient 211 enregistrements de tenseurs, soit 189 515 valeurs. Leurs
SHA-256 ont été recalculés à partir des valeurs et types sérialisés. Les entrées et
les listes de paramètres — noms, dimensions, hashes — sont identiques sur les
trois plateformes. Ces listes ne contiennent pas les valeurs des paramètres :
l'audit établit l'identité des enregistrements, sans prétendre reconstituer leurs
payloads. Les quatre fichiers de résultats Windows sont identiques octet pour
octet à la collecte source locale précédente.

| Cible | PyTorch | Capacité CPU enregistrée | Threads intra/inter |
|---|---|---|---|
| Windows x64 | 2.10.0+cpu | AVX2 | 1 / 1 |
| Ubuntu x64 | 2.10.0+cpu | AVX512 | 1 / 1 |
| macOS arm64 | 2.10.0 | DEFAULT | 1 / 1 |

Python 3.12.10 sert uniquement au laboratoire isolé. Les tests .NET consomment les
JSON incorporés et exécutent le port avec libtorch. Les manifests enregistrent les
hashes des bibliothèques natives mesurés par les runners ; les artifacts JSON ne
contiennent pas ces binaires pour un second hachage indépendant. Une version
commune ne garantit pas des binaires ou une sélection d'instructions identiques.

Le corpus couvre la topologie complète sélectionnée des U-Net SD1/SD2 à largeur
réduite, le VAE classique à largeur réduite, les frontières sigma/EPS/V et les
deux politiques CFG. Les huit sorties U-Net, les huit sorties VAE et les six
sorties CFG sont répétées trois fois dans la source. Les neuf frontières internes
par cas U-Net sont capturées à la première répétition seulement.

La source elle-même produit des différences entre appels CFG séparés et
concaténés. Pour les longueurs de contexte (2,3) et (3,3), respectivement 2/19
éléments sous Windows, 1/3 sous Linux et 27/51 sous macOS dépassent la borne
numérique lorsqu'on compare ces deux politiques entre elles. Le cas (2,10),
incompatible avec la concaténation, utilise le même repli séparé et reste exact.
Chaque politique du port est donc comparée à sa propre référence source, avec
la [tolérance fixée avant les résultats](sd-components-native210-cpu-f32-v1.md).
Ces différences entre politiques ne constituent pas une dispense pour un écart
entre source et port.

Aucun poids préentraîné, graphe aux dimensions complètes, workflow de génération
ou backend GPU n'est qualifié par cette campagne. Le profil CLIP antérieur et ses
échecs encore ouverts restent inchangés.
