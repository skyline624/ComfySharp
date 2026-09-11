# Composants SD CPU — qualification partielle à ca41c5f

La [campagne 34635368463](https://github.com/skyline624/ComfySharp/actions/runs/34635368463)
teste le commit `ca41c5fd059027b2f97801f18ca782db740edad5`. Les nouveaux composants
SD passent sur Windows et macOS ; cinq comparaisons échouent sous Linux. La
qualification multiplateforme reste ouverte, avec les références et le profil
numérique inchangés.

Le [relevé de résultats et des comparaisons](sd-components-ca41c5f.json) a fait
l'objet d'un recomptage indépendant des 35 rapports TRX et d'un recalcul des
hashes des 240 tenseurs de traces et de leurs références.

| Cible | Suite agrégée et CLIP aux dimensions complètes | Nouveaux tests SD | Contrôles de premier accès natif |
|---|---|---|---|
| Windows x64 | 1 245 réussis, aucun échec | 93 réussis | 182 réussis |
| Ubuntu x64 | Non exécutées après l'échec de la première étape numérique | Non exécutés en suite agrégée | 177 réussis, 5 échecs |
| macOS arm64 | 1 243 réussis, 2 échecs CLIP déjà documentés | 93 réussis | 182 réussis |

Les contrôles de premier accès rejouent des tests dans des processus distincts ;
ils ne s'ajoutent pas au nombre de tests uniques. Aucun des tests exécutés n'est
ignoré. Les étapes Linux non exécutées ne sont pas comptées comme des validations.
L'absence de traces CLIP après cette interruption entraîne également un échec
d'upload d'artifact ; elle ne constitue pas un résultat numérique supplémentaire.

Les 93 nouveaux tests couvrent le chargement et les banques de poids (27), le
graphe U-Net (10), le graphe et les conventions VAE (11), le sampling (11), le
denoiser/CFG (8), la provenance (1), puis les références source sampling (3),
U-Net (8), VAE (8) et CFG (6). Ces références sont décrites dans le
[relevé du laboratoire indépendant](sd-source-86812a9.md).

Sous Linux, les trois tests source de sampling et les huit tests source VAE
passent. Le cas U-Net SD2 avec batch de deux, temps partagé et dimensions
spatiales impaires dépasse la borne sur quatre valeurs de sortie ; l'erreur
absolue maximale est d'environ `4.47e-5`. Quatre comparaisons CFG échouent pour
les contextes de longueurs (2,3) et (3,3), dans les deux politiques. Le repli
séparé du cas (2,10) passe.

Les traces U-Net comprennent neuf frontières internes et une sortie par cas,
soit 80 tenseurs par cible. Leurs valeurs, dimensions et SHA-256 ont été vérifiés.
Les 80 tenseurs Windows et les 80 tenseurs macOS sont identiques bit pour bit à
leurs références respectives. Sous Linux, 23 sont identiques ; toutes les
frontières internes respectent la borne, puis quatre éléments de la sortie du
cas SD2 cité la dépassent. Les huit embeddings temporels sont exacts. La première
divergence enregistrée du cas fautif apparaît à `down0`, uniquement dans la
seconde ligne du batch ; d'autres divergences peuvent apparaître ensuite.

Le manifeste source Linux enregistre AVX512. La sélection d'instructions, les
layouts et l'allocation des buffers font l'objet de contrôles distincts. Les
traces seules ne permettent pas d'attribuer la cause à l'un de ces facteurs.
Aucune copie supplémentaire des entrées produit ni modification d'équation
n'est justifiée sur cette seule observation.

Les tests du moteur, du Host, des documents, de la tokenisation et de l'interface,
les diagnostics CLI, le démarrage Desktop/Host et les probes CPU passent sur
Windows et macOS à cette révision. Les deux échecs macOS concernent le
[profil CLIP antérieur](clip-cpu-investigation-5810a4f.md) ; ils restent visibles
et ne sont pas remplacés par les résultats SD.

Cette campagne utilise des poids synthétiques et des graphes de largeur réduite.
Elle ne qualifie aucun checkpoint préentraîné, workflow génératif, graphe SD aux
dimensions complètes, backend GPU ou famille de modèles. Le dépôt contient
toujours 25 nœuds enregistrés. Le [plan complet](../MIGRATION.md) reste inchangé.
