# Diagnostic du parcours réduit SD1.5

`ComfySharp.RuntimeProbe sd15-pipeline` compose la tokenisation C#, CLIP SD1-L,
le débruiteur U-Net EPS, CFG Separate, Euler sans churn et le décodage VAE.
Ce diagnostic utilise des graphes réduits et des poids synthétiques déterministes.
Il ne charge pas de checkpoint préentraîné et ne déclare aucune famille compatible.

Les quatre cas viennent du [protocole publié avant calcul](qualification/sd15-pipeline-native210-cpu-f32-v1.md).
La copie embarquée est vérifiée par SHA-256 avant lecture. La vraie tokenisation
est comparée aux IDs, poids binaires64 et positions des mots de la source figée.
Les cas fixent également le bruit d'entrée, les sigmas et les options de guidage.
Le bruit est une entrée de test déterministe, sans prétention de reproduction d'un seed Gaussian.

| Cas | Conditionnement positif | Étapes | CFG | Départ maximal |
|---|---|---:|---:|---|
| `empty-one-step` | Texte vide | 1 | 1 | Non |
| `weighted-three-step` | Pondérations imbriquées | 3 | 3,5 | Non |
| `two-chunks-separate` | Deux séquences CLIP | 2 | 3,5 | Non |
| `maximum-start` | Pondération implicite | 2 | 7 | Oui |

Le texte négatif est encodé même lorsqu'il est vide ou que CFG vaut 1.
Le latent initial est construit avec la règle de départ du protocole. Après Euler,
le latent de diffusion est converti une fois par division par `0.18215`, puis le VAE
produit une image NHWC. Le pooling CLIP est enregistré mais n'alimente pas U-Net.
La projection CLIP est présente dans la banque, sans calcul projeté demandé par ce parcours.

## Utilisation

Construire la solution avec le runtime CPU de la plateforme, selon le README.
Depuis la racine du dépôt, la commande suivante ne calcule que le plan :

```text
dotnet run --no-build -c Release --project tools/ComfySharp.RuntimeProbe -- sd15-pipeline --case two-chunks-separate
```

Le plan annonce 971 tenseurs de poids, soit 58 137 836 octets en F32 : 37 pour
CLIP, 686 pour U-Net et 248 pour le VAE. Les dimensions réduites sont CLIP largeur16,
deux couches et quatre têtes ; U-Net base32 et contexte16 ; VAE base32.
Les banques contiennent aussi des poids inutilisés par ce parcours de décodage.
Le latent est `[1,4,4,5]` et l'image `[1,32,40,3]`.

L'exécution demande un budget explicite et un dossier absolu qui n'existe pas :

```text
dotnet run --no-build -c Release --project tools/ComfySharp.RuntimeProbe -- sd15-pipeline --case weighted-three-step --synthetic-reduced --execute --memory-budget-mib 4096 --output <nouveau-dossier-absolu>
```

Le plan additionne les poids et entrées, ainsi que des provisions de 512 Mio pour
le runtime, 512 Mio pour les calculs et 256 Mio de marge. Les contrôles de mémoire
du processus complètent ce budget ; ces estimations ne garantissent pas le pic RAM
et ne prouvent pas l'absence de fuite. Un budget insuffisant est un refus explicite.
Aucun modèle n'est téléchargé. Ctrl+C demande l'annulation du parcours actif.

## Résultats et ressources

La commande exécute trois fois le cas, avec observation des frontières désactivée,
activée, puis désactivée. Les trois sorties finales — latent de diffusion, latent VAE
et image — doivent avoir les mêmes hashes à chaque répétition. Les poids et entrées
sont vérifiés avant/après. Chaque graphe retient ses banques pendant l'opération ;
les sorties possèdent leurs ressources et les chemins d'échec les libèrent.

Le dossier reçoit les entrées bruit/sigmas et huit captures F32 : hidden et pooling
positifs, hidden et pooling négatifs, latent initial, latent final, latent VAE et image.
La sérialisation rend la copie contiguë ; les dimensions, strides et offset enregistrés
décrivent le tenseur utilisé dans le calcul. L'image conserve sa vue NHWC dans le moteur.
Cette commande n'expose pas les captures internes des étapes Euler ; elle ne suffit
donc pas à valider les 64 captures du corpus source complet.

`manifest.json` n'apparaît qu'après succès des répétitions, contrôles d'intégrité
et libération des banques. Les écritures existantes ne sont pas remplacées.
Une exécution interrompue peut laisser des fichiers partiels, sans manifeste terminé.
Le manifeste précise les versions déclarées et les bibliothèques natives effectivement
observées dans ce processus avec leurs hashes ; une observation indisponible reste signalée.
Les chemins locaux des bibliothèques ne sont pas inclus.

Les résultats portent les mentions `modelCompatibility: not_assessed` et
`numericalQualification: not_performed`. La production indépendante des références
ComfyUI et leur comparaison relèvent du [laboratoire source](../labs/sd15-pipeline-source/README.md).
Des sorties répétables ne constituent pas une preuve d'accord avec cette source.

## Validation de cette tranche

La suite locale Windows CPU passe 1 351 tests, dont 29 nouveaux : 12 sur les ressources,
erreurs et annulations du parcours réel réduit, six sur les cas embarqués et onze sur
le contrat CLI, les captures de vues et l'exécution synthétique répétée. Les tests de
durée de vie emploient un safetensors synthétique réduit distinct du corpus numérique.
La CI exécute ces contrats dans une étape dédiée avant les comparaisons historiques.

Le [relevé de validation](qualification/sd15-pipeline-composition.json) contient les
empreintes des sources et des résultats. Dans un processus neuf, le vrai point d'entrée
de RuntimeProbe a également produit le plan avec seulement ses assemblies managées :
zéro import natif tenté et zéro bibliothèque native Torch observée avant/après.
Le refus d'un budget de 1 Mio et celui d'une exécution incomplètement configurée
passent dans ce même processus.

La [CI de cette tranche](https://github.com/skyline624/ComfySharp/actions/runs/34657528312)
passe ses 29 nouveaux tests sur Windows, Linux et macOS. La suite globale de cette
révision conserve deux échecs CLIP macOS et quatorze échecs numériques Linux ;
la suite agrégée Linux n'est pas exécutée après ces échecs. Le succès des nouveaux
contrats ne résout pas ces comparaisons antérieures.

Les [références source collectées séparément](qualification/sd15-pipeline-source-6ad8207.md)
sont ensuite consommées par quatre tests .NET, sans Python. Sur le CPU Windows
local, les [64 captures du parcours](qualification/sd15-pipeline-integration.md)
correspondent bit pour bit à la source Windows, y compris les états et scalaires
Euler : 28 832 valeurs F32 comparées, avec les trois sorties finales vérifiées à
chaque répétition. Le vrai CLI, lancé dans quatre processus séparés, produit aussi
les 32 frontières identiques à cette source. La suite locale suivante passe
**1 363 tests**, incluant ces quatre références et huit tests de propriété et
d'intégrité des buffers du diagnostic natif. La campagne CI de cette nouvelle
comparaison reste distincte de celle des 29 contrats.

La [campagne CI suivante, au commit 8264b88](qualification/sd15-pipeline-ci-8264b88.md),
passe les quatre cas sur Windows et macOS, avec les 64 captures bit à bit
identiques à leur référence respective. Linux passe trois cas ; `maximum-start`
présente une première divergence capturée dans le denoised du premier pas,
alors que les sorties CLIP, le latent initial et les sigmas sont exacts. Toutes
les traces sont conservées, y compris celles écrites avant l'assertion en échec.
Ce résultat ne permet pas encore d'identifier une primitive ou un runtime à corriger.

La qualification numérique sur les trois OS, les dimensions complètes, les vrais poids,
le GPU et l'intégration aux nœuds de génération restent ouvertes. Cette tranche n'ajoute
aucun identifiant de nœud au catalogue et ne clôture aucun critère de publication V1.
