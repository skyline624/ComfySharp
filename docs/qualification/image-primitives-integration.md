# Primitives IMAGE : intégration locale

La campagne complète Windows CPU termine avec **2 309 PASS, zéro échec et zéro test ignoré**, après un build solution Release sans avertissement ni erreur. Root confirme le retour 0 des deux commandes et a audité les six TRX, leurs compteurs et identités uniques. Cette synthèse vérifie la cohérence de cet audit et des snapshots, sans refaire les calculs ni le parsing intégral des six TRX. Les [preuves structurées](image-primitives-integration.json) conservent leurs empreintes.

| Projet | PASS | Nouveaux |
|---|---:|---:|
| Core | 750 | 0 |
| Desktop | 43 | 1 |
| Host | 259 | 15 |
| Inference | 642 | 65 |
| Tokenization | 540 | 0 |
| Workflow | 75 | 3 |

Les **84 nouveaux cas** couvrent 28 opérations IMAGE, neuf nœuds, 28 comparaisons source, 15 routes Host, trois workflows et un test d’éditeur. Les campagnes ciblées ne sont pas ajoutées au total. Les quatre nœuds sont enregistrés parmi 34 : `EmptyImage`, `ImageInvert`, `RepeatImageBatch` et `ImageFromBatch`.

## Référence et correction du test

Le protocole et le collecteur ont été publiés au commit `829a6b87f48ef7b1022da512524c5966e998a369` avant la [collecte source](image-primitives-source-829a6b8.md), qui conserve 28 cas et trois enregistrements complets par cas. Son artefact SHA256 `6b31ffb26843506bed8ee0e41cc97e91087e5f46ce01e39cdedf27fa3e218a62` provient des vrais corps ComfyUI figés et de PyTorch CPU 2.10.0, sans expected dérivé du produit.

Chaque test C# invoque une fois le vrai nœud enregistré et compare ses captures aux trois enregistrements source : octets logiques F32 exacts, schémas, layouts, entrée inchangée et propriété indépendante de la sortie après mutation contrôlée. Il n’y a aucune tolérance numérique. Les 84 appels source ne deviennent pas 84 tests C# ni trois exécutions du corps géré par cas.

Le premier run était **22 PASS / 6 FAIL** : la fonction de capture d’alignement du test tentait de lire `bytes()` sur des vues non contiguës et échouait avant le corps. La correction utilise une vue du premier scalaire au même offset, sans copie ; le ValueTask déjà terminé est ensuite consommé avec await. Le produit, les fixtures et le protocole sont restés inchangés. Le second run a passé **28/28**, puis ces cas ont passé dans la campagne complète. Les deux TRX ciblés restent épinglés.

## Route réelle et mémoire

Le smoke graphique a terminé avec exit0 : fenêtre Desktop, document compilé, Host supervisé et quatre primitives dans le trajet `EmptyImage → RepeatImageBatch → ImageFromBatch → ImageInvert → PreviewAny`. La restitution est **tensor-as-text**, pas une image PNG. Le produit n’a pas changé entre ce smoke et la campagne finale ; seuls des tests ont été ajoutés ou corrigés. Les 15 tests Host vérifient aussi les schémas réels, les refus avant queue et les documents 0.4/1.0 jusqu’à la preview textuelle.

Le profil est CPU/Float32, NHWC RGB ou RGBA, avec batch/hauteur/largeur positifs, sous no_grad. Les sorties possèdent un stockage indépendant. L’inversion conserve l’alpha RGBA sans clamp ; la répétition copie même avec amount=1 ; l’extraction borne l’index puis clone. Le plafond par appel de **512 MiB** est configurable dans l’API : EmptyImage compte ses canaux et leur concaténation, les autres opérations leur sortie. Il exclut entrées empruntées, runtime/allocateur, caches et concurrence. L’annulation est vérifiée aux frontières gérées sans prétendre interrompre un kernel natif synchrone.

## Provenance et limites

Root a capturé **23 fichiers avant le build**, puis vérifié leur identité après les tests. Les deux snapshots sont identiques ; leurs pins et ceux des logs sont conservés, sans inventer de date de capture ni attester les DLL chargées. Les annotations de cette preuve sont postérieures à la campagne.

Cette qualification reste locale et partielle. Elle n’ajoute ni GPU/autres dtypes, codec, chargement de poids, VAE, génération SD complète, transport d’image ou nouvelle validation multiplateforme. Elle ne résout pas les gates numériques CI précédemment rouges. La preuve source historique reste distincte ; la matrice ne promeut pas ici ses indicateurs de plateforme ou de workflow général.
