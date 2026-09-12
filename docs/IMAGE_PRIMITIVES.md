# Primitives IMAGE

Quatre nœuds exécutent des opérations réelles sur des tenseurs : `EmptyImage`, `ImageInvert`, `RepeatImageBatch` et `ImageFromBatch`. Ils sont enregistrés dans le Host et disponibles dans le compilateur de workflows et l’éditeur. Le profil initial est **CPU, Float32, NHWC**, avec batch, hauteur et largeur strictement positifs et trois ou quatre canaux. Il ne requiert aucun checkpoint.

| Nœud | Opération et contrat |
|---|---|
| EmptyImage | Crée un batch RGB uniforme. Décompose `color` en trois octets, divise par 255 en double, crée trois canaux Float32 puis les concatène. |
| ImageInvert | Calcule `1 - image`, sans clamp ; pour RGBA, recopie le canal alpha d’origine. |
| RepeatImageBatch | Répète la séquence entière du batch, selon `amount`. Même `amount=1` produit un stockage indépendant. |
| ImageFromBatch | Ajoute une fois la taille du batch à un index négatif, borne l’index, tronque la longueur disponible et clone les images sélectionnées. |

Les schémas conservent leurs types IMAGE/INT et leurs options : largeur et hauteur 1–16 384, batch et répétition 1–4 096, couleur 0–16 777 215, index −16 384–16 384 et longueur 1–4 096. `ImageInvert` conserve `essentials_category="Image Tools"` et l’alias `reverse colors`. Repeat et FromBatch utilisent la projection V3 ; Empty et Invert conservent la forme legacy. Les bornes des widgets et de l’admission des littéraux ne remplacent pas les contrôles des opérations sur les tenseurs.

## Mémoire et exécution

`ImageOperations` emprunte ses entrées sans les modifier. Chaque sortie possède un stockage indépendant, survit au scope d’appel et doit être libérée par son propriétaire. Les nœuds transfèrent cette propriété au contexte d’exécution ; une erreur pendant ce transfert libère le tenseur. Des vues d’entrée stridées sont admises si elles satisfont le profil dense CPU/Float32 et les dimensions NHWC.

Un plafond local de **512 MiB de nouveaux payloads tensoriels par appel** est appliqué par défaut. Il est configurable via le paramètre `maxAllocationBytes` de l’API C#, pas via un nouveau widget source. EmptyImage compte deux fois la taille finale : trois canaux intermédiaires puis leur concaténation. Les trois autres opérations comptent leur sortie. Les entrées empruntées, l’overhead du runtime et de l’allocateur, ses caches et les appels concurrents sont exclus : ce plafond n’est ni une garantie de consommation RAM totale ni une limite provenant de ComfyUI. Les calculs de taille vérifiés refusent aussi les débordements.

Les opérations s’exécutent sous `no_grad` sans changer le mode appelant. L’annulation est vérifiée avant le travail et aux frontières où la sortie peut être libérée ; un kernel natif synchrone déjà lancé n’est pas interrompu au milieu de son calcul. Les tenseurs GPU, les autres dtypes, les formats sparse, les dimensions vides et les nombres de canaux autres que 3/4 sont explicitement refusés. La sélection dynamique de périphérique et de dtype de ComfyUI n’est pas portée par cette tranche.

## Route Host et limites de restitution

Le parcours validé est `EmptyImage → RepeatImageBatch → ImageFromBatch → ImageInvert → PreviewAny`, à partir d’un véritable document compilé. L’IMAGE reste une valeur native entre les nœuds. **PreviewAny retourne du texte décrivant le tenseur** ; il ne produit ni PNG, ni fichier image, ni aperçu raster. Le JSON envoyé au Host ne permet pas de substituer un tableau littéral à un buffer IMAGE natif. Un lien incompatible STRING→IMAGE est refusé à l’admission.

Les **quinze nouveaux tests Host ont réussi**, après un build Host sans avertissement ni erreur : quatre schémas servis par `/object_info`, neuf refus HTTP de bornes, types ou entrée manquante avant mise en queue, et deux documents de formats 0.4/1.0 compilés et exécutés jusqu’au texte de PreviewAny. Ils testent les vraies routes et n’introduisent pas d’oracle numérique calculé depuis le produit.

Le smoke graphique réel a également terminé avec exit0 : fenêtre Desktop, Host supervisé, quatre primitives IMAGE et restitution `tensor-as-text`. Le marqueur de succès confirme le parcours IMAGE avec preview textuelle après ses assertions. Le produit est resté inchangé entre ce smoke et la campagne finale ; seuls des tests ont été ajoutés ou corrigés. Cette validation locale n’ajoute aucun PNG ni transport d’image.

La [campagne complète locale](qualification/image-primitives-integration.md) termine avec **2 309 PASS, zéro échec et zéro test ignoré**, après un build solution sans avertissement ni erreur. Les **84 nouveaux cas** comprennent 28 opérations, neuf nœuds, 28 comparaisons source, 15 Host, trois Workflow et un Desktop. Les campagnes ciblées ne sont pas ajoutées une seconde fois au total. Les 23 fichiers capturés avant le build sont identiques après les tests.

Les 28 comparaisons invoquent les vrais nœuds C# une fois par cas et confrontent leurs captures aux trois enregistrements source de la [collecte publiée avant exécution](qualification/image-primitives-source-829a6b8.md) : octets F32 logiques exacts, schémas, layouts et indépendance des sorties après mutation contrôlée. Le premier run avait 22 PASS et six échecs avant le corps, dans la capture d’alignement du test qui refusait des vues non contiguës. Une vue du premier scalaire au même offset, sans copie, a corrigé ce seul test ; le second run a passé les 28 cas sans changement du produit, du protocole ou des fixtures. Le registre compte 34 nœuds ; la qualification de ces primitives reste limitée au profil CPU/F32 local, avec les validations multiplateformes encore à établir.

## Source et portée

Le backend figé est ComfyUI `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a` : [ImageInvert](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/nodes.py#L1936), [EmptyImage](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/nodes.py#L1979), [RepeatImageBatch](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_extras/nodes_images.py#L260) et [ImageFromBatch](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_extras/nodes_images.py#L283). Les blobs LF lus statiquement ont pour SHA256 `f7b6a909bf9a47296b974b730e5db0ec2137a3c6a64dda045970951d7382d4c7` (`nodes.py`) et `f0b2d9060e1772fb6cd317f1965708ec4e55d8d007f3ff583c427c0d07a4a418` (`nodes_images.py`). La collecte de référence Python est isolée dans le laboratoire documenté ci-dessus ; les opérations, le Host, le Desktop et les tests distribués s’exécutent sans Python.

La matrice reste partielle : aucun décodage/encodage de fichier image, SaveImage, chargement de modèle, VAE ou parcours de génération SD n’est ajouté ici. La conformité des opérations dans le profil annoncé ne signifie pas la parité générale des périphériques, dtypes, transports HTTP ou workflows ComfyUI.
