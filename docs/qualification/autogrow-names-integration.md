# Intégration locale du contrat TemplateNames

La suite complète Windows Release passe **1643 tests uniques, zéro échec et zéro ignoré** après une correction de l'ordre des arguments ordinaires des groupes Autogrow. Les 25 cas du comparateur sont inclus dans ce total : ils couvrent **24 retours source de transport d'arguments et une frontière d'échec du mapper**, pas 25 algorithmes de nœud ou 25 sorties successful. Les comptes, hashes des TRX et six fichiers concernés sont disponibles dans la [preuve JSON](autogrow-names-integration.json).

Cette campagne porte sur les fichiers locaux précisément épinglés ; elle n'est pas attribuée à un HEAD antérieur aux changements. Le build Core puis le build solution ont été rapportés par l'exécutant avec zéro avertissement, zéro erreur et code de sortie 0. Les résultats ci-dessous ont été recalculés indépendamment depuis les TRX persistés, sans relancer les tests.

## Le RED initial est conservé

La première exécution des 25 cas a produit **24 PASS et 1 FAIL**, sans skip. Le seul échec était `ordinary-inputs-around-group` : le source recevait les arguments racine dans l'ordre `tail`, `head`, `values`, tandis que le moteur recevait `head`, `tail`, `values`. Les valeurs des arguments concordaient ; l'ordre explicite capturé ne concordait pas. Le TRX initial de 45459 octets, SHA-256 `c7d35ab918a81ad19a9b5962a08394bfabfdc78e8db95741c3bdf29ba6bbe73e`, reste identifié dans la preuve.

La correction dans `EngineService` intervient après le contrôle existant du premier blocker. Pour un nœud Autogrow, elle présente les arguments plats au binder dans l'ordre du prompt ; le binder ajoute ensuite les groupes dans l'ordre de déclaration et leurs membres dans l'ordre du template. L'ordre de résolution des producteurs, la résolution lazy et le chemin des nœuds ordinaires restent inchangés. Les assertions du comparateur, les deux ressources et leurs octets n'ont pas été ajustés pour faire disparaître l'échec. Quatre régressions d'ordre accompagnent le correctif.

La campagne Core après correction a passé **253 tests**, dont les 25 comparaisons source, les quatre régressions d'ordre et les 36 tests de comportement Names existants. La campagne solution qui suit inclut ces 253 tests ; les campagnes répétées ne sont pas additionnées au total unique.

| Projet | Tests passés dans la suite complète |
|---|---:|
| Core | 253 |
| Desktop | 35 |
| Host | 189 |
| Inference | 577 |
| Tokenization | 540 |
| Workflow | 49 |
| **Total unique** | **1643** |

Les six TRX contiennent 1643 identifiants distincts, aucun doublon entre projets, aucun test non exécuté et aucun échec. Les 253 identifiants de la campagne Core ciblée sont également présents dans la suite complète. Les quatre nouveaux tests d'ordre et les 25 comparaisons ne doivent pas être ajoutés une seconde fois à ce total.

## Source indépendante et périmètre exact

La [collecte source](autogrow-names-source-4312bda.md) a été exécutée après publication du laboratoire au commit [`4312bdafabe2906b74bc51cdca678aa26aeec5ee`](https://github.com/skyline624/ComfySharp/tree/4312bdafabe2906b74bc51cdca678aa26aeec5ee/labs/autogrow-names-source), avec Python 3.12.10 et le backend figé `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`. Le corpus de 296326 octets, SHA-256 `f3f6772bdf05aa464cc831edb8e929ff179f73df3d01ea2f4546616eafcbe4e3`, et son protocole de 47770 octets ont été copiés exactement comme ressources embarquées. Leurs hashes complets sont vérifiés à chaque lecture du comparateur.

Le corpus conserve ses 36 observations et ses classifications prospectives : 25 comparables, six templates non pris en charge, une erreur de constructeur, deux cas StringFormat source-only et deux entrées non prises en charge. Les dix blobs et 73 sélections AST proviennent du laboratoire publié. Les onze observations hors périmètre ne deviennent pas des succès de port ; les deux corps StringFormat ne sont pas exécutés par le comparateur C#.

`FixtureNames` est explicitement un transport d'arguments de laboratoire. Côté source, il utilise les vrais ComfyNode, Schema, TemplateNames, NodeOutput, binder et mapper, puis retourne ses kwargs. Côté C#, une sonde de test utilise les vrais contrats et le vrai EngineService, capture les arguments empruntés et retourne une Map détenue par le contexte. Aucun nœud produit n'est enregistré par ce test et aucun algorithme de StringFormat n'est implémenté.

Le comparateur vérifie les métadonnées originales, les noms et snapshots, les partitions requises/optionnelles, les ordres, les arguments après mapping et les valeurs de transport. Un seul callback C# est capturé et comparé aux observations source du retour du binder et de l'entrée du corps ; la preuve ne prétend pas posséder deux hooks internes C#. Les **23 appels source au binder/corps et 27 lignes transportées** sont des observations dans les 25 cas, pas des tests supplémentaires ni une qualification d'algorithme de nœud.

Les producteurs typés des tests tirent leurs valeurs exclusivement du cache déclaré dans le protocole. Un marqueur `$blocker` est décodé uniquement dans cette fixture de cache. Les dictionnaires du prompt ne reçoivent aucune syntaxe privée permettant de créer un blocker. Le choix du premier blocker reste effectué sur les entrées plates dans l'ordre du prompt ; les marqueurs imbriqués restent des valeurs transportées.

Le cas `empty-list-before-blocker` conserve son résultat source `IndexError` à l'étape mapper, avec zéro appel au binder/corps et aucun report blocker. Le port produit son diagnostic local `execution_error` pour l'impossibilité de répéter le dernier élément d'une liste vide, sans sortie ni appel au corps. Ces deux frontières de panne sont affirmées séparément ; les types et messages Python/.NET ne sont pas déclarés identiques.

## Intégrité et limites

Les six fichiers épinglés sont le nouveau comparateur, son csproj, les deux ressources, `EngineService.cs` et `AutogrowArgumentOrderTests.cs`. Leurs octets UTF8 LF correspondent aux blobs Git canoniques ; les snapshots pris pendant puis après la validation complète sont égaux. Ce contrôle de fichiers n'est ni une attestation des binaires chargés ni un snapshot pris avant compilation. Les quatre fichiers du comparateur et de ses ressources sont également restés identiques depuis le premier RED. La documentation du contrat et les preuves elles-mêmes ne sont pas incluses récursivement dans ces six pins.

Ce résultat établit les comparaisons locales décrites. Il n'annonce aucune parité HTTP ou frontend, aucune API d'acquisition C# fictive équivalente au cache/hidden V3 source, aucune qualification de StringFormat, ni nouvelle validation multi-plateforme. Les limites fonctionnelles de [TemplateNames](../AUTOGROW_NAMES.md) restent applicables.
