# Générateurs analytiques de niveaux de bruit

`ComfySharp.Inference.SigmaSchedules` porte cinq fonctions de la [source ComfyUI figée](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/k_diffusion/sampling.py#L23) : Karras, Exponential, Polyexponential, Laplace et VP. Elles produisent un tenseur CPU Float32 possédé par l'appelant ou son dispose scope actif. Elles ne sont pas encore enregistrées comme nœuds dans le Host et n'exécutent aucun modèle ni sampler.

| Méthode | Nombre de valeurs | Convention préservée |
|---|---:|---|
| `Karras` | steps + 1 | Puissances des bornes calculées en double, puis opérations tensorielles F32 ; zéro ajouté |
| `Exponential` | steps + 1 | Logs scalaires en double, intervalle et exponentielle F32 ; zéro ajouté |
| `Polyexponential` | steps + 1 | Puissance du ramp F32 ; rho = 0 conserve le comportement 0^0 = 1 ; zéro ajouté |
| `Laplace` | steps | Epsilon 1e-5 et clamp conservés ; aucun zéro ajouté |
| `VP` | steps + 1 | `expm1` natif, y compris aux petits exposants ; zéro ajouté |

Les bornes ne sont pas réordonnées. Une demande ascendante reste ascendante ; le clamp Laplace avec bornes inversées reproduit l'amont. Des zéros supplémentaires peuvent exister lorsque les paramètres les produisent. Un conteneur imposant partout une suite strictement décroissante terminée par un unique zéro serait incompatible avec ces fonctions.

## Domaines et erreurs

`steps` est limité à 1–10 000, comme les widgets amont. Les paramètres doivent être finis. Les sigmas sont non négatifs et strictement positifs pour Exponential/Polyexponential. Karras exige rho > 0 ; Polyexponential accepte rho = 0. Laplace exige beta >= 0 ; VP exige betaD/betaMin >= 0 et epsS dans [0, 1]. Les dépassements de ces domaines produisent `ArgumentOutOfRangeException`. Les autres maxima des widgets ne sont pas des limites supplémentaires de cette API mathématique.

Ces vérifications sont plus explicites que les helpers Python bruts. Karras rho = 0 et les logs de zéro échouent dans la référence ; VP peut renvoyer des infinis pour de grandes valeurs de betaD. Le port rejette une sortie non finie par `ArithmeticException` au lieu de la présenter comme une suite exploitable. Cette différence de diagnostic est intentionnelle et ne fournit aucune valeur numérique de remplacement.

Les ressources temporaires sont libérées par dispose scope en cas de réussite, erreur ou annulation. Le jeton est vérifié avant le calcul et avant restitution ; il ne peut pas interrompre une opération native déjà en cours. Ces petits calculs bornés ne prouvent pas la stabilité mémoire d'une génération complète.

## Corpus et profil de comparaison

Le [corpus distribué](../tests/ComfySharp.Inference.Tests/Fixtures/sigma-schedules.cpu-f32.json) contient **46 cas valides et quatre cas invalides**. SHA-256 du fichier : `c78e03c13d3feed5aaf876a9efa5cc2f9f53822765be5cb6c97b0469f63412e1`. Chaque cas contient paramètres, valeurs, bits Float32 et hash des octets ; les sources portent le commit et les hashes AST des fonctions extraites.

Le laboratoire séparé a utilisé Python 3.12.10, PyTorch **2.13.0+cu130**, CPU Float32, un thread, sur Windows x64. L'étiquette CUDA du paquet ne décrit pas le périphérique utilisé : tous les calculs de référence ont été forcés sur CPU. Aucun modèle n'a été chargé. La machine locale expose le processeur `AMD Eng Sample: 100-000000053-04_32/20_N`. La cible C# utilise TorchSharp 0.107/libtorch **2.10** ; la différence de runtime est enregistrée et ne doit pas être effacée des preuves.

Pour reproduire la collecte dans un laboratoire extérieur au produit : lire `comfy/k_diffusion/sampling.py` avec `git show` au commit figé ; extraire ses six définitions de fonctions `append_zero` et `get_sigmas_*` correspondant aux cinq générateurs ; exécuter uniquement ces AST avec `math` et `torch` ; fixer CPU, le dtype par défaut Float32 et un thread ; appeler chaque cas avec les paramètres du corpus, puis exporter les bits et hashes. Aucun import de ComfyUI ou fonction C# n'intervient. Deux collectes successives ont produit le même fichier. Le laboratoire et son interpréteur ne sont pas inclus dans le dépôt produit ; les tests .NET lisent uniquement la ressource JSON embarquée.

Le profil **sigma-cpu-f32-v1** est fixé avant l'acceptation du port :

- Forme, dtype et emplacement des zéros exacts ; bits des zéros exacts.
- Pour chaque valeur finie : `abs(actual - expected) <= 1e-7 + 2e-6 * abs(expected)`.
- Écart ULP et erreur absolue maximale consignés par cas pour examiner les différences de calcul ; aucun assouplissement automatique du seuil.
- Domaines invalides vérifiés comme erreurs explicites, avec résultat ou exception amont conservé dans le corpus.

Ce profil encadre les arrondis Float32 de ces opérations élémentaires. Il ne constitue pas une tolérance de modèle, de gradient d'entraînement ou de sampler complet. Toute évolution du corpus ou du profil exige une revue et de nouvelles preuves ; les tests vérifient aussi son hash.

La [campagne CI `34608289162`](https://github.com/skyline624/ComfySharp/actions/runs/34608289162), au commit [`d5b6a4e`](https://github.com/skyline624/ComfySharp/commit/d5b6a4e7c555776328d38659925d62a2ac3c6d2b), passe les 51 tests de corpus et 37 tests comportementaux sur chaque OS, dans une suite de **338 tests par plateforme**, sans test marqué ignoré. Le [rapport de mesures](qualification/sigma-cpu-f32-d5b6a4e.json) conserve les erreurs par cas, extraites des artefacts TRX.

| Runner CPU | Erreur absolue maximale sur le corpus | Écart ULP maximal |
|---|---:|---:|
| Windows Server 2025 x64 | 0 | 0 |
| Ubuntu 24.04 x64 | 1,9073486328125e-5 | 10 |
| macOS 14 ARM64 | 2,288818359375e-5 | 10 |

Toutes les valeurs satisfont le seuil absolu **plus relatif** fixé ; ces maxima ne remplacent pas le contrôle par valeur. La référence Windows et les résultats Windows de ce corpus sont identiques bit à bit. Cela ne promet pas une identité pour tous les paramètres, runtimes ou backends possibles. Aucun profil ni vecteur de référence n'a été modifié après la comparaison.

Les neuf schedulers du registre global, les opérations SIGMAS, les tables spécialisées, les contrats de `model_sampling` et leur intégration aux vrais modèles restent au lot 6. Le manifeste ne déclare aucun nœud supplémentaire porté à partir de cette seule API.
