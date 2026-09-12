# CaseConverter : intégration locale

La campagne locale Windows termine avec **2 225 tests réussis, zéro échec et zéro test ignoré**, dans six TRX. Le build solution Release a terminé avec zéro avertissement et zéro erreur ; root confirme le retour 0 du build et de la commande complète. Cet audit des artefacts vérifie séparément les compteurs, les identités de tests et les empreintes, sans relancer de calcul. Les [preuves structurées](case-converter-integration.json) conservent les pins et l’historique.

| Projet | Tests réussis |
|---|---:|
| Core | 750 |
| Desktop | 42 |
| Host | 244 |
| Inference | 577 |
| Tokenization | 540 |
| Workflow | 72 |

Les **249 nouveaux cas** comprennent 28 comportements du nœud, 68 tests du helper Unicode, 85 comparaisons d’empreintes source, 40 cas du nœud source, quatre régressions d’ordre des arguments, 19 tests Host, trois Workflow et deux Desktop. Les campagnes ciblées sont des sous-ensembles ou répétitions et ne s’ajoutent pas au total. Les 2 225 testId et executionId sont uniques ; deux anciens noms de théories Inference tronqués correspondent à plusieurs identités distinctes.

## Calculs et référence source

`CaseConverter` est enregistré et accessible dans le compilateur de workflows, le Host et l’éditeur. Il conserve les entrées `string` et `mode`, les quatre choix `UPPERCASE`, `lowercase`, `Capitalize`, `Title Case`, et la sortie STRING. La projection V3 COMBO suit la source ; la sélection initiale UPPERCASE de l’éditeur est un choix local, pas un défaut de paramètre déclaré par la source.

Le [laboratoire publié avant collecte](case-converter-source-74e1f79.md), commit `74e1f790f0c22ddd91d1230322ed3abc44b0143a`, utilise 10 blobs et 74 déclarations AST de ComfyUI `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`, avec CPython 3.12.10 / Unicode 15.0.0. Le corpus de 268 611 octets, SHA256 `649e592edbb38b02220f9f7b92fda634cb6e8a3c1eea1bc5bde95e2165f98e7b`, et le protocole sont embarqués sans modification. La collecte n’a consulté aucune table ni sortie C#.

Les **85 tests** appellent réellement `PythonUnicodeCase` pour les 1 112 064 scalaires valides dans cinq constructions : upper(cp), capitalize(cp), capitalize(A + cp + Σ), title(A + cp + Σ) et title(AΣ + cp + A). Cela représente **5 560 320 entrées par passage**. Entrées et sorties utilisent le même cadre : codepoint uint32 little endian, longueur UTF-8 stricte uint32 little endian, puis octets UTF-8. Compteurs, longueurs et hashes concordent. La source a calculé trois passages réels, soit 16 680 960 appels builtin ; le comparateur C# fait **un passage par plan** et le compare aux trois hashes source. Ces cinq constructions ne prouvent pas toutes les chaînes ni tous les contextes. Les 68 anciennes comparaisons lowercase restent couvertes par la suite ; leur corpus n’a pas été régénéré.

Le helper utilise les mappings complets upper/title extraits statiquement des tables C de CPython 3.12.10. La capitalisation et le titre analysent le texte original : le titre dépend du caractère immédiatement précédent, tandis que le sigma final ignore les caractères Case_Ignorable pour son contexte. Le contrat porte sur des scalaires Unicode valides, sans normalisation, casefold ni culture courante. Les surrogates UTF-16 isolés sont refusés localement.

Les **40 tests du nœud** passent par le vrai registre, le moteur et une observation qui délègue au vrai corps. Parmi les 28 comparables, 27 cas textuels produisent 32 chaînes avec le mapping de listes ; un blocker empêche tout appel du corps. Métadonnées, acquisition/binding, arguments, sorties et signalement du blocker sont comparés sans normaliser leur ordre.

Les huit erreurs source et quatre modes invalides restent distincts. Pour null ou un entier avec un mode reconnu, la source lève AttributeError ; le moteur local convertit ces valeurs en chaînes `None` ou `7`, alors qu’un appel local typé les refuse avec ArgumentException. Pour les quatre modes hors COMBO, la source observée sans admission globale retourne un slot contenant la chaîne inchangée. Le corps local direct conserve cette identité, mais l’admission ordinaire du moteur rejette le mode. Aucune égalité d’exceptions ou d’admission HTTP n’est revendiquée.

## Échec initial et correction vérifiée

La première campagne de références avait **124 Passed / 1 Failed sur 125**. Dans le cas au prompt inversé, la source acquiert et construit les arguments dans l’ordre `mode, string`, tandis que l’invocation C# recevait `string, mode`. L’ordre de signature observé dans le corps Python reste une observation différente. Le corpus et ses assertions ont été conservés.

La correction d’`EngineService` ordonne désormais les arguments des invocations ordinaires selon le prompt après la détection des blockers. Elle préserve l’ordre de résolution des dépendances et évite de demander les entrées lazy inutilisées. Quatre régressions croisent les contrats legacy/V3 et le mapping scalaire/InputIsList. Core a ensuite passé **750 cas**, puis la campagne complète 2 225.

Les premières étapes restent traçables : un ancien run de 23 cas échouait avant l’enregistrement du nœud ; un run sans résultat sur une ancienne DLL ne constitue pas une validation. Une accolade manquante avait été corrigée uniquement dans un test. Les campagnes Core 621, Host 244, Workflow 72 et Desktop 42 étaient vertes avant la correction finale d’ordre ; elles ne suffisent pas seules à qualifier cette correction.

## Smoke réel et provenance

Le **second smoke**, après la correction Engine et le build complet, ouvre la fenêtre Desktop, compile les workflows, les soumet au Host supervisé et lit les previews depuis l’historique terminé. Pour `ǳABC AΣ`, les quatre modes affichent respectivement `ǱABC AΣ`, `ǳabc aς`, `ǲabc aς` et `ǲabc Aς`. Le résultat est exit0, avec stderr vide ; le marqueur de succès intervient après ces assertions. Les empreintes du script et des trois artefacts sont conservées.

Les pins actuels de code, tests et ressources sont capturés **après la campagne**, puis comparés aux gels et revues antérieurs A/B/C. Il n’existe pas de nouveau snapshot pré-full établi par cet audit. Root confirme la continuité du code depuis la correction finale, Core 750 et le build complet ; les annotations documentaires post-campagne sont identifiées séparément. Ces empreintes ne constituent ni une attestation des DLL chargées ni une attribution globale à un commit déjà publié. Le pin de l’exécutable source identifie un lanceur venv, pas la DLL de l’interpréteur.

L’auteur du collecteur a réalisé cet audit d’artefacts d’intégration sans rejouer la source, le builtin casing ou les tests .NET. Cette campagne locale ne qualifie pas les autres OS, tous les contextes Unicode, toutes les routes HTTP ni de nouveaux résultats numériques d’inférence.
