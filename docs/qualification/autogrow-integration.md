# Autogrow / CreateList — comparaison locale avec la source

Les **18 cas comparables passent sur Windows**, dans une exécution ciblée de
**188 tests Core distincts : 188 réussis, zéro échec, zéro ignoré**. Le build Core
a terminé sans avertissement ni erreur, selon le résultat communiqué par la
racine. Le TRX a été recompté indépendamment, sans réexécuter .NET ou la source.

Cette campagne concerne Core seulement. La précédente campagne complète comptait
1499 tests ; les 18 nouvelles références ne constituent pas une nouvelle campagne
complète de 1517 tests. Aucun résultat Linux ou macOS n'est attesté ici.

Le [JSON de preuve](autogrow-integration.json) épingle les quatre fichiers testés,
le TRX, les 18 identifiants de tests et les six fichiers produit inchangés depuis
`3eafec37fd8d76cb4d43c34626224da5a1d3d4f2`. Le test final a pour SHA-256
`c795b28987612a8d53fd5fae394183997b475711a3202b12d754a19b8799a765`.
Le seul delta depuis son gel est la correction équivalente xUnit2029
`Assert.Empty(events.Where(...))` vers `Assert.DoesNotContain(events, ...)`.
Le TRX a pour SHA-256
`72dbaae8d76e1812c39f2158f23595461fb1229e90034289eb25d60386dd4da3`.

La [collecte source indépendante](autogrow-source-ae241a0.md) provient du
laboratoire publié avant exécution au commit
`ae241a0ae32f4e053f0bc5754e8d4d0ebfc7d5f4`, sur le backend figé
`1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`. Les deux ressources sont conservées
octet pour octet :

- Collecte de 36 cas, 279905 octets :
  `04f2f2d1cea2a66460a832d471f751fcf3effd45ad4bc36a6508d540a02d2924`.
- Protocole Git LF, 39714 octets :
  `772b018f80b35dd7e1d4f4655ab6d58ccbc62c34fcc915f5a93e9cb8ad1d164b`.

Les attendus viennent uniquement de cette source. Les 18 observations exclues
restent dans la fixture ; aucun test ignoré ne les transforme en validation du
port. Les paramètres des six schémas génériques viennent du protocole prospectif,
pas des résultats attendus.

| Comparaison | Couverture réelle |
|---|---|
| Nœuds et schémas | 12 cas du vrai CreateList, six templates de préfixe ; object_info original complet et INPUT_TYPES via NodeRegistry. |
| Finalisation | Feuilles, options et ordre des partitions required/optional via NodeInputExpansion. |
| Regroupement et appels | 15 appels réels via EngineService, comparés aux 15 captures source ; BindArguments exercé aussi directement sur les entrées prospectives non bloquées. |
| Sorties | Listes d'exécution, ordre de concaténation, liste vide, item tableau vide, conteneurs imbriqués et liens dupliqués. |
| Blockers | Trois cas sans appel CreateList ni regroupement ; deux messages observés, priorité du prompt, distinction chaîne vide/null et marqueur imbriqué préservé. |
| Immutabilité | Entrées/cache de la source et prompt C# inchangés ; captures JSON sans conservation de valeurs runtime empruntées. |

Le cas du littéral `[]` est comparé à la frontière des valeurs acquises par
`get_input_data`. Son prompt C# utilise explicitement `{ "__value__": [] }`.
Le validateur upstream prend également en charge ce wrapper ; le laboratoire
n'exécute pas la validation globale de PromptExecutor. Ce cas ne prouve donc ni
l'acceptation HTTP d'un tableau brut ni une incompatibilité entre les grammaires.

Les champs internes hidden/dynamic_paths, les lectures du cache source et
l'admission HTTP complète ne sont pas comparés. L'effet du regroupement est
vérifié par ses arguments et leur ordre, sans fabriquer d'équivalent de ces
champs dans l'API C#. Les cas sans entrée requise, les extras et les templates
non pris en charge restent hors de cette comparaison.

La preuve n'étend pas la portée aux ports automatiques du frontend, à la
propagation MatchType, au scheduler complet, aux modèles ou à la compatibilité
V1 globale. Les détails de provenance et les limites des répétitions source
off/on/off restent décrits dans la preuve source liée.
