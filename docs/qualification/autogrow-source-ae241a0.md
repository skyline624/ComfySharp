# Autogrow / CreateList — observations source prospectives

La collecte du [laboratoire](../../labs/autogrow-source/README.md), publié au commit
`ae241a0ae32f4e053f0bc5754e8d4d0ebfc7d5f4` avant sa première exécution, a terminé
avec **36 cas : 29 retours et sept erreurs prévues**. Elle utilise Python 3.12.10
sur Windows, sans modèle ni backend numérique. Cette preuve porte uniquement sur
la source ; aucun résultat C# n'est comparé ici.

Le fichier collecté fait **279 905 octets**, SHA-256
`04f2f2d1cea2a66460a832d471f751fcf3effd45ad4bc36a6508d540a02d2924`.
Le [JSON de preuve](autogrow-source-ae241a0.json) contient les identités, les
résumés par cas, l'object_info CreateList complet et les limites de comparaison.
Le [protocole prospectif](../../labs/autogrow-source/protocol.json) reste inchangé,
SHA-256 `772b018f80b35dd7e1d4f4655ab6d58ccbc62c34fcc915f5a93e9cb8ad1d164b`.

## Provenance et contrôle

L'audit indépendant de la collecte a passé **766 contrôles stdlib** : octets du
résultat, trois fichiers du laboratoire au commit publié, neuf blobs Git du backend
`1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`, 72 AST/segments/lignes, ordre des cas,
entrées et caches inchangés, classifications et erreurs prescrites. L'audit n'a
réexécuté aucune déclaration source, aucun code .NET ou calcul natif.

Les captures contiennent 31 schémas, 26 appels réels à CreateList, 28 retours réels
de `build_nested_inputs` et deux reports de blockers. Le collecteur a effectué
trois passages frais **off/on/off** par cas. Les 36 records conservés, hors captures
du profileur, ont été rehashés indépendamment et correspondent aux trois hashes
déclarés pour chaque cas. Les payloads des deux passages sans profileur ne sont
pas conservés : leur égalité est attestée par le contrôle effectué dans le
collecteur épinglé, pas par un second rehash de ces payloads absents.

## Contrats observés

Les 20 cas du vrai [CreateList](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_extras/nodes_toolkit.py#L5)
produisent exactement le même object_info. Il conserve le template original
`input.required.inputs` de type `COMFY_AUTOGROW_V3`, son prototype
`COMFY_MATCHTYPE_V3` avec `template_id="type"` et `allowed_types="*"`, le préfixe
`input`, min=1 et max=10. Il n'ajoute pas de section optional vide. Le nœud est
InputIsList et sa sortie `list` est IsList avec le MatchType `type`. Son module
est `comfy_extras.nodes_toolkit`, obtenu par l'affectation explicite du loader
source, et non par un défaut du harnais.

La [finalisation source](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_api/latest/_io.py#L1878)
déclare `inputs.input0` requis et les neuf suivants optionnels. Les chemins
dynamiques présents suivent l'ordre numérique du template. Dans `reverse-gap`,
les inputs plats arrivent dans l'ordre input2/input0, mais les arguments regroupés
et la concaténation suivent input0/input2 : `[1,2]` puis `[3]` donnent `[1,2,3]`.
Une liste d'exécution vide contribue zéro item ; un tableau littéral vide reste
un item. Les conteneurs imbriqués ne sont pas aplatis récursivement.

Cette observation du tableau vide concerne les **valeurs acquises**, pas
l'admission HTTP du prompt brut : le [validateur upstream](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L917)
traite toute liste brute comme une connexion et rejette `[]` comme lien malformé.
Il prend aussi en charge le [wrapper `__value__`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L987).
Le cas `literal-empty-item` ne prouve donc ni l'acceptation HTTP de `[]` ni une
incompatibilité du wrapper entre les deux systèmes. Sa comparaison future doit
porter sur l'item d'exécution obtenu, avec la préparation de prompt appropriée.

Les templates génériques confirment min=0 avec groupe vide, max=1 et max=100,
min=3/max=2 accepté avec deux inputs requis, ainsi qu'un prototype optional
ignorant le minimum. Le groupe extérieur optional ne produit pas le même effet :
son cas source garde un input requis. Ce dernier reste hors tranche du port.

Les trois cas de blocker direct enregistrent **zéro regroupement et zéro appel
CreateList**. La priorité suit le prompt : le message `first` gagne dans l'ordre
inversé ; un premier marqueur silencieux supprime le message suivant ; une chaîne
vide déclenche bien un report. Un marqueur enfoui dans un item n'est pas recherché
récursivement et reste dans le résultat.

Les quatre extras littéraux — noms hors limite, zéro initial, préfixe inconnu
avec valeur null et objet regroupé — sont ignorés par l'acquisition de ces cas.
Les deux extras sous forme de lien restent des arguments supplémentaires et
lèvent `TypeError` au mapper, avant l'entrée dans CreateList.execute. Un lien
introuvable porte en plus `missingKeys` et `[null]`. Les cinq autres erreurs sont
celles des constructeurs : trois bornes invalides, prototype dynamique interdit
et type MatchType invalide. Les étapes et types correspondent exactement aux
autorisations prospectives du protocole.

## Partition proposée pour les tests du port

| Groupe | Cas | Comparaison admissible |
|---|---:|---|
| Comparables | 18 | Valeurs, schémas, ordre, regroupement et champs communs des blockers : 12 CreateList et six templates génériques. |
| Unsupported-input | 8 | Vérifier le diagnostic anticipé convenu du port, pas l'égalité avec les retours source. |
| Unsupported-template | 5 | Vérifier la limite explicite : groupe optional, widget, custom extra_dict, MatchType non-wildcard et prototype dynamique. |
| Constructor-error | 4 | Vérifier le refus des bornes/types invalides, sans exiger une classe d'exception Python en C#. |
| Source-only | 1 | Conserver l'observation d'allowed_types vide sans la convertir en wildcard ni en attendu de succès du port. |

Les identifiants exacts de chaque groupe figurent dans le JSON. La politique
choisie pour le port refuse les noms dynamiques inconnus avant les producteurs ;
elle est plus stricte que les extras littéraux ignorés ici. Par ailleurs,
finalisation/acquisition ne constituent pas la validation des entrées obligatoires
de PromptExecutor : les deux cas sans input0 ne prouvent pas qu'un prompt complet
upstream serait accepté.

Cette collecte ne qualifie ni l'implémentation C#, ni les dix ports fixes du
Desktop, ni l'ajout automatique de ports et la propagation MatchType du frontend.
PromptExecutor complet, ordonnancement, expansion de graphes, DynamicCombo/Slot,
caches persistants, modèles et compatibilité V1 globale restent hors preuve.
