# StringFormat : profil texte partiel

`StringFormat` (`Format Text`, catégorie `text`) applique le profil C# limité `python-format-text-v1`. Le nœud utilise le moteur, le binder Autogrow et le mapping existants ; il n'embarque aucun runtime Python. Son statut reste **partial** : toutes les fonctionnalités de `str.format` ne sont pas portées.

Le [corps ComfyUI figé](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_extras/nodes_string.py#L9) appelle `f_string.format(**values)`. Le schéma conserve `values` avant `f_string`, le template Names `a` à `z` avec minimum zéro, le prototype wildcard `value`, une sortie STRING ordinaire et les alias `string` / `format`. La description locale annonce volontairement le profil partiel ; elle ne reprend pas la promesse source de toutes les fonctionnalités Python.

## Entrées et exécution

`/object_info/StringFormat` expose le template V3 original `COMFY_AUTOGROW_V3`. Le prompt utilise ses feuilles plates `values.a` à `values.z`, avec des trous autorisés. Le binder construit ensuite la Map `values` dans l'ordre du template. Les noms en dehors de ce domaine sont rejetés avant la mise en file.

`f_string` est une entrée STRING requise. Son widget est multiligne et son défaut est `{a}`. Ce défaut sert à créer le widget ; le moteur ne l'insère pas lorsqu'un prompt HTTP omet l'entrée requise. Une constante peut réussir sans aucune feuille `values.*`. Un champ réellement utilisé doit exister.

Exemple de prompt pour une substitution simple :

```json
{
  "text": {"class_type": "PrimitiveString", "inputs": {"value": "Bonjour"}},
  "format": {"class_type": "StringFormat", "inputs": {
    "values.a": ["text", 0], "f_string": "Texte : {a}"
  }},
  "preview": {"class_type": "PreviewAny", "inputs": {"source": ["format", 0]}}
}
```

Les entrées sont eager. Un blocker direct bloque l'appel, même si le texte ne référence pas cette entrée. Une valeur non formatable mais inutilisée peut être transportée sans être convertie. Le nœud ne conserve pas les RuntimeValue après l'appel et rend une chaîne détenue par son contexte.

Le mapping reste scalaire (`InputIsList=false`) : une sortie CreateList peut produire plusieurs appels ; une colonne plus courte répète son dernier élément. `f_string` peut lui-même être connecté et mappé. Le résultat de chaque appel est une chaîne, pas une liste concaténée implicitement.

## Formats admis et refus explicites

| Admis dans ce profil | Hors profil |
|---|---|
| Texte littéral, `{{` et `}}`, champs nommés simples et répétés | Accès `.attr` ou `[item]`, champs imbriqués |
| Chaînes Unicode valides, format vide ou `s`, conversion `!s` | Conversions `!r`, `!a`, objets personnalisés ou réflexion |
| Spec texte `[[fill]align][width][.precision][s]`, alignements `<`, `>`, `^` | Formats numériques, locale, signes, regroupement, `=`, `#`, `z` |
| Remplissage d'un code point, largeur/précision ASCII bornées, remplissage explicite `0>6` | Remplissage par accolade, largeur avec zéros initiaux |
| null, bool et entier Int64 avec format vide ; avec `!s`, application d'une spec texte | Float, entier hors Int64, liste, dictionnaire, ressource native ou blocker imbriqué effectivement formaté |

Les valeurs scalaires admises sans spec deviennent `None`, `True`, `False` ou un entier décimal invariant. Une spec non vide sur null/bool/int requiert `!s` ; elle n'est pas appliquée implicitement comme si la valeur était déjà une chaîne. Il n'existe aucun repli arbitraire vers `ToString` ou JSON pour les autres types.

Largeur et précision comptent les code points Unicode, pas les unités UTF-16 ni les graphèmes. Les surrogates isolés sont refusés. Les champs sont évalués de gauche à droite ; une erreur tardive ne doit pas remplacer une erreur antérieure. `{}` et `{0}` n'accèdent à aucun argument positionnel et ne sont pas des alias de `a`.

Le format est limité à 65536 code points ; largeur, précision et sortie à 1048576 code points. Ces limites de ressources sont locales au produit, pas des règles Python. Leur dépassement produit un refus explicite, sans troncature silencieuse. L'annulation reste celle du moteur et ne devient pas une erreur de format.

Le moteur expose les erreurs du corps avec le code existant `execution_error`. Le message indique une catégorie du profil : `unsupported_format_feature`, `unsupported_format_value`, `format_syntax`, `format_missing_field` ou `format_limit`. Une fonctionnalité Python valide mais absente du profil est signalée comme non prise en charge. Les diagnostics Python et C# ne sont pas déclarés identiques.

## Workflow et Desktop

Le compilateur persiste le widget `f_string` et accepte ses connexions, qui remplacent le littéral sauvegardé. Les feuilles connectées restent plates. Les slots et indices des workflows importés sont préservés ; le compilateur ne force pas une réécriture en 26 ports.

Le Desktop propose 26 ports wildcard fixes `values.a` à `values.z` et le widget multiligne. Cette présentation ne constitue pas une implémentation de la croissance dynamique du frontend ComfyUI, ni d'une propagation générale de types entre ports. Le schéma HTTP original reste un template Names.

## Validation de cette tranche

Des tests ordinaires Host couvrent la projection du vrai schéma, la compilation et l'exécution HTTP vers PreviewAny, les connexions sur le widget, les trous, le mapping avec CreateList, les erreurs de profil et les refus d'admission. Ils utilisent uniquement les nœuds enregistrés, sans substitution par un faux StringFormat. Les tests de comportement ne sont pas un oracle source généré en C#.

La [preuve d'intégration locale](qualification/string-format-integration.md) consigne 1752 tests uniques PASS sous Windows, dont 36 tests du profil, 44 comparaisons source, 12 tests Host, 14 Workflow et trois Desktop. Les 44 cas restent répartis en 24 comparaisons de sorties (26 chaînes exactes), 12 frontières d'erreurs et huit refus explicites de fonctionnalités hors profil ; ce ne sont pas 44 succès de formatage. Le smoke Desktop et son Host supervisé ont également exécuté StringFormat et vérifié son aperçu.

Le [laboratoire source publié](qualification/string-format-source-52fe346.md) a produit ses observations avant la comparaison C#. Ces attendus n'ont pas été calculés par le formateur C#. Les répétitions et campagnes ciblées sont incluses sans double comptage dans la preuve ; les fichiers et TRX sont épinglés.

Le manifeste porte `unitTests=true` sur l'implémentation `partial`. `realWorkflow` et tous les indicateurs de plateforme restent faux : ces parcours ciblés ne qualifient ni tous les workflows source, ni une plateforme entière, ni toutes les fonctionnalités Python.
