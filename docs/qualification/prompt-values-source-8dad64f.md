# Valeurs de prompt — collecte source indépendante du commit 8dad64f

La collecte source contient **28 cas distincts**, avec trois observations fraîches identiques par cas. La première validation accepte **16 cas et en rejette 12**. Ces résultats décrivent des fonctions backend figées ; ils ne sont ni une comparaison C#, ni un test du serveur HTTP ou du frontend.

Le [laboratoire publié](https://github.com/skyline624/ComfySharp/tree/8dad64f0cf9b04548ee722b7533776b25d3b4c46/labs/prompt-values-source) utilise CPython **3.12.10**, ici sur Windows, et le backend `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`. Le coordinateur a observé une sortie de processus 0. Le document source achevé mesure **677389 octets**, SHA256 `04af1fcf3c83c91d70cfd7a1d0b97f51a6eca5bdcf9c612ac9ea29fa420bee3f`. [La preuve JSON](prompt-values-source-8dad64f.json) conserve ses identités, les compteurs et chaque observation de cas résumée.

| Phase | Résultats source |
|---|---|
| Reconnaissance originale `is_link` | 7 vrais, 20 faux ; 1 clé absente, prédicat non appelé |
| Acquisition directe originale | 26 retours, 2 TypeError |
| Première validation | 16 acceptations, 12 rejets ; 9 prompts modifiés |
| Acquisition après validation | 16 retours ; 12 non exécutées car validation rejetée |
| Second passage explicite, trois enveloppes | 2 acceptations, 1 rejet ; 1 nouvelle mutation |
| Acquisition après second passage | 2 retours ; 1 non exécutée car validation rejetée |

Les phases non exécutées restent visibles comme conséquences d’un rejet. Elles ne sont ni supprimées ni transformées en réussite. Les trois répétitions ne s’ajoutent pas aux 28 cas uniques.

La frontière observée confirme que **l’acquisition directe n’est pas la validation API**. Une liste brute vide est un littéral pour `get_input_data`, mais le validateur la rejette comme lien mal formé. Le slot booléen false et le slot négatif -1 sont acceptés dans ces fixtures à une sortie ; les slots 0.0 et 0.5 satisfont `is_link`, puis échouent sur l’indexation. Ces comportements source ne sont pas adoptés automatiquement par le moteur C#.

L’enveloppe `__value__` existe bien dans la source. Le validateur déplie une couche et modifie le prompt. L’enveloppe de `[]` est acceptée une première fois ; le second passage voit la liste désormais brute et la rejette. L’enveloppe de `["id",0]` est acceptée puis l’acquisition lit la sortie du cache, car la valeur dépliée a la forme d’un lien. La double enveloppe garde une dict après le premier passage, puis devient l’entier 1 au second. Les quatre conversions scalaires ordinaires enregistrent aussi leurs vrais types, sans assimilation bool/int/float.

L’audit indépendant a vérifié les trois fichiers du laboratoire contre les octets Git publiés, les trois helpers Autogrow épinglés, **12 blobs backend** et **86 sélections AST** (72 héritées et 14 additionnelles ; une déclaration future-annotations est sélectionnée deux fois). Il a recalculé les 28 hashes de records représentatifs contre leurs **84 hashes de répétition enregistrés**, validé **3540 nœuds typés** et **46 snapshots de cache**. Les trois sorties complètes de chaque répétition ne sont pas stockées séparément : le collector conserve un record représentatif et les trois hashes égaux. L’auditeur n’a donc pas prétendu relire 84 payloads complets indépendants.

La publication du résultat est précédée, dans le code épinglé, d’une nouvelle vérification du HEAD, des fichiers publics, des helpers et des sources. L’audit rapproche la provenance finale de ces blobs publiés ; il n’invente pas des snapshots de système de fichiers avant/après qui ne figurent pas dans l’artefact. Les valeurs encodées conservent bool, int, bits float64, ordre des dicts, liste/tuple et absence/null. Seul le texte des tracebacks est omis selon le protocole, avec le nombre de frames conservé. Le scan des chaînes décodées n’a trouvé aucun marqueur de chemin privé.

Le registre contient de vraies classes source, dont PreviewAny comme nœud de sortie de validation. Le cache est une infrastructure synthétique déclarée. Le garde de profil interdit toute tentative d’appel `execute`/`main`, même si le validateur intercepte l’exception ; aucune trace exhaustive indépendante du profiler n’est fournie. Aucun calcul de nœud, modèle, Torch, serveur HTTP ou frontend n’a été exécuté dans ce laboratoire. Les exceptions inattendues de dépendance ne sont pas converties en observations acceptées.

Cette preuve établit une collecte source achevée et auditée dans son périmètre. **Aucune parité C#, HTTP, frontend ou compatibilité générale de prompt n’est qualifiée.** L’audit a effectué uniquement des lectures, analyses AST et recalculs de hashes avec la bibliothèque standard ; il n’a relancé ni le collector, ni .NET, ni une opération native.
