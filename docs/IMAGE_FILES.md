# Nommage et stockage des images

Ces composants assurent le nommage et le stockage de `SaveImage` et `PreviewImage`, désormais enregistrés dans le Host. Leur [raccordement tensoriel, métadonnées et API `/view`](IMAGE_FILE_NODES.md) est documenté séparément. Le catalogue compte 37 identifiants ; les [aperçus bitmap de Desktop](BITMAP_PREVIEWS.md) disposent maintenant d’une tranche distincte. Les résultats ci-dessous décrivent la campagne antérieure des fondations.

## Règles de nommage

`ImageFileNaming.Prepare` applique les substitutions de dimensions et d'heure locale, normalise le préfixe, prépare le sous-dossier et examine les entrées existantes. La référence est [`folder_paths.get_save_image_path`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/folder_paths.py#L520), complétée par le nom final de [`SaveImage.save_images`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/nodes.py#L1708).

Le scan considère toutes les entrées, y compris les dossiers et les extensions autres que PNG. Il prend la partie entière située après le préfixe et avant le premier point ou underscore. Les compteurs signés, les espaces autorisés par le profil Python et les chiffres décimaux Unicode 15 sont pris en charge avec `BigInteger`. Les portions non numériques donnent zéro. Les slices de noms utilisent les scalaires Unicode ; Windows applique les tables de minuscules Python déjà figées dans le projet.

Le compteur initial vaut un en l'absence de correspondance. Sinon il vaut le maximum trouvé plus un, même lorsque tous les compteurs correspondants sont négatifs. Le suffixe utilise cinq caractères au minimum, signe compris. `%batch_num%` est remplacé **après** le scan : une nouvelle sauvegarde peut donc retrouver le compteur un et réutiliser un ancien nom. Le stockage conserve ce comportement de remplacement.

Les substitutions backend portent sur `%width%`, `%height%`, `%year%`, `%month%`, `%day%`, `%hour%`, `%minute%` et `%second%`. Les formats frontend, dont `%date:...%` et les références aux widgets, restent littéraux à ce niveau et doivent être résolus par le compilateur de document. L'heure est fournie explicitement au plan de nommage pour permettre des tests déterministes.

## Accès aux fichiers

Le contrat `IImageFileStore` appartient à Contracts et ne transporte aucun tenseur. `ImageFileStore`, dans Storage, prépare les dossiers et écrit ou lit les fichiers dans les répertoires `output` et `temp` des données propres à ComfySharp. Il ne dépend ni de TorchSharp ni des règles Python de nommage. Son ajout ne change que les références entre projets dans les 16 verrous concernés ; les paquets et leurs hashes sont conservés.

Les chemins sont vérifiés après résolution des liens symboliques et des jonctions. Les liens internes sont admis ; un chemin sortant de son répertoire média est refusé. Le contrôle est refait à l'ouverture du fichier, pour détecter notamment le remplacement d'un dossier préparé par un lien externe. La résolution utilise la cible exposée par [`FileSystemInfo.LinkTarget`](https://learn.microsoft.com/en-us/dotnet/api/system.io.filesysteminfo.linktarget?view=net-10.0). Ce contrôle ne verrouille pas l'arborescence contre une modification externe concurrente entre vérification et ouverture.

Chaque écriture remplace et tronque le fichier nommé. Elle n'est pas une transaction sur tout un batch : les fichiers précédemment écrits restent présents si une opération suivante échoue. Une annulation déjà demandée est vérifiée avant l'ouverture ; une interruption pendant une écriture peut laisser un fichier partiel. L'interface de stockage reçoit des octets encodés et ne déclare pas leur contenu valide à elle seule.

## Qualification

La [campagne locale Windows CPU](qualification/image-file-foundations.json) passe **2 450 tests sélectionnés, dont 41 nouveaux**, sans échec parmi ces tests, avec une compilation sans avertissement. **Un test supplémentaire de liens symboliques de fichiers n'est pas sélectionné dans cette campagne locale**, pour la raison ci-dessous. Les 27 fichiers source, tests, verrous et configurations relevés avant compilation sont identiques après la campagne. La restauration des huit graphes de dépendances build/publish Windows CPU, Linux CPU, macOS CPU et Windows CUDA a réussi, puis le graphe local a été restauré en mode verrouillé.

Les tests de nommage utilisent des entrées de répertoire explicites. Les tests de stockage créent de vrais fichiers, dossiers et liens temporaires. Sous Windows, les tests de liens de dossiers utilisent des jonctions créées par un helper C# fondé sur le [buffer de point de montage Windows](https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-fscc/ca069dad-ed16-42aa-b057-b6b207f447cc) ; sous Unix, ils utilisent des liens symboliques. Le test de liens symboliques de fichiers reste distinct et actif dans la suite, avec le trait `Requires=SymbolicLinks`.

Le poste Windows local ne dispose pas du privilège de création de liens symboliques : le premier essai de ce test a échoué pendant la préparation de son environnement, avant l'accès au composant testé. Les jonctions sont testées localement ; le test de liens symboliques de fichiers doit être vérifié dans la CI ou sur un poste disposant du privilège requis. Il n'est pas assimilé à un test réussi.

Les comparaisons exécutées contre les fonctions Python figées et les noms contenant des séquences UTF-16 invalides restent à qualifier. Le parcours tensoriel est vérifié dans la [tranche suivante](IMAGE_FILE_NODES.md), dans son profil borné. Aucun statut complet de compatibilité de `SaveImage`, `PreviewImage` ou d'une famille de modèles n'est déduit de ces fondations.
