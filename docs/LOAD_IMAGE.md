# Import d'images et SD1.5

`LoadImage` est raccordé au Host et au canvas natif. Il produit un tenseur
`IMAGE` Float32 `[frames,height,width,3]` et un `MASK` égal à `1-alpha`.
Sans canal alpha, le masque vide conserve la forme source `[frames,64,64]`.
Le contenu est relu à chaque exécution et son SHA-256 est enregistré dans
l'historique. Ce hash ne constitue pas encore un port du cache `IS_CHANGED`.

Dans l'application, **Import image…** copie l'image choisie dans les données
`input` du Host et ajoute un nœud. Relier sa sortie à `VAEEncode`, puis à
`KSampler`. Pour ouvrir le workflow d'exemple avec une image existante :

```text
dotnet run --no-build -c Release --project src/ComfySharp.Desktop -- --inference-device cuda:0 --models-dir <dossier-modeles-partage> --workflow docs/workflows/sd15-image-to-image.api.json --import-image <image.png>
```

Cette option remplit le seul `LoadImage` du document. Les poids restent lus
dans le dossier partagé : aucune copie de checkpoint et aucun téléchargement
de modèle ne sont effectués par l'application.

`POST /upload/image` et `/api/upload/image` acceptent le formulaire multipart
`image`, `subfolder`, `overwrite` et `type=input`. Un contenu identique est
réutilisé ; une collision différente reçoit un suffixe sauf demande explicite
de remplacement. Décodage et validation précèdent l'écriture atomique.
Les noms annotés `[input]`, `[output]` et `[temp]` sont acceptés en lecture
dans leurs racines respectives, avec contrôle des chemins physiques.

Le décodage utilise SkiaSharp 3.119.4 et une implémentation C# du PNG 16 bits
pour conserver tous les bits des échantillons, filtres et entrelacement Adam7.
Les tests couvrent RGB, alpha non prémultiplié, palette opaque, orientation
PNG EXIF 6, JPEG, GIF à deux images et PNG 16 bits. Les neuf petites images
de test sont originales ; elles ne contiennent ni poids ni images utilisateur.

Limites actuelles :

- Au plus 16 millions de pixels cumulés et 64 MiB de contenu encodé dans le
  service. Le serveur HTTP garde également sa limite de requête par défaut,
  qui peut être inférieure ; le téléversement de 64 MiB n'est pas qualifié.
- Un PNG animé est refusé si le codec ne restitue pas toutes ses images ;
  les animations PNG 16 bits sont explicitement refusées.
- Les autres formats proposés par le sélecteur dépendent des codecs natifs.
  Leur couverture, leurs espaces colorimétriques et les formats haute précision
  hors PNG ne sont pas qualifiés. Le chemin général demande une sortie 8 bits.
- Les huit transformations EXIF sont implémentées ; seule l'orientation 6 est
  exercée ici. Le décodage n'est pas encore comparé intégralement au chemin
  FFmpeg/PyAV de la référence ComfyUI.
- Le sampling reste limité à SD1.5, Euler, [Heun](HEUN.md) ou [DPM++ 2M](DPMPLUSPLUS_2M.md) avec les [schedulers SD](SD_SCHEDULERS.md), un élément, de 32 à 512 pixels
  après recadrage VAE. Le masque produit n'active pas encore l'inpainting.
- L'essai natif utilise l'option d'import qui partage le traitement du bouton ;
  il ne valide pas l'interaction avec le sélecteur de fichiers de chaque OS.

La [preuve Windows/CUDA](qualification/image-input.json) décrit le parcours
réel en 12,9 secondes : import PNG, `LoadImage`, `VAEEncode`, 20 étapes avec
`denoise=0.7`, `VAEDecode`, `SaveImage` et aperçu natif 512 × 512. Les trois
composants du modèle sont sur `cuda:0`, en Float32, TF32 désactivé. L'image
reste principalement rouge malgré le prompt vert et présente une forte
saturation. Ce résultat ne qualifie ni la fidélité sémantique, ni la parité
numérique image-vers-image, ni la famille complète.

Référence fonctionnelle : [`LoadImage` dans le backend figé](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/nodes.py)
et [`VideoFromFile`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_api/latest/_input_impl/video_types.py).
