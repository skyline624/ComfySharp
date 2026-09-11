# Nœuds tensoriels et sorties d'interface

Le Host fournit cinq générateurs (`KarrasScheduler`, `ExponentialScheduler`, `PolyexponentialScheduler`, `LaplaceScheduler`, `VPScheduler`) et six traitements (`SplitSigmas`, `SplitSigmasDenoise`, `FlipSigmas`, `SetFirstSigma`, `ExtendIntermediateSigmas`, `ManualSigmas`). Leurs identifiants, paramètres et slots suivent [nodes_custom_sampler.py figé](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_extras/nodes_custom_sampler.py). Ils échangent des tenseurs natifs, conservés dans le processus Host.

Les générateurs créent actuellement des SIGMAS CPU/F32 ; choisir la distribution CUDA ne transforme pas ce calcul en calcul GPU. Les domaines mathématiques non définis produisent une erreur explicite. Le corpus et les tolérances des cinq générateurs restent décrits dans [SIGMA_SCHEDULES.md](SIGMA_SCHEDULES.md). Les opérations de découpage conservent le storage des vues ; celles qui modifient une valeur créent leur copie lorsque la source l'exige.

## Contrats locaux

Le serveur applique la distinction entre résultat interne et UI de [execution.py](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py). `POST /prompt` sélectionne les nœuds `OUTPUT_NODE`. Un filtre `partial_execution_targets` ne transforme pas un nœud de calcul en sortie. Un `class_type` absent est refusé même dans une branche inaccessible ; les erreurs de sorties indépendantes permettent aux autres sorties valides de rester exécutables.

`PreviewAny` est le vrai nœud de sortie amont : slot STRING et UI `text`. `/history/{id}` contient `outputs[nodeId].text` et `meta`, sans slots tensoriels sérialisés. Les événements `executed` ne contiennent que cette UI. La file enregistre l'historique après libération des ressources, puis émet le terminal. Les identités d'affichage actuelles sont celles des graphes statiques ; cela ne valide pas l'expansion ou les sous-graphes.

Les événements d'exécution sont adressés au client effectif, avec priorité `client_id` racine sur `extra_data.client_id`. Ce client est conservé dans le tuple de queue/historique. Les statuts de queue sont globaux ; l'interruption anonyme reste l'exception prévue par la source. Une reconnexion au même sid remplace le destinataire précédent. Les codecs binaires, la négociation complète et le détail exhaustif des erreurs/statuts amont restent à porter.

## Prévisualisation et preuve indépendante

La référence fonctionnelle est [PreviewAny.main](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_extras/nodes_preview_any.py#L23). Le laboratoire séparé a exécuté exactement son AST, extrait du commit figé par `git show`, avec Python 3.12.10/PyTorch 2.13.0 CPU. Aucune méthode C# ne produit les valeurs attendues et aucun interpréteur n'est nécessaire aux tests distribués.

Le [corpus embarqué](../tests/ComfySharp.Inference.Tests/Fixtures/preview-any.cpu.json), SHA-256 `d2febc42079e6ba57fa0ae105d7479d19d4ac2e63e8e0a7f354fb5fb7ac76954`, contient 58 cas : 44 tenseurs et 14 valeurs JSON. Chaque bloc tensoriel possède son hash de données. Les tests comparent exactement texte, espaces, sauts de ligne et slot de sortie, sans normalisation. Une seconde lecture indépendante a rejoué les 58 cas et vérifié les empreintes du fichier source et de son AST. Un test natif distinct vérifie réellement une vue non contiguë.

Le formateur couvre les tenseurs denses réels/bool F32/F64/F16/BF16/I8/U8/I16/I32/I64/BOOL, les tenseurs vides et les gradients feuilles. Il sélectionne les bords affichés avant transfert CPU, avec précision 4, seuil 1000, six éléments de bord et largeur 80. Il ne modifie pas les réglages globaux libtorch. Les tenseurs sparse, complexes, quantifiés et les représentations autograd non feuilles restent explicitement non pris en charge. Les objets de modèles et autres ressources natives nécessiteront leur adaptateur ; PreviewAny reste donc marqué `partial` dans le manifeste.

Les sources PyTorch et crédits rgthree sont conservés dans [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). La différence entre laboratoire PyTorch 2.13 et runtime du produit libtorch 2.10 est explicite. Ces fixtures ne prouvent aucune famille de modèles ni backend GPU.

## Éditeur et distribution

Les templates SIGMAS se compilent en prompts et les résultats s'affichent en texte brut natif. Conformément au [frontend figé](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/src/extensions/core/textPreviewWidgets.ts), `preview_text` et `preview_mode` ne deviennent pas des widgets persistés ; les messages sont joints par deux sauts de ligne. Le mode Markdown est à porter. Les valeurs de widgets sont encore éditées via le panneau JSON initial.

Une modification du document invalide l'aperçu. Les résultats tardifs d'une ancienne soumission ou session Host ne s'appliquent pas au document ; ils ne sont pas rejoués. Les workflows restent sauvegardables sans données d'exécution. Les exemples [texte](../examples/text-length.workflow.json) et [sigmas](../examples/sigma-preview.workflow.json) sont des parcours réellement exécutables sans modèle.

Desktop est publié à la racine, Host et sa fermeture de dépendances dans `host/`. Les verrous et répertoires de compilation sont distincts pour les variantes natives. Voir [packaging.md](packaging.md) : l'archive de développement locale n'est pas une distribution V1, les notices/SBOM complets demeurent un préalable à la publication des binaires, et macOS/MPS ainsi que Linux/CUDA restent à qualifier sur matériel réel.
