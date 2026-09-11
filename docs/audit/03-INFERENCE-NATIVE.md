> Audit statique antérieur à l’implémentation. Les choix définitifs et le périmètre V1 sont fixés par [le plan approuvé](../MIGRATION.md). Les observations ci-dessous ne sont pas des résultats de tests du port C#.

# Conversion du moteur d’inférence vers C# et bibliothèques natives

Audit du 11 septembre 2026. Dépôt : `le clone de référence`. Révision étudiée : `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`.

La cible demandée est une application sans Python, avec logique et interface réécrites en C#, et bibliothèques natives autorisées. Ce document traite du moteur d’inférence, des modèles, du chargement, de la tokenisation et de la gestion mémoire. La conversion de l’interface et du moteur de workflows est traitée dans les autres rapports.

Il s’agit d’un **audit statique** : aucun modèle téléchargé, aucune dépendance installée, aucune exécution ComfyUI et aucun benchmark d’inférence. Les propositions de performance et de compatibilité restent à vérifier par prototype. La machine inventoriée dans l’audit global possède une **NVIDIA RTX 3090, 24 576 MiB de VRAM, pilote 610.88** ; cette observation ne constitue pas un test des runtimes proposés.

## 1. Conclusion de faisabilité

Le moteur d’inférence représente le chantier principal. Une application C# sans Python exige de réimplémenter les architectures de modèles, leurs prétraitements, les mécanismes de chargement et les politiques mémoire. Le remplacement du serveur ou de la boucle de workflows ne suffit pas à produire un ComfyUI autonome.

Pour conserver les checkpoints existants et construire progressivement une extensibilité proche du projet, le candidat à prototyper en premier est **C# + TorchSharp/libtorch**, complété par des bindings natifs ciblés lorsque nécessaires. Cette préférence est une recommandation d’architecture, pas une preuve de couverture complète des opérations.

**C# + ONNX Runtime** est une autre voie sans Python à l’exécution, particulièrement adaptée à un catalogue de modèles préparés et limité. Elle nécessite cependant de fournir des graphes appropriés : ONNX Runtime ne reconstitue pas les architectures ComfyUI à partir de fichiers safetensors. Le projet doit aussi résoudre la préparation des nouveaux modèles sans imposer un outil Python à l’utilisateur.

Une conversion de tout le catalogue actuel, avec tous les samplers, les formats quantifiés, les hooks et les comportements sur faible VRAM, est un programme de développement important. Elle ne doit pas être présentée comme une traduction automatique du code Python.

## 2. Taille et diversité constatées

Le répertoire `comfy/` contient **312 fichiers Python, 120 027 lignes physiques**, dont 101 196 lignes non vides. Ces mesures proviennent d’une lecture des fichiers, sans import du projet.

Le registre de modèles déclare **102 classes de configuration**, variantes comprises. Ce nombre ne signifie pas 102 architectures indépendantes. Les listes de sampling exposent **45 noms de samplers**, variantes et alias compris, et **9 schedulers**. Les noms k-diffusion sont définis dans [samplers.py](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/samplers.py#L971), les trois noms supplémentaires dans [SAMPLER_NAMES](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/samplers.py#L1356), et les schedulers dans [SCHEDULER_HANDLERS](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/samplers.py#L1365).

| Domaine | Exemples présents dans la source | Références représentatives |
|---|---|---|
| Diffusion image UNet | SD1.5, SD2, SDXL, refiner, instruct-pix2pix, upscaler, dérivés SSD/KOALA | [SD1.5](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/supported_models.py#L49), [SD2](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/supported_models.py#L95), [SDXL](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/supported_models.py#L202), [instruct-pix2pix](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/supported_models.py#L505) |
| Génération en cascade | Stable Cascade B/C | [Stable Cascade](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/supported_models.py#L439) |
| Diffusion image DiT/flow | SD3, AuraFlow, PixArt, HunyuanDiT, Flux/Flux2, Lumina, ZImage, PixelDiT | [SD3](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/supported_models.py#L549), [Flux](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/supported_models.py#L735), [Flux2](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/supported_models.py#L796), [ZImage](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/supported_models.py#L1203) |
| Autres modèles image | HiDream, Chroma, Omnigen2, Ideogram4, Krea2, QwenImage, JoyImage, LongCat, Ernie | [HiDream](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/supported_models.py#L1651), [QwenImage](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/supported_models.py#L2024), [Ernie](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/supported_models.py#L2355) |
| Vidéo et audiovisuel | SVD, Mochi, LTXV/LTXAV, HunyuanVideo, Cosmos, Wan et ses variantes, CogVideoX | [SVD](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/supported_models.py#L314), [LTXV](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/supported_models.py#L914), [Wan](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/supported_models.py#L1309), [CogVideoX](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/supported_models.py#L2436) |
| Audio et musique | StableAudio/3, ACEStep/1.5, MiniMaxMusic3 | [StableAudio](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/supported_models.py#L585), [ACEStep](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/supported_models.py#L1845), [MiniMaxMusic3](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/supported_models.py#L2268) |
| Géométrie et 3D | SV3D, Zero123, Trellis2, Hunyuan3D, TripoSplat | [SV3D](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/supported_models.py#L345), [Trellis2](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/supported_models.py#L1482), [Hunyuan3D](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/supported_models.py#L1569) |
| Analyse visuelle | RT-DETR, DepthAnything3, SAM3/SAM3.1 | [RT-DETR](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/supported_models.py#L2323), [DepthAnything3](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/supported_models.py#L2338), [SAM3](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/supported_models.py#L2385) |
| Encodeurs texte | CLIP-L/G, T5/UMT5, Llama, variantes Qwen/Mistral/Gemma, encodeurs multimodaux | [Détection des encodeurs](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/sd.py#L1574), [chargement](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/sd.py#L1724) |
| Adaptation des poids | LoRA, LoHa, LoKr, GLoRA, OFT, BOFT, DoRA et injection bypass | [Registre des adapters](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/weight_adapter/__init__.py#L1), [chargement LoRA](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/lora.py#L37), [hooks bypass](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/weight_adapter/bypass.py#L107) |

Les architectures sont implémentées dans `comfy/ldm/`. Chaque famille apporte des conventions de tenseurs, conditionings, tokenisation, paramètres et VAE. Une API commune au-dessus des réseaux ne supprime pas les différences entre leurs calculs.

## 3. Chaîne de chargement à reconstruire

### 3.1 Lecture des poids et métadonnées

[utils.load_safetensors](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/utils.py#L96) lit les safetensors avec mappage mémoire via `comfy_aimdo.model_mmap`, valide les offsets, tailles, shapes et types numériques, et construit les tensors associés. [load_torch_file](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/utils.py#L158) distingue `.safetensors/.sft` du chargement torch ; les autres formats passent par [`torch.load(..., weights_only=True)`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/utils.py#L191).

Un lecteur safetensors managed ou natif est une cible pertinente pour C#. Il doit contrôler les dimensions, tailles, offsets, limites de fichier et métadonnées avant d’exposer les buffers au runtime. La compatibilité `.ckpt/.pt` n’est pas obtenue automatiquement avec ce lecteur, ni avec un nom d’API ressemblant à `torch.load` côté C#.

### 3.2 Détection de l’architecture

[detect_unet_config](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/model_detection.py#L44), [model_config_from_unet_config](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/model_detection.py#L1286), [model_config_from_unet](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/model_detection.py#L1294) et [unet_prefix_from_state_dict](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/model_detection.py#L1310) déduisent les configurations depuis les clés et les shapes du state dict. ComfyUI accepte ainsi des fichiers sans que l’utilisateur sélectionne toujours l’architecture explicitement.

La cible C# doit choisir entre reproduire cette détection pour les formats acceptés et demander un manifeste explicite. Une détection partielle ne doit pas faire croire à la prise en charge de variantes dont les calculs ou les poids ne sont pas compatibles.

### 3.3 Normalisation et construction du réseau

[load_state_dict_guess_config](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/sd.py#L2159) coordonne la détection, les anciens formats quantifiés, la sélection des dtypes, la création du modèle, le patcher, le VAE et les encodeurs. [load_diffusion_model_state_dict](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/sd.py#L2262) gère aussi les diffusion models séparés et les variantes Diffusers.

Les conventions de nommage font partie de la compatibilité : [SD1.5 transforme les clés CLIP](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/supported_models.py#L66), tandis que [SDXL traite deux espaces CLIP](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/supported_models.py#L241).

[BaseModel](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/model_base.py#L166) construit le réseau avec un jeu d’opérations sélectionné. Le checkpoint n’inclut pas à lui seul l’implémentation Python du réseau : il faut recréer cette structure en C# ou fournir un graphe natif préparé correspondant exactement à la variante.

### 3.4 Placement et cycle de vie

[sd.py](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/sd.py#L2205) crée un patcher, choisit les devices de chargement et d’offload, puis affecte les poids. [load_models_gpu](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/model_management.py#L925) arbitre ensuite la mise en GPU. La lecture des matrices est donc une étape d’une chaîne plus large.

Pour le premier livrable, accepter un format précis : **checkpoint SD1.5 dense safetensors contenant UNet, CLIP-L et VAE**, ou composants séparés avec manifeste. Tout format supplémentaire doit posséder ses propres fixtures de chargement et d’inférence.

## 4. Tokenisation et encodage des prompts

[SDTokenizer](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/sd1_clip.py#L486) ajoute au tokenizer sous-jacent la segmentation, les tokens spéciaux, le padding, les embeddings textuels et la pondération. La syntaxe de parenthèses et de poids est dans [parse_parentheses](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/sd1_clip.py#L320) et [token_weights](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/sd1_clip.py#L348). Le chargement des embeddings est dans [load_embed](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/sd1_clip.py#L415), la tokenisation pondérée dans [tokenize_with_weights](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/sd1_clip.py#L572), et l’application des poids aux embeddings dans [ClipTokenWeightEncoder](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/sd1_clip.py#L27).

Les familles ne partagent pas toutes les mêmes règles :

- [Flux T5](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/text_encoders/flux.py#L11) utilise une longueur minimale de 256 et un padding spécifique ; le même fichier introduit [Mistral](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/text_encoders/flux.py#L85) et [Qwen](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/text_encoders/flux.py#L143).
- [SD3Tokenizer](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/text_encoders/sd3_clip.py#L41) combine CLIP-L, CLIP-G et T5.
- [SPieceTokenizer](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/text_encoders/spiece_tokenizer.py#L13) utilise SentencePiece.
- Un [BPE Python autonome](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/text_encoders/bpe_tokenizer.py#L117) existe aussi, avec lecture de [tokenizer.json](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/text_encoders/bpe_tokenizer.py#L228) et [Tekken](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/text_encoders/bpe_tokenizer.py#L260).

Une bibliothèque de tokenisation C# ou native apporte des primitives, mais ne garantit pas les mêmes sorties. Le contrat de compatibilité doit couvrir exactement les IDs, tokens spéciaux, normalisation Unicode, découpage, padding, masques et pondérations. Il faut ensuite comparer les embeddings produits. Une simple affirmation « compatible CLIP » serait insuffisante.

Pour le MVP SD1.5, un tokenizer BPE fidèle avec les assets correspondants suffit. Les embeddings personnalisés et les encodeurs T5/LLM pourront être ajoutés dans des jalons distincts.

## 5. Samplers, bruit, conditioning et latents

### 5.1 Reproductibilité du bruit

[prepare_noise_inner](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/sample.py#L9) génère le bruit avec `torch.randn` **sur CPU en float32**, puis convertit vers le dtype du latent. [prepare_noise](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/sample.py#L22) initialise le générateur depuis la seed. La logique des index de batch consomme aussi une séquence déterminée de tirages.

Remplacer ce mécanisme par `System.Random` ne reproduirait pas les mêmes latents avec le même nombre affiché. La cible doit soit reproduire le générateur et son ordre d’utilisation, soit documenter sa propre reproductibilité. Le corpus de validation doit fournir du bruit initial enregistré pour séparer les différences de RNG de celles du calcul.

### 5.2 Types de prédiction et conduite du sampling

[model_sampling.py](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/model_sampling.py#L30) distingue notamment EPS, V-prediction, EDM et flow. [model_base.model_sampling](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/model_base.py#L113) sélectionne le comportement selon la configuration.

[BaseModel._apply_model](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/model_base.py#L214) transforme l’entrée à partir de sigma, convertit le timestep et combine les conditionings. Le forward du réseau seul ne représente pas la totalité de la prédiction ComfyUI.

[cfg_function](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/samplers.py#L592) porte le CFG et ses extensions ; [KSamplerX0Inpaint](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/samplers.py#L630) gère l’inpainting. Les [batchs conditionnés](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/samplers.py#L221) et leur [variante multi-GPU](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/samplers.py#L358) ajoutent une autre couche de comportement.

[Euler](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/k_diffusion/sampling.py#L190) et [DPM++ 2M](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/k_diffusion/sampling.py#L864) sont des candidats progressifs. Les variantes stochastiques font intervenir les [Brownian trees](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/k_diffusion/sampling.py#L91) et [BrownianTreeNoiseSampler](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/k_diffusion/sampling.py#L127), avec `torchsde`. Certains calculs de schedulers utilisent SciPy.

Chaque sampler accepté doit être validé avec son scheduler, son type de prédiction, ses conditions initiales et finales et son format latent. Le nombre de noms exposés ne mesure pas à lui seul l’effort : plusieurs variantes partagent un cœur, tandis que certaines introduisent une gestion du bruit ou des callbacks différente.

### 5.3 Formats latents et VAE

Tous les modèles ne partagent pas le layout `[N,4,H/8,W/8]`. [SD1.5](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/latent_formats.py#L25) utilise l’échelle `0.18215`, [SDXL](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/latent_formats.py#L37) `0.13025`. [Flux](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/latent_formats.py#L165) ajoute 16 canaux et un shift/scale ; [Flux2](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/latent_formats.py#L197) déclare 128 canaux et une organisation différente. L’audio, la vidéo et les flux multiples demandent leurs propres layouts.

[VAE](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/sd.py#L487) comprend un important sélecteur d’architectures. Le [décodage en tuiles 2D](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/sd.py#L1134), le [décodage 3D](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/sd.py#L1158), le [décodage général](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/sd.py#L1221) et l’[encodage](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/sd.py#L1362) adaptent les batchs et la mémoire. Les reprises après manque de mémoire se trouvent dans les voies [decode](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/sd.py#L1258) et [encode](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/sd.py#L1395).

La vidéo ajoute des contraintes temporelles et des caches locaux à certaines architectures. Un VAE image générique ne suffit pas à couvrir ces familles.

## 6. Gestion mémoire, patching et kernels

### 6.1 Politique mémoire

[model_management.py](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/model_management.py#L45) possède des politiques VRAM, du calcul de budget, du chargement partiel et de l’éviction. Les points d’entrée majeurs sont [free_memory](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/model_management.py#L879), [load_models_gpu](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/model_management.py#L925), [unet_dtype](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/model_management.py#L1122) et [unet_manual_cast](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/model_management.py#L1175).

Le même module gère des [streams de transfert](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/model_management.py#L1475), des [buffers de conversion](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/model_management.py#L1395), leur [variante aimdo](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/model_management.py#L1427) et la [mémoire hôte épinglée](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/model_management.py#L1624). Les capacités matérielles sont sondées pour [FP8](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/model_management.py#L1971), [NVFP4](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/model_management.py#L1995) et [MXFP8](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/model_management.py#L2005).

Pour C#, la gestion des tensors natifs doit être explicite : ownership, durée de vie, libération déterministe, partage des poids, clôture des sessions et synchronisation avant réutilisation des buffers. La collecte des petits wrappers managed ne constitue pas une stratégie suffisante de contrôle de la VRAM.

Une première politique par composant — encodeur, réseau de diffusion, VAE — est plus simple à vérifier. La parité avec l’offload dynamique de ComfyUI représente un jalon séparé.

### 6.2 Patching et adaptation

`ModelPatcher` couvre plusieurs mécanismes :

| Mécanisme | Références |
|---|---|
| Clones et partage de poids | [clone](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/model_patcher.py#L430) |
| Remplacements CFG, UNet, attention et blocs | [API de patching](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/model_patcher.py#L638) |
| Patches de poids et application par device | [add_patches](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/model_patcher.py#L842), [patch_weight_to_device](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/model_patcher.py#L899) |
| Chargement et retrait partiels | [load](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/model_patcher.py#L982), [partially_unload](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/model_patcher.py#L1171), [partially_load](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/model_patcher.py#L1256) |
| Callbacks, injections et hooks temporels | [callbacks](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/model_patcher.py#L1319), [inject_model](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/model_patcher.py#L1419), [keyframes](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/model_patcher.py#L1467) |
| Gestion dynamique spécialisée | [ModelPatcherDynamic](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/model_patcher.py#L1749) |

Le support d’une LoRA standard ne représente donc qu’une partie de la parité. [lora.load_lora](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/lora.py#L37) sélectionne plusieurs adapters, et [calculate_weight](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/lora.py#L451) applique strengths, offsets, compositions et transformations.

### 6.3 Préchargement et optimisation native

[model_prefetch.prefetch_queue_pop](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/model_prefetch.py#L140) combine préchargement, états `model_vbar`, streams et [capture/rejeu CUDA graphs](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/model_prefetch.py#L230). [read_tensor_file_slice_into](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/memory_management.py#L18) prend en charge la lecture de portions de tensors, avec une [voie vers le device](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/memory_management.py#L59).

Ces optimisations dépendent de la durée de vie des buffers et du runtime. Leur réutilisation doit être étudiée au niveau de l’ABI et des interfaces natives réelles, sans supposer que les paquets Python exposent déjà une API C exploitable par P/Invoke.

[pick_operations](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/ops.py#L1694) sélectionne les opérations selon les types et la quantification : initialisation désactivée, conversion à l’utilisation, kernels FP8, opérations mixtes, cuBLAS.

La [sélection d’attention](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/ldm/modules/attention.py#L857) couvre PyTorch, xformers, SageAttention, FlashAttention, Comfy Kitchen et des implémentations de repli. Le support d’un GPU par un runtime ne prouve pas celui de chacun de ces chemins. Pour le MVP, une attention dense correcte et mesurée est un meilleur critère de décision que la présence nominale de plusieurs backends.

## 7. Quantification : utiliser le code comme référence

[QUANTIZATION.md](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/QUANTIZATION.md#L33) explique le principe des tensors quantifiés, du dispatch et des opérations mixtes. Certaines descriptions ont toutefois pris du retard : le document décrit [`MixedPrecisionOps` et `layer_quant_config`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/QUANTIZATION.md#L83), tandis que le code expose [mixed_precision_ops](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/ops.py#L1300) et sélectionne `model_config.quant_config` dans [pick_operations](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/ops.py#L1695). Sa [table de formats](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/QUANTIZATION.md#L126) ne constitue pas une spécification exhaustive du code actuel.

Les formats enregistrés sont définis dans [QUANT_ALGOS](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/quant_ops.py#L211) :

| Format | Particularité constatée |
|---|---|
| `float8_e4m3fn` | Stockage FP8, scales des poids et des entrées |
| `float8_e5m2` | Autre format FP8, scales des poids et des entrées |
| `nvfp4` | Stockage uint8, plusieurs scales, groupes de 16 |
| `mxfp8` | Enregistrement conditionnel, groupes de 32 |
| `int8_tensorwise` | Quantification entière des poids avec scale |
| `convrot_w4a4` | Layout dédié, stockage int8 |
| `asym_w4a8_int8` | Layout dédié, stockage int8 |

Le code importe les abstractions [`comfy_kitchen.tensor`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/quant_ops.py#L8). Les subclasses de tensors, layouts, transformations et règles de dispatch ne deviennent pas utilisables en C# simplement parce que les calculs finaux sont natifs. Il faut des bindings adéquats ou réimplémenter les recettes compatibles.

Déquantifier vers FP16/FP32 peut fournir une première voie fonctionnelle pour un format précisément implémenté. Cela augmente la mémoire et ne reproduit ni les performances ni nécessairement les mêmes résultats numériques. Sur la RTX 3090 inventoriée, le prototype doit commencer par des poids denses et valider séparément toute quantification. Les tests FP8/FP4 demandent aussi une matrice de capacités propre au matériel et au runtime.

## 8. Deux voies d’inférence sans Python

| Critère | C# + TorchSharp/libtorch | C# + ONNX Runtime |
|---|---|---|
| Architecture du modèle | Modules et forwards réécrits en C#, alimentés par les poids | Graphes ONNX préparés pour chaque architecture et configuration |
| Safetensors existants | Lecteur et mapping des poids vers les modules recréés | Poids incorporés dans un graphe ou associés à un graphe compatible |
| Sampling | Algorithmes C# manipulant des tensors natifs | Algorithmes C# pilotant des sessions et tensors sur device |
| LoRA et patches | Approche plus proche de la source dynamique, à réimplémenter | Dépend du graphe, de la possibilité de modifier les poids et des points d’extension prévus |
| Offload | Politique à construire sur les fonctions natives disponibles | Politique à construire autour des sessions, allocateurs et entrées/sorties |
| Risque principal | Couverture des API, kernels spécialisés, ABI, synchronisation et durée de vie | Préparation/export, opérateurs, shapes, providers et modifications dynamiques |
| Objectif initial pertinent | Moteur compatible construit progressivement | Application à catalogue délimité de graphes préparés |

ONNX n’interdit pas les LoRA ou les opérations personnalisées. Elles demandent une conception et une validation adaptées ; elles ne doivent pas être promises pour les modifications arbitraires actuellement permises par les hooks Python.

Le choix sans Python doit aussi considérer la chaîne fournie pour préparer de nouveaux modèles. Si chaque utilisateur doit exporter ses checkpoints au moyen d’un outil Python, cette étape reste dépendante de Python. Des ONNX préparés peuvent éviter Python sur la machine de l’utilisateur, mais imposent un catalogue, une distribution des artefacts et une maintenance des variantes.

### Écart de versions à vérifier

Le dépôt ne fixe pas la version de [`torch`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/requirements.txt#L4), mais fixe [Comfy Kitchen et aimdo](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/requirements.txt#L25). Le [README ComfyUI](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/README.md#L195) décrit une installation NVIDIA récente avec CUDA 13.0 ou supérieur. Les [notes officielles TorchSharp](https://github.com/dotnet/TorchSharp/blob/main/RELEASENOTES.md) consultées pour l’audit global indiquent libtorch 2.10 et CUDA 12.8 pour la version 0.106.0 ; les notes 0.107 concernent aussi le passage à .NET 8.

Cet écart doit être testé. Il ne démontre pas une incompatibilité totale avec les opérations nécessaires à SD1.5, et ne permet pas de garantir le catalogue ComfyUI récent. Il faut fixer un couple précis TorchSharp/libtorch, vérifier la disponibilité des API utilisées, démarrer sur la RTX 3090 inventoriée et mesurer le parcours complet. Les indications de README et les notes de versions peuvent diverger ; conserver les versions et preuves retenues dans le dossier de décision.

## 9. MVP précis proposé

Le premier jalon d’inférence doit réaliser :

1. Chargement d’un checkpoint **SD1.5 dense safetensors identifié**, avec UNet, CLIP-L et VAE.
2. Tokenisation CLIP fidèle et syntaxe de pondération explicitement retenue.
3. Conditioning positif et négatif, bruit initial, CFG, sampler **Euler**, scheduler **normal**, `denoise=1` pour le premier parcours validé.
4. Génération texte vers image **512 × 512, batch 1**, avec format latent SD1.5 et conversion VAE correcte.
5. Voie **CPU float32** de référence et voie **NVIDIA/CUDA sur RTX 3090** avec dtype fixé, toutes deux validées par le même corpus.
6. Politique simple de transfert par composant, annulation entre étapes et libération déterministe des ressources.

Ce jalon constitue le cœur natif à intégrer à l’interface C# et au moteur de workflows décrits dans le plan global. Il ne revendique pas encore quantification, LoRA, vidéo, audio, 3D, multi-GPU, hooks arbitraires ou offload dynamique complexe. Ces fonctions doivent être absentes de la liste des capacités acceptées tant qu’elles ne sont pas implémentées et validées.

L’ordre d’extension recommandé est : img2img et inpainting ; LoRA standard ; DPM++ 2M/Karras ; VAE par tuiles ; SDXL et ses deux encodeurs CLIP ; une seule famille DiT sélectionnée ; puis quantification et familles temporelles selon les besoins réels.

La source contient aussi des [adapters d’entraînement](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/weight_adapter/base.py#L156) et des variantes trainables dans les modules LoRA, LoHa, LoKr et OFT. Une conversion intégrale doit distinguer leur prise en charge de celle de l’inférence. Le MVP proposé ne comprend pas l’entraînement.

## 10. Validation et critères de passage

Le référentiel doit fixer le SHA ComfyUI, les hashes des poids et tokenizers, les paramètres, les versions des runtimes, le matériel, le backend d’attention et les dtypes. Les fixtures doivent être obtenues depuis un environnement de référence maîtrisé ou un corpus existant de confiance. Cet audit n’en a pas produit et ne prouve donc aucune parité numérique à ce stade.

| Niveau | Critère à vérifier |
|---|---|
| Lecture et mapping | Mêmes tensors, shapes, dtypes et métadonnées ; erreur claire sur format ou clés incompatibles |
| Tokenisation | Égalité exacte des IDs, tokens spéciaux, masques et poids sur accents, Unicode, ponctuation, parenthèses, prompts longs et vides |
| Encodeur | Comparaison des embeddings intermédiaires et des sorties utilisées par le conditioning |
| Réseau de diffusion | Une prédiction UNet à entrées fixées, puis plusieurs timesteps et dimensions acceptées |
| VAE | Encodage et décodage de fixtures, contrôle des scales, layouts, NaN/Inf et conversions pixels |
| Sampling | Bruit initial enregistré, comparaison étape par étape des latents et conditions finales |
| Reproductibilité | Séparer reproductibilité interne C# et concordance avec la référence ComfyUI ; tester le générateur et l’ordre des tirages |
| Ressources | Générations répétées, changement de checkpoint, annulation, contrôle de la mémoire native, RAM/VRAM et récupération après erreur |
| Distribution | Démarrage et génération sur une installation de test sans interpréteur Python ni paquet Python requis |

Les tolérances absolues et relatives doivent être fixées après mesure sur l’environnement de référence. Une seule comparaison visuelle de l’image finale masque les erreurs intermédiaires. Inversement, exiger une identité bit à bit entre matériels et kernels différents sans étude numérique serait une promesse injustifiée.

Le passage du prototype au développement du catalogue dépend de trois résultats : fonctionnement intégral sur le modèle retenu, écarts numériques expliqués et acceptés, et comportement mémoire stable. Les mesures de temps et de VRAM serviront ensuite à décider si les kernels et politiques supplémentaires justifient leur coût de portage.
