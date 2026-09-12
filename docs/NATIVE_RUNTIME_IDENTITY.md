# Identité du runtime natif

`NativeRuntimeBuildInfo.Read()` interroge les API `at::get_cpu_capability()` et
`at::show_config()` du libtorch chargé. Le résultat distingue le mode de calcul
effectif des instructions simplement disponibles sur le processeur. La lecture
ne sélectionne aucun mode CPU et ne modifie pas le nombre de threads.

Le bridge `ComfySharp.Native` vérifie l'ABI libtorch 2.10.0. Il garde ses chaînes
UTF-8 dans un stockage local au thread ; le code C# les copie avant de retourner.
Les exceptions natives sont traduites en erreurs C#. Une bibliothèque absente,
ancienne ou incompatible produit un diagnostic explicite, sans déduire une
identité depuis le matériel.

Après compilation du bridge avec le SDK libtorch correspondant :

```powershell
dotnet build tools/ComfySharp.RuntimeProbe -c Release -p:NativeGeneratorBridge=<chemin-absolu-du-bridge>
dotnet run --no-build -c Release --project tools/ComfySharp.RuntimeProbe -- runtime-info
```

La commande émet du JSON contenant le mode CPU, la configuration libtorch, les
threads, les contrôles d'environnement relatifs au calcul, les hashes des images
natives chargées et de TorchSharp. Les chemins absolus des bibliothèques ne sont
pas exposés. Les images chargées depuis des chemins différents restent des
entrées distinctes, même si leurs noms sont identiques. `identitySha256` identifie
ce relevé complet ; il ne désigne pas encore un profil numérique qualifié.

Les premiers contrôles Windows CPU ont vérifié :

- une erreur explicite sans bridge ;
- cinq tests de copie des chaînes, d'ABI et d'erreurs ;
- une identité identique dans deux processus indépendants ;
- `AVX2` en mode automatique et `DEFAULT` avec ce mode demandé dans un processus
  séparé, sans modification permanente de l'environnement.

La campagne Linux compile également le bridge avec CMake depuis les headers du
SDK source épinglé. Le candidat doit annoncer exactement le même mode CPU et la
même configuration de build que la source, en plus des comparaisons existantes
de fichiers et de tenseurs. Ces contrôles préparent la sélection de profils
explicites ; ils ne remplacent pas les références historiques et ne qualifient
ni les modèles préentraînés, ni CUDA, ni MPS.

La [preuve Windows](qualification/native-runtime-identity.json) identifie le
binaire compilé, les sources et les contrôles exécutés. La solution Release
compile sans avertissement ; les 870 tests ordinaires d'inférence passent
localement, dont les cinq nouveaux tests de l'interface native.

La [campagne Linux d9c386b](https://github.com/skyline624/ComfySharp/actions/runs/34716664659)
valide ensuite la compilation CMake et le chemin de chargement `$ORIGIN`. Le
bundle candidat et la source annoncent exactement le même build et le mode
effectif `AVX2` ; les 282 captures d'entraînement restent exactes. La suite
ordinaire passe 855 tests sur 870 avec le candidat, contre 848 avec le bundle
actuel. Les 15 échecs historiques du candidat restent ouverts.

La [CI normale correspondante](https://github.com/skyline624/ComfySharp/actions/runs/34716664668)
échoue encore sur les gradients d'entraînement Windows, les comparaisons
d'entraînement Linux et CLIP aux dimensions complètes macOS. L'identification
du runtime est désormais opérationnelle ; la distribution du bundle natif et
la sélection de profils sources explicitement identifiés sont les étapes suivantes.
