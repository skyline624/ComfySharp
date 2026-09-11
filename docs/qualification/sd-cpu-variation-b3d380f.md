# Qualification SD CPU : variation Windows et précondition du laboratoire

La [CI normale du commit b3d380f](https://github.com/skyline624/ComfySharp/actions/runs/34641013585)
montre désormais des échecs U-Net/CFG sur Windows également. Les sources C#,
tests, dépendances, fixtures, profil et workflow normal sont identiques à ceux
de la campagne précédente, où ces tests Windows réussissaient. Cette variation
doit rester visible ; elle ne démontre pas la cause d'une régression.

Le [relevé complet](sd-components-b3d380f.json) a été contrôlé indépendamment :
neuf artifacts téléchargés sur les neuf publiés, 28 rapports TRX, 240 payloads
de traces et 240 payloads de référence rehashés. Les compteurs concordent avec
les résultats individuels. Aucun test émis dans les TRX n'est ignoré ; les
étapes non exécutées restent explicitement absentes des validations.

| Cible | Premier accès natif | Suite complète avec comparaisons CLIP |
|---|---|---|
| Windows x64 | 172 réussis, 10 échecs | Non exécutée |
| Linux x64 | 177 réussis, 5 échecs | Non exécutée |
| macOS arm64 | 182 réussis | 1 243 réussis, 2 échecs CLIP antérieurs |

Les contrôles de premier accès sont des répétitions séparées. La suite complète
compte toujours 1 245 tests dans ce commit ; seule macOS l'a exécutée. Les
93 tests SD y passent. Sur Windows/Linux, les tests source sampling et VAE
passent, mais l'échec U-Net/CFG précède les tests agrégés, diagnostics CLI,
démarrage Desktop/Host et probe CPU : ces étapes suivantes ne sont pas validées
par cette campagne.

## Traces Windows

Les huit embeddings temporels sont exacts. La première frontière différente
est `down0` dans les huit cas ; cet intervalle contient plusieurs opérations et
n'isole aucune primitive. Les 72 frontières internes restent dans la borne,
puis 68 valeurs de sortie la dépassent :

| Cas | Valeurs hors borne | Différence absolue maximale |
|---|---:|---:|
| SD1.5 carré | 9 / 256 | `6.479024887084961e-5` |
| SD1.5 rectangle impair | 2 / 396 | `4.1484832763671875e-5` |
| SD1.5 batch, temps distincts | 9 / 640 | `6.949901580810547e-5` |
| SD2 batch, temps partagé, dimensions impaires | 48 / 504 | `2.6564300060272217e-4` |

Les six cas CFG Windows échouent aussi. Linux conserve les quatre valeurs hors
borne de sortie SD2 et les mêmes 80 hashes de traces que la campagne `ca41c5f`.
Les 80 tenseurs macOS restent exacts.

Les artifacts Windows actuels ne contiennent ni le modèle de CPU, ni la
capacité ATen effective du produit, ni les hashes des bibliothèques chargées.
L'image CI Windows Server 2025 ne permet pas de déduire ces informations. Le
manifeste source accepté rapporte AVX2 ; il appartient à une autre exécution
et ne remplace pas une mesure du processus produit actuel. La
[GroupNorm identifiée sous Linux](sd-operator-diagnostic-7549b1c.md) n'explique
pas à elle seule cette nouvelle observation Windows.

## Expérience de provenance non exécutée

Le [laboratoire natif distinct](https://github.com/skyline624/ComfySharp/actions/runs/34641013630)
du même commit a restauré et compilé correctement, puis a refusé son
environnement **avant de lancer les six processus prévus**. L'action qui
installe Python ajoute son répertoire de bibliothèques à `LD_LIBRARY_PATH` ;
la précondition initiale exigeait une variable vide.

Le [relevé de cet arrêt](sd-native-precondition-b3d380f.json) confirme zéro
processus enfant, zéro trace et zéro artifact. Il n'existe donc aucune
observation de substitution de libtorch, de neutralité des observateurs ou de
comparaison numérique pour cette tentative.

La correction du laboratoire admet uniquement un environnement vide ou le
répertoire `lib` vérifié de l'interpréteur sélectionné, après identification
des bibliothèques Python et vérification de l'absence de candidats libtorch,
binding et OpenMP dans ce répertoire. Ce chemin reste identique pour les
groupes de contrôle. Le groupe expérimental ajoute le répertoire des
bibliothèques natives à comparer ; les hashes effectivement chargés restent
obligatoires. Les chemins supplémentaires inconnus et les préchargements
préexistants sont refusés. Un prochain lancement devra vérifier cette
intégration sur Linux avant toute conclusion numérique.

Les modèles préentraînés, workflows complets, dimensions stock et GPU restent
non qualifiés. Aucune référence, tolérance ou bibliothèque du produit n'est
remplacée pour faire disparaître ces échecs.
