# Primitives IMAGE : CI du commit c4dc65d

La [campagne 34673470153](https://github.com/skyline624/ComfySharp/actions/runs/34673470153) teste exactement `c4dc65d82dc8c3348abd00a781a32e5c580a8ab8`. Les **84 nouveaux cas passent sur les trois OS**, ainsi que le parcours réel Desktop/Host des quatre primitives avec aperçu textuel. La campagne globale reste en échec à cause des contrôles numériques SD et CLIP.

| Configuration | Contrats applicatifs | Suite IMAGE dédiée | Total ordinaire et CLIP stock |
|---|---:|---:|---|
| Windows x64 CPU | 1 667 PASS | 65 PASS | 2 309 PASS |
| Linux x64 CPU | 1 667 PASS | 65 PASS | Inference ordinaire et stock non exécutés après le contrôle SD en échec |
| macOS ARM64 CPU | 1 667 PASS | 65 PASS | 2 307 PASS, deux échecs CLIP |

Les 84 nouveaux cas comprennent 28 opérations, neuf nœuds, 28 comparaisons source, 15 tests Host, trois Workflow et un Desktop. Cela donne 252 exécutions réussies entre les trois systèmes. La suite dédiée recouvre 65 de ces cas et n'est pas ajoutée une seconde fois aux tests ordinaires. Les tests applicatifs couvrent Core 750, Desktop 43, Host 259, Tokenization 540 et Workflow 75.

Le smoke utilise une vraie fenêtre native, un document compilé, le Host supervisé et PreviewAny. Sous Linux, la fenêtre fonctionne avec Xvfb. Les quatre primitives produisent un tenseur affiché sous forme de texte ; aucun fichier PNG ou aperçu raster n'est testé ici.

Le premier accès natif passe ses 194 références sur Windows et macOS. Linux passe 178 références et en échoue 16 : deux U-Net, six CFG, sept Euler et une pipeline. Par rapport à la [campagne 7d65375](case-converter-ci-7d65375.md), cinq noms deviennent en échec : U-Net `odd-rectangle`, CFG `guidance-2-10` pour les deux politiques, Euler `near-one-null` et `near-one-present`. La pipeline `two-chunks-separate` repasse. Les deux messages CLIP stock macOS restent identiques. Les graphes, fixtures et seuils numériques existants ne sont pas modifiés ; ces variations ne démontrent pas leur cause.

L'audit a vérifié 49 TRX provenant de onze artefacts, leurs compteurs et identités, et les marqueurs de smoke de la révision testée. Les 84 noms ont 86 IDs bruts : seul l'identifiant du cas Avalonia varie selon l'OS. Les digests d'archives déclarés par GitHub sont distingués des SHA recalculés sur les fichiers extraits. Les contrôles dédiés et de première utilisation ne sont pas cumulés comme des tests uniques.

Cette campagne qualifie ces primitives dans leur profil CPU/F32 borné. Elle ne valide ni CUDA/MPS, codecs, familles de modèles ni catalogue complet. Les contrôles numériques ouverts continuent de bloquer l'intégration de cette branche dans `main`.
