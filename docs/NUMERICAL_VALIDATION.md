# Validation numérique et corpus

État : protocoles de référence à établir ; aucune famille de modèles acceptée.

Chaque scénario réel doit enregistrer : capacité/variante, révisions backend/frontend/port, SHA-256 et licence des poids/tokeniseurs, workflow normalisé, seeds et bruit initial, dtype de stockage/calcul, backend natif, matériel/pilote, paramètres, tenseurs de référence aux frontières et sorties médias. Les fichiers privés ou poids non redistribuables ne vont pas dans Git.

Les tokens, IDs, masques discrets, dimensions, métadonnées et structures sont comparés exactement. Les tenseurs flottants utilisent des profils par opération/famille/backend : erreur absolue, erreur relative, mesures agrégées et règles de traitement NaN/Inf. Les valeurs seuils seront choisies à partir d'une qualification du laboratoire puis verrouillées **avant** d'accepter le port ; aucune tolérance arbitraire universelle n'est déclarée ici.

Le laboratoire Python amont peut produire des fixtures séparées. Les tests distribués sont .NET et lisent les fixtures sans importer ni démarrer Python. Les tests élémentaires actuels utilisent des valeurs mathématiques contrôlées et ne constituent pas ces références de famille.

L'entraînement ajoute loss, gradients, paramètres après mise à jour, accumulation et adapter sauvegardé/rechargé. Vidéo/audio/3D ajoutent continuité, fréquence/durée, synchronisation, repères, échelles et contrôles perceptifs appropriés. Tests ressources : warmup séparé, répétitions prolongées, mesures RAM/VRAM, erreurs, annulations et changements de modèles. Une stabilisation ponctuelle de mémoire privée n'est pas une preuve d'absence de fuite.
