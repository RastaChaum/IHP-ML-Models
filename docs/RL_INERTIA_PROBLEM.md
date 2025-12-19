# Problème fondamental : Inertie thermique et attribution du reward

## 🚨 Diagnostic

### Le problème identifié

L'utilisateur a correctement identifié un **problème fondamental** dans notre implémentation actuelle du reward :

```
t=0min:  Action=TURN_ON, temp 18.8°C → 18.7°C (-0.1°C), reward=-0.750 ❌
         ↑ PÉNALITÉ pour une BONNE action (à cause de l'inertie)

t=30min: Action=NO_OP, temp 18.2°C → 18.3°C (+0.1°C), reward=+0.100 ✅
         ↑ RÉCOMPENSE pour une action neutre (grâce à l'action précédente)
```

### Pourquoi c'est grave

**Le modèle RL va apprendre à éviter d'allumer le chauffage** car :
1. Quand il allume → température baisse temporairement (inertie) → reward négatif
2. Quand il ne fait rien après avoir allumé → température monte → reward positif

C'est l'inverse du comportement souhaité ! Le modèle sera incité à dire "je ne fais rien" plutôt que d'agir proactivement.

## 📊 Analyse des logs

### Exemple concret d'inertie

```log
2025-12-14 20:50:30 - calculate_reward:62 - action=HeatingActionType.TURN_ON, prev_temp=18.80, curr_temp=18.70, target=20.00
2025-12-14 20:50:30 - calculate_reward:77 - Progress reward: -0.750
```

**Contexte** : Plancher chauffant avec **15-30 minutes d'inertie** thermique
- Action TURN_ON est la bonne décision (temp < target)
- Mais l'effet ne se voit qu'après 15-30 minutes
- Entre-temps, la température continue de baisser (perte thermique naturelle)
- **Résultat : reward négatif pour une action correcte**

### Succession de NO_OP avec reward positif

```log
2025-12-14 20:50:30 - calculate_reward:62 - action=HeatingActionType.NO_OP, prev_temp=18.20, curr_temp=18.20, target=20.00
2025-12-14 20:50:30 - calculate_reward:77 - Progress reward: 0.000

[...30 minutes plus tard...]

2025-12-14 20:51:00 - calculate_reward:62 - action=HeatingActionType.NO_OP, prev_temp=18.20, curr_temp=18.30, target=20.00
2025-12-14 20:51:00 - calculate_reward:77 - Progress reward: +0.100
```

**Le chauffage était allumé depuis 30min** (action précédente), donc :
- L'inertie se manifeste enfin
- La température monte
- Reward positif attribué à NO_OP au lieu de l'action TURN_ON initiale

## 🎯 Le vrai problème : Credit Assignment

C'est un problème classique en RL appelé **"Delayed Credit Assignment Problem"** :
- L'action A cause l'effet E après un délai Δt
- Le reward est attribué à l'action B qui se produit au moment de E
- **L'action A (la vraie cause) reçoit un mauvais signal**

### Aggravé par l'inertie thermique

Systèmes à forte inertie (plancher chauffant, radiateurs) :
- Délai action → effet : **15-30 minutes**
- Échantillonnage : **5 minutes**
- **3 à 6 actions entre cause et effet** → attribution erronée quasi-systématique

## 💡 Solutions proposées

### Solution 1 : Reward basé sur l'action correcte (court terme)

Au lieu de pénaliser immédiatement une baisse de température après TURN_ON :

```python
# AVANT (naïf)
progress = (curr_temp - prev_temp) * factor  # -0.1°C * 7.5 = -0.75

# APRÈS (tient compte de l'action)
if action == TURN_ON and curr_temp < target:
    # Action correcte → reward positif même si temp baisse temporairement
    progress = +0.1  # Encourage l'action correcte
elif action == NO_OP and is_heating_on:
    # Maintien du chauffage → reward neutre ou faible
    progress = 0.0
else:
    # Évaluation normale
    progress = (curr_temp - prev_temp) * factor
```

**Avantage** : Simple, encourage les actions correctes
**Inconvénient** : Ne résout pas le problème fondamental du délai

### Solution 2 : Reward différé (moyen terme)

Attribuer le reward non pas à l'action immédiate, mais à l'action **N steps avant** :

```python
# Buffer des N dernières actions (ex: N=6 pour 30min avec échantillonnage 5min)
action_buffer = deque(maxlen=6)

def calculate_delayed_reward(current_state, action_buffer):
    """Attribue le reward à l'action d'il y a 30 minutes."""
    temp_change = current_state.temp - state_30min_ago.temp
    
    # Trouver l'action causale (celle d'il y a 30min)
    causal_action = action_buffer[0]  # La plus ancienne
    
    # Calculer reward basé sur causal_action
    if causal_action == TURN_ON and temp_change > 0:
        reward = +1.0  # Bonne action, bon résultat
    elif causal_action == TURN_OFF and temp_change < 0:
        reward = +0.5
    else:
        reward = -0.5  # Mauvais timing ou mauvaise action
    
    return reward
```

**Avantage** : Attribution correcte du crédit
**Inconvénient** : Complexe, nécessite un buffer d'états/actions

### Solution 3 : Modéliser l'inertie (long terme recommandé)

Ajouter l'inertie thermique comme feature dans l'observation :

```python
@dataclass(frozen=True)
class RLObservation:
    # ... features existants ...
    
    # Nouvelles features pour modéliser l'inertie
    heating_duration_minutes: float  # Depuis combien de temps le chauffage est allumé
    estimated_temp_in_15min: float   # Prédiction basée sur modèle thermique simple
    estimated_temp_in_30min: float
```

Le modèle RL apprend alors à associer :
- `heating_duration_minutes=5` + `temp_change=-0.1` → Normal (inertie)
- `heating_duration_minutes=30` + `temp_change=+0.5` → Effet visible

**Avantage** : Le modèle apprend la dynamique naturellement
**Inconvénient** : Nécessite plus de données d'entraînement

### Solution 4 : Reward sparse (approche alternative)

Au lieu de rewards denses (à chaque step), utiliser des **rewards sparses** uniquement aux moments clés :

```python
def calculate_reward(prev_state, action, curr_state):
    # Pas de reward intermédiaire
    if not curr_state.episode_done:
        return 0.0
    
    # Reward terminal uniquement
    if target_achieved_on_time:
        return +100.0
    elif target_achieved_late:
        return +50.0 - (lateness_minutes * 0.5)
    else:
        return -100.0
```

**Avantage** : Pas d'attribution erronée pendant l'inertie
**Inconvénient** : Signal faible, apprentissage plus lent

## 🔧 Recommandation immédiate

**Combiner Solution 1 + Solution 3** :

1. **Court terme** : Modifier `HeatingRewardCalculator` pour ne pas pénaliser TURN_ON quand temp < target
2. **Moyen terme** : Ajouter `heating_duration_minutes` dans `RLObservation`
3. **Long terme** : Implémenter un modèle thermique simple pour prédire l'évolution future

## 📈 Impact attendu

Avec ces corrections :
- Le modèle apprendra à **anticiper** l'inertie
- Les actions TURN_ON recevront des rewards positifs même si temp baisse temporairement
- Les épisodes seront plus longs (pas de terminaison prématurée)
- Les rewards seront plus diversifiés (pas toujours 1.0)

## 🔍 Problème secondaire identifié : Épisodes de 30 steps

**Cause** : `RLEpisodeService.is_episode_done()` termine dès que `|temp - target| ≤ 0.3°C`

**Conséquence** :
- Épisode 1 : 30 steps pour atteindre 20°C depuis 18.8°C → done=True
- Épisodes 2-17 : 1 step chacun car température oscille autour de 20°C → done=True à chaque observation

**Solution** : Ajouter une condition de stabilité (température stable pendant N minutes) avant de terminer, comme proposé initialement mais refusé par l'utilisateur.

**Alternative** : Ne pas terminer sur cible atteinte, mais seulement sur changement de consigne ou timeout réel (ex: 3h de données historiques).
