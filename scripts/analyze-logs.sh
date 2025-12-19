#!/bin/bash
# Script d'analyse des logs RL pour diagnostiquer les problèmes d'entraînement

set -e

# Configuration
LOG_DIR="${LOG_DIR:-/data/logs}"
DEBUG_LOG="$LOG_DIR/ihp_ml_debug.log"
MAIN_LOG="$LOG_DIR/ihp_ml.log"

echo "=========================================="
echo "IHP ML Models - Log Analysis"
echo "=========================================="
echo ""

# Vérifier que les logs existent
if [ ! -f "$DEBUG_LOG" ]; then
    echo "❌ Debug log not found: $DEBUG_LOG"
    exit 1
fi

if [ ! -f "$MAIN_LOG" ]; then
    echo "⚠️  Main log not found: $MAIN_LOG (continuing with debug log only)"
fi

# 1. Statistiques globales
echo "📊 Global Statistics"
echo "===================="
echo ""

# Nombre d'épisodes
EPISODE_COUNT=$(grep -c "Episode.*finished" "$MAIN_LOG" 2>/dev/null || echo "0")
echo "Total episodes: $EPISODE_COUNT"

# Nombre d'expériences créées
EXPERIENCES=$(grep "Created.*RL experiences" "$DEBUG_LOG" | tail -1 | grep -oP '\d+(?= RL experiences)')
echo "Total RL experiences: ${EXPERIENCES:-N/A}"

# Nombre d'observations échantillonnées
OBSERVATIONS=$(grep "Sampled.*observations from history" "$DEBUG_LOG" | tail -1 | grep -oP '\d+(?= observations)')
echo "Sampled observations: ${OBSERVATIONS:-N/A}"

echo ""

# 2. Statistiques par épisode
if [ "$EPISODE_COUNT" -gt 0 ]; then
    echo "📈 Episode Statistics"
    echo "====================="
    echo ""
    
    # Durées des épisodes
    echo "Episode lengths:"
    grep "Episode.*finished" "$MAIN_LOG" | grep -oP 'length=\K\d+' | \
        awk '{sum+=$1; count++; lengths[count]=$1} END {
            printf "  Min: %d steps\n", lengths[1]
            printf "  Max: %d steps\n", lengths[count]
            printf "  Avg: %.1f steps\n", sum/count
        }'
    
    echo ""
    
    # Rewards des épisodes
    echo "Episode rewards:"
    grep "Episode.*finished" "$MAIN_LOG" | grep -oP 'reward=\K[0-9.]+' | \
        awk '{sum+=$1; count++; if(NR==1 || $1<min) min=$1; if(NR==1 || $1>max) max=$1} END {
            printf "  Min: %.3f\n", min
            printf "  Max: %.3f\n", max
            printf "  Avg: %.3f\n", sum/count
        }'
    
    echo ""
fi

# 3. Analyse des rewards par step
echo "💰 Reward Analysis"
echo "=================="
echo ""

# Extraire tous les rewards des steps
STEP_REWARDS=$(grep "Step.*reward=" "$DEBUG_LOG" | grep -oP 'reward=\K[0-9.-]+')

if [ -n "$STEP_REWARDS" ]; then
    echo "$STEP_REWARDS" | awk '
    {
        sum+=$1
        count++
        if($1 == 0) zeros++
        if($1 > 0) positives++
        if($1 < 0) negatives++
        if(NR==1 || $1<min) min=$1
        if(NR==1 || $1>max) max=$1
    }
    END {
        printf "Step rewards distribution:\n"
        printf "  Total steps: %d\n", count
        printf "  Zero rewards: %d (%.1f%%)\n", zeros, 100*zeros/count
        printf "  Positive rewards: %d (%.1f%%)\n", positives, 100*positives/count
        printf "  Negative rewards: %d (%.1f%%)\n", negatives, 100*negatives/count
        printf "  Min: %.3f\n", min
        printf "  Max: %.3f\n", max
        printf "  Avg: %.4f\n", sum/count
    }'
else
    echo "⚠️  No step rewards found in logs"
fi

echo ""

# 4. Détection de fins d'épisodes
echo "🏁 Episode Termination Analysis"
echo "================================"
echo ""

# Compter les raisons de fin d'épisode
TERMINATED_EPISODES=$(grep -c "Episode.*ended" "$DEBUG_LOG" 2>/dev/null || echo "0")
echo "Episodes terminated by episode service: $TERMINATED_EPISODES"

if [ "$TERMINATED_EPISODES" -gt 0 ]; then
    echo ""
    echo "Last 5 episode terminations:"
    grep "Episode.*ended" "$DEBUG_LOG" | tail -5 | while read line; do
        EPISODE_NUM=$(echo "$line" | grep -oP 'Episode \K\d+')
        EXP_COUNT=$(echo "$line" | grep -oP '\d+(?= experiences)')
        TEMP=$(echo "$line" | grep -oP 'temp=\K[0-9.]+')
        TARGET=$(echo "$line" | grep -oP 'target=\K[0-9.]+')
        echo "  Episode #$EPISODE_NUM: $EXP_COUNT experiences, temp=${TEMP}°C (target=${TARGET}°C)"
    done
fi

echo ""

# 5. Vérifier les problèmes potentiels
echo "⚠️  Potential Issues"
echo "===================="
echo ""

ISSUES=0

# Vérifier si tous les épisodes ont la même longueur
if [ "$EPISODE_COUNT" -gt 1 ]; then
    UNIQUE_LENGTHS=$(grep "Episode.*finished" "$MAIN_LOG" | grep -oP 'length=\K\d+' | sort -u | wc -l)
    if [ "$UNIQUE_LENGTHS" -eq 1 ]; then
        LENGTH=$(grep "Episode.*finished" "$MAIN_LOG" | head -1 | grep -oP 'length=\K\d+')
        echo "⚠️  All episodes have the same length ($LENGTH steps)"
        echo "    → May indicate environment is replaying a fixed sequence"
        ISSUES=$((ISSUES + 1))
        echo ""
    fi
fi

# Vérifier si tous les épisodes ont le même reward
if [ "$EPISODE_COUNT" -gt 1 ]; then
    UNIQUE_REWARDS=$(grep "Episode.*finished" "$MAIN_LOG" | grep -oP 'reward=\K[0-9.]+' | sort -u | wc -l)
    if [ "$UNIQUE_REWARDS" -eq 1 ]; then
        REWARD=$(grep "Episode.*finished" "$MAIN_LOG" | head -1 | grep -oP 'reward=\K[0-9.]+')
        echo "⚠️  All episodes have the same reward ($REWARD)"
        echo "    → May indicate reward calculation issue"
        ISSUES=$((ISSUES + 1))
        echo ""
    fi
fi

# Vérifier si le reward moyen est très faible
if [ "$EPISODE_COUNT" -gt 0 ]; then
    AVG_REWARD=$(grep "Episode.*finished" "$MAIN_LOG" | grep -oP 'reward=\K[0-9.]+' | awk '{sum+=$1; count++} END {print sum/count}')
    if [ "$(echo "$AVG_REWARD < 2.0" | bc -l)" -eq 1 ]; then
        echo "⚠️  Average episode reward is very low ($AVG_REWARD)"
        echo "    → May indicate missing terminal rewards or incorrect reward shaping"
        ISSUES=$((ISSUES + 1))
        echo ""
    fi
fi

if [ "$ISSUES" -eq 0 ]; then
    echo "✅ No obvious issues detected"
    echo ""
fi

# 6. Recommendations
echo "💡 Recommendations"
echo "=================="
echo ""

if [ "$EPISODE_COUNT" -eq 0 ]; then
    echo "⚠️  No episodes found - training may have failed"
    echo "   → Check main log for errors: tail -100 $MAIN_LOG"
elif [ "$ISSUES" -gt 0 ]; then
    echo "📝 To diagnose further:"
    echo "   1. Check detailed episode logs:"
    echo "      grep 'Episode.*reset\\|Episode.*terminated' $DEBUG_LOG | tail -20"
    echo ""
    echo "   2. Examine reward calculation:"
    echo "      grep 'reward_calculator\\|calculate_reward' $DEBUG_LOG | tail -50"
    echo ""
    echo "   3. Verify environment behavior:"
    echo "      grep 'gymnasium_heating_env' $DEBUG_LOG | tail -100"
else
    echo "✅ Training appears to be working correctly"
    echo "   → Continue monitoring episode rewards and lengths"
fi

echo ""
echo "=========================================="
echo "Log files location:"
echo "  Main: $MAIN_LOG"
echo "  Debug: $DEBUG_LOG"
echo "=========================================="
