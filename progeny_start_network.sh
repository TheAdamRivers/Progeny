#!/bin/bash
# Progeny Sovereign V4.0 - Multi-Agent Network Launcher
# This script launches multiple Progeny agents for inter-agent communication

echo "=========================================="
echo "🚀 Progeny Sovereign Network Launcher"
echo "=========================================="
echo ""

# Check if config file exists
if [ ! -f "progeny_config_v4.json" ]; then
    echo "❌ Error: progeny_config_v4.json not found"
    echo "Please ensure the config file exists in the current directory"
    exit 1
fi

# Check if main script exists
if [ ! -f "progeny_sovereign_v4_0.py" ]; then
    echo "❌ Error: progeny_sovereign_v4_0.py not found"
    exit 1
fi

echo "📦 Creating agent configurations..."

# Create Agent Alpha config (Port 8080)
cat > progeny_config_alpha.json << 'EOF'
{
    "embedding_model": "all-MiniLM-L6-v2",
    "enable_consolidation": true,
    "enable_verification": false,
    "enable_prediction": true,
    "enable_multi_source": true,
    "enable_vision": true,
    "enable_inter_agent": true,
    "enable_autonomous_curiosity": true,
    "consolidation_threshold": 0.85,
    "max_insights": 150,
    "heartbeat_seconds": 15,
    "arxiv_enabled": true,
    "un_data_enabled": false,
    "adaptive_thresholds": true,
    "curiosity_threshold": 0.6,
    "agent_id": "",
    "agent_name": "Progeny-Alpha"
}
EOF

# Create Agent Beta config (Port 8082)
cat > progeny_config_beta.json << 'EOF'
{
    "embedding_model": "all-MiniLM-L6-v2",
    "enable_consolidation": true,
    "enable_verification": false,
    "enable_prediction": true,
    "enable_multi_source": true,
    "enable_vision": true,
    "enable_inter_agent": true,
    "enable_autonomous_curiosity": true,
    "consolidation_threshold": 0.85,
    "max_insights": 150,
    "heartbeat_seconds": 15,
    "arxiv_enabled": true,
    "un_data_enabled": false,
    "adaptive_thresholds": true,
    "curiosity_threshold": 0.6,
    "agent_id": "",
    "agent_name": "Progeny-Beta"
}
EOF

echo "✅ Configurations created"
echo ""
echo "Starting agents..."
echo ""

# Launch Agent Alpha
echo "🤖 Starting Progeny-Alpha on port 8080..."
PROGENY_CONFIG=progeny_config_alpha.json PROGENY_PORT=8080 python3 progeny_sovereign_v4_0.py > logs_alpha.txt 2>&1 &
ALPHA_PID=$!
echo "   └─ PID: $ALPHA_PID"

# Wait for Alpha to initialize
sleep 3

# Launch Agent Beta
echo "🤖 Starting Progeny-Beta on port 8082..."
PROGENY_CONFIG=progeny_config_beta.json PROGENY_PORT=8082 python3 progeny_sovereign_v4_0.py > logs_beta.txt 2>&1 &
BETA_PID=$!
echo "   └─ PID: $BETA_PID"

sleep 2

echo ""
echo "=========================================="
echo "✅ Progeny Sovereign Network is ACTIVE"
echo "=========================================="
echo ""
echo "🌐 Agent Dashboards:"
echo "   • Progeny-Alpha: http://127.0.0.1:8080"
echo "   • Progeny-Beta:  http://127.0.0.1:8082"
echo ""
echo "📊 Logs:"
echo "   • Alpha: tail -f logs_alpha.txt"
echo "   • Beta:  tail -f logs_beta.txt"
echo ""
echo "🛑 Shutdown:"
echo "   Press Ctrl+C or run: kill $ALPHA_PID $BETA_PID"
echo ""

# Save PIDs to file for easy shutdown
echo "$ALPHA_PID" > .progeny_network.pid
echo "$BETA_PID" >> .progeny_network.pid

# Wait for user interrupt
trap "echo ''; echo '🛑 Shutting down network...'; kill $ALPHA_PID $BETA_PID 2>/dev/null; rm -f .progeny_network.pid; echo '✅ Network stopped'; exit 0" INT TERM

# Keep script running
wait