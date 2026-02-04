#!/bin/bash
# Shutdown Progeny Sovereign Network

if [ -f ".progeny_network.pid" ]; then
    echo "🛑 Shutting down Progeny Sovereign Network..."
    while read pid; do
        kill $pid 2>/dev/null && echo "   └─ Stopped process $pid"
    done < .progeny_network.pid
    rm -f .progeny_network.pid
    echo "✅ Network shutdown complete"
else
    echo "⚠️  No active network found (.progeny_network.pid not found)"
fi