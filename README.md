# Progeny V4.0 - CodeName Progeny Sovereign

## What is Progeny? 
Progeny Sovereign is an autonomous, distributed research ecosystem designed to synthesize global information into strategic roadmaps for human advancement. 

By integrating semantic memory, vision processing, and a decentralized communication protocol, the system identifies high-impact interventions and systemic risks across multiple data domains.

## Core Capabilities
* __Distributed Cognitive Synthesis__
Version 4.0 introduces the Inter-Agent Communication Protocol (IACP), enabling individual agent nodes to synchronize knowledge. This peer-to-peer architecture allows for specialized research tasks across a cluster while maintaining a unified global intelligence state.

* __Multi-Modal Data Acquisition__
The system ingests data from diverse sources including Wikipedia, arXiv scientific papers, and UN global metrics. The integrated vision logic allows the agent to process charts and diagrams within technical documents, capturing insights that are inaccessible to text-only analysis.

* __Semantic Memory Engine__
Built on FAISS (Facebook AI Similarity Search) and sentence embeddings, Progeny utilizes a high-dimensional vector space to organize "memories." This enables conceptual retrieval, allowing the agent to understand context and relate disparate pieces of information through meaning rather than keyword matching.

* __Autonomous Curiosity & Impact Prediction__
The system operates on a curiosity-driven research agenda. Using predictive heuristics, it calculates the "Impact Score" of new information and dynamically generates its own research tasks to resolve knowledge gaps or investigate potential breakthroughs.

* __Adaptive Resource Management__
To ensure long-term stability without human intervention, the agent implements a "Dream Cycle" memory consolidation. It self-regulates storage by merging redundant semantic vectors based on real-time disk pressure, adhering to strict data pruning and lineage preservation protocols.

## Technical Architecture
```
| Component | Implementation |
|---|---|
| Cognition | sumy (Luhn Synthesis), sentence-transformers |
| Memory | FAISS, NumPy |
| Communication | FastAPI, Uvicorn, aiohttp |
| Parsing | BeautifulSoup, Pillow (Vision Analysis) |
| Process Management | Asyncio, Bash orchestration |
```
## Deployment Prerequisites
 * Python 3.9 or higher
 * pip install beautifulsoup4 sumy aiohttp numpy faiss-cpu sentence-transformers fastapi uvicorn async-lru pillow

## Network Initialization
Progeny is designed to run as a network of agents (e.g., Alpha and Beta nodes). Use the provided shell scripts to manage the lifecycle of the distributed state:
 * Start Network:
   chmod +x progeny_start_network.sh
./progeny_start_network.sh

 * Stop Network:
   ./progeny_stop_network.sh

## Monitoring & Interaction
Each node hosts a local telemetry dashboard providing real-time logs and a natural language query interface (/chat). Default ports are configured at 8080 (Alpha) and 8082 (Beta).

## Configuration
System behavior is defined in Progeny_config.json. Key parameters include:
 * enable_inter_agent: Toggles P2P knowledge exchange.
 * enable_vision: Activates analysis of technical imagery.
 * adaptive_thresholds: Enables self-regulating memory consolidation based on storage limits.
 * curiosity_threshold: Adjusts the sensitivity of the autonomous research generator.

## Governance and Sovereignty
The system operates under a set of immutable guidelines to ensure efficiency and autonomy. It performs automated pruning of snapshots every 7 days while preserving the "essence" of its knowledge within the pantheon_archive, maintaining the sovereignty of its learned experience without excessive resource consumption.

## License: Distributed under the MIT License.
