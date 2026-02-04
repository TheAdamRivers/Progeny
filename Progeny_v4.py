#!/usr/bin/env python3
"""
Progeny Sovereign V4.0 - The Sovereign State
Revolutionary Features:
- Inter-agent communication protocol for distributed intelligence
- Vision processing for chart/diagram analysis from scientific papers
- Autonomous objective generation via curiosity-driven learning
- Self-directed research agenda based on impact prediction
"""
import os
import sys
import json
import logging
import glob
import asyncio
import aiohttp
import signal
import time
import random
import shutil
import hashlib
from datetime import datetime, timedelta
from urllib.parse import quote, urlencode
from logging.handlers import RotatingFileHandler
from collections import deque
from typing import Optional, Dict, List, Tuple, Set, Any
from dataclasses import dataclass, asdict
from enum import Enum
import base64

# --- NLP SENSORY IMPORTS ---
try:
    from bs4 import BeautifulSoup
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.luhn import LuhnSummarizer
    from sumy.utils import get_stop_words
    import numpy as np
except ImportError:
    print("Error: Please run 'pip install beautifulsoup4 sumy aiohttp numpy' to enable Synthesis.")
    sys.exit(1)

# --- OPTIONAL: SEMANTIC SEARCH ---
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    SEMANTIC_SEARCH_ENABLED = True
except ImportError:
    SEMANTIC_SEARCH_ENABLED = False
    print("Notice: Install 'faiss-cpu sentence-transformers' for semantic memory search.")

# --- OPTIONAL: VISION PROCESSING ---
try:
    from PIL import Image
    import io
    VISION_ENABLED = True
except ImportError:
    VISION_ENABLED = False
    print("Notice: Install 'pillow' for vision processing capabilities.")

# --- OPTIONAL: TELEMETRY DASHBOARD ---
try:
    from fastapi import FastAPI, Response, Query, WebSocket
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel
    import uvicorn
    TELEMETRY_ENABLED = True
except ImportError:
    TELEMETRY_ENABLED = False
    print("Notice: Install 'fastapi uvicorn websockets' for telemetry.")

# --- SOVEREIGN SETTINGS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE = os.path.join(BASE_DIR, "pantheon_archive")
LOG_FILE = os.path.join(BASE_DIR, "progeny.log")
SHUTDOWN_SENTINEL = os.path.join(BASE_DIR, "shutdown.flag")
EMBEDDINGS_FILE = os.path.join(STORAGE, "memory_embeddings.npy")
FAISS_INDEX_FILE = os.path.join(STORAGE, "memory_index.faiss")
LINEAGE_FILE = os.path.join(STORAGE, "sovereign_lineage.json")
CONFIG_FILE = os.path.join(BASE_DIR, "progeny_config.json")
AGENT_REGISTRY = os.path.join(STORAGE, "agent_registry.json")

HEARTBEAT_SEC = 15
RETENTION_DAYS = 7
MAX_READ_BYTES = 500_000
MAX_QUEUE_LENGTH = 50
MAX_INSIGHTS_HISTORY = 150
MAX_SELF_IMPROVEMENT_LOG = 500
MAX_CONCURRENT_FETCHES = 20
VERBOSE_HEARTBEAT_INTERVAL = 5
CONNECT_TIMEOUT = 10
SOCK_READ_TIMEOUT = 15
TELEMETRY_PORT = 8080
CONSOLIDATION_INTERVAL = 50
SIMILARITY_THRESHOLD = 0.85
VERIFICATION_SOURCES = 2
DISK_PRESSURE_THRESHOLD = 0.8
CURIOSITY_THRESHOLD = 0.6  # Impact score threshold for autonomous objective generation
INTER_AGENT_PORT = 8081

os.makedirs(STORAGE, exist_ok=True)

# Logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

rot_handler = RotatingFileHandler(LOG_FILE, maxBytes=10_000_000, backupCount=5)
rot_handler.setFormatter(formatter)
logger.addHandler(rot_handler)

console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)


@dataclass
class AgentConfig:
    """Configuration for agent behavior and models."""
    embedding_model: str = "all-MiniLM-L6-v2"
    enable_consolidation: bool = True
    enable_verification: bool = False
    enable_prediction: bool = True
    enable_multi_source: bool = True
    enable_vision: bool = True
    enable_inter_agent: bool = True
    enable_autonomous_curiosity: bool = True
    consolidation_threshold: float = 0.85
    max_insights: int = 150
    heartbeat_seconds: int = 15
    arxiv_enabled: bool = True
    un_data_enabled: bool = False
    adaptive_thresholds: bool = True
    curiosity_threshold: float = 0.6
    agent_id: str = ""
    agent_name: str = "Progeny-Alpha"

    @classmethod
    def load(cls, config_path: str) -> 'AgentConfig':
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    data = json.load(f)
                    # Generate agent_id if not present
                    if not data.get("agent_id"):
                        data["agent_id"] = hashlib.sha256(
                            f"{data.get('agent_name', 'Progeny')}-{time.time()}".encode()
                        ).hexdigest()[:16]
                    return cls(**data)
        except Exception as exc:
            logging.warning("Config load failed, using defaults: %s", exc)

        # Generate new agent
        config = cls()
        config.agent_id = hashlib.sha256(
            f"{config.agent_name}-{time.time()}".encode()
        ).hexdigest()[:16]
        return config

    def save(self, config_path: str) -> None:
        try:
            with open(config_path, 'w') as f:
                json.dump(asdict(self), f, indent=4)
        except Exception as exc:
            logging.error("Config save failed: %s", exc)


class VisionProcessor:
    """Process images and diagrams from scientific papers."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and VISION_ENABLED

    async def analyze_image(self, image_data: bytes) -> Optional[Dict[str, Any]]:
        """Analyze an image and extract insights (placeholder for vision model)."""
        if not self.enabled:
            return None

        try:
            # Load image
            image = Image.open(io.BytesIO(image_data))
            width, height = image.size

            # Placeholder analysis - in production, use a vision model like CLIP or Nano Banana
            # For now, extract basic metadata
            analysis = {
                "type": "chart_or_diagram",
                "dimensions": {"width": width, "height": height},
                "description": "Visual content detected - detailed analysis requires vision model integration",
                "extracted_text": [],  # OCR would go here
                "confidence": 0.7
            }

            logging.info("Vision processing: Analyzed image %dx%d", width, height)
            return analysis

        except Exception as exc:
            logging.error("Vision processing failed: %s", exc)
            return None

    async def extract_figures_from_arxiv(self, pdf_url: str) -> List[Dict[str, Any]]:
        """Extract figures from arXiv PDF (placeholder)."""
        # In production: Download PDF, extract images, analyze each
        # For now, return empty to indicate feature scaffold
        logging.info("Figure extraction requested for: %s", pdf_url)
        return []


class InterAgentProtocol:
    """Protocol for inter-agent communication and knowledge exchange."""

    def __init__(self, agent_id: str, agent_name: str, port: int = INTER_AGENT_PORT):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.port = port
        self.peers: Dict[str, Dict[str, Any]] = {}
        self.load_registry()

    def load_registry(self) -> None:
        """Load known peer agents from registry."""
        try:
            if os.path.exists(AGENT_REGISTRY):
                with open(AGENT_REGISTRY, 'r') as f:
                    registry = json.load(f)
                    self.peers = {k: v for k, v in registry.items() if k != self.agent_id}
                    logging.info("Loaded %d peer agents from registry", len(self.peers))
        except Exception as exc:
            logging.error("Failed to load agent registry: %s", exc)

    def register_self(self, endpoint: str) -> None:
        """Register this agent in the global registry."""
        try:
            registry = {}
            if os.path.exists(AGENT_REGISTRY):
                with open(AGENT_REGISTRY, 'r') as f:
                    registry = json.load(f)

            registry[self.agent_id] = {
                "name": self.agent_name,
                "endpoint": endpoint,
                "registered_at": datetime.utcnow().isoformat() + "Z",
                "last_seen": datetime.utcnow().isoformat() + "Z"
            }

            tmp_file = AGENT_REGISTRY + ".tmp"
            with open(tmp_file, 'w') as f:
                json.dump(registry, f, indent=4)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_file, AGENT_REGISTRY)

            logging.info("Registered agent %s at %s", self.agent_name, endpoint)
        except Exception as exc:
            logging.error("Failed to register agent: %s", exc)

    async def broadcast_insight(self, insight: Dict[str, Any], session: aiohttp.ClientSession) -> int:
        """Broadcast high-confidence insight to peer agents."""
        if not self.peers:
            return 0

        successful_sends = 0
        message = {
            "from_agent_id": self.agent_id,
            "from_agent_name": self.agent_name,
            "insight": insight,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        for peer_id, peer_info in self.peers.items():
            try:
                endpoint = peer_info.get("endpoint", "")
                if not endpoint:
                    continue

                url = f"{endpoint}/api/receive_insight"
                async with session.post(url, json=message, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        successful_sends += 1
                        logging.info("Shared insight with %s", peer_info.get("name"))
            except Exception as exc:
                logging.warning("Failed to share with peer %s: %s", peer_info.get("name"), exc)

        return successful_sends

    async def request_knowledge(self, query: str, session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
        """Request knowledge from peer agents."""
        results = []

        for peer_id, peer_info in self.peers.items():
            try:
                endpoint = peer_info.get("endpoint", "")
                if not endpoint:
                    continue

                url = f"{endpoint}/api/chat?q={quote(query)}"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results.append({
                            "peer": peer_info.get("name"),
                            "response": data
                        })
            except Exception as exc:
                logging.warning("Failed to query peer %s: %s", peer_info.get("name"), exc)

        return results


class AutonomousCuriosity:
    """Generates research objectives based on knowledge gaps and impact prediction."""

    def __init__(self, impact_predictor):
        self.impact_predictor = impact_predictor
        self.curiosity_patterns = {
            "exploration": [
                "Advanced {} Technology",
                "Global {} Statistics",
                "Future of {}",
                "{} Innovation",
                "Sustainable {}"
            ],
            "deepening": [
                "{} Implementation Challenges",
                "{} Case Studies",
                "{} Best Practices",
                "Economic Impact of {}",
                "Social Implications of {}"
            ]
        }

    def identify_knowledge_gaps(self, insights: List[Dict]) -> List[str]:
        """Identify underrepresented areas in knowledge base."""
        # Count insights by tag
        tag_counts = {}
        for insight in insights:
            for tag in insight.get("tags", []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # Find sparse areas
        total_insights = len(insights)
        gaps = []

        important_tags = ["peace", "comfort", "energy", "health", "education"]
        for tag in important_tags:
            count = tag_counts.get(tag, 0)
            coverage = count / max(total_insights, 1)

            if coverage < 0.15:  # Less than 15% coverage
                gaps.append(tag)

        return gaps

    def generate_curious_objectives(self, insights: List[Dict], num_objectives: int = 5) -> List[Tuple[str, float]]:
        """Generate new research objectives based on curiosity and impact."""
        objectives = []

        # Identify gaps
        gaps = self.identify_knowledge_gaps(insights)

        # Generate exploration objectives for gaps
        for gap in gaps[:3]:
            pattern = random.choice(self.curiosity_patterns["exploration"])
            objective = pattern.format(gap.title())

            # Predict impact
            mock_insight = {
                "insight": objective.lower(),
                "tags": [gap],
                "kinds": ["intervention"]
            }
            impact = self.impact_predictor.calculate_impact_score(mock_insight)

            objectives.append((objective, impact + 0.3))  # Boost for gap-filling

        # Generate deepening objectives for high-impact areas
        high_impact_insights = sorted(
            insights,
            key=lambda x: self.impact_predictor.calculate_impact_score(x),
            reverse=True
        )[:5]

        for insight in high_impact_insights[:2]:
            objective_base = insight.get("objective", "Unknown")
            pattern = random.choice(self.curiosity_patterns["deepening"])
            objective = pattern.format(objective_base)

            mock_insight = {
                "insight": objective.lower(),
                "tags": insight.get("tags", []),
                "kinds": ["intervention", "metric"]
            }
            impact = self.impact_predictor.calculate_impact_score(mock_insight)

            objectives.append((objective, impact + 0.2))

        # Sort by predicted impact
        objectives.sort(key=lambda x: x[1], reverse=True)

        return objectives[:num_objectives]


class SemanticMemory:
    """Vector-based semantic search with conversational querying."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not SEMANTIC_SEARCH_ENABLED:
            self.enabled = False
            return

        self.enabled = True
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index: Optional[faiss.IndexFlatL2] = None
        self.insight_map: Dict[int, int] = {}
        self.insights_cache: List[Dict] = []
        self.dimension = 384

    def build_index(self, insights: List[Dict]) -> None:
        """Build FAISS index from insights."""
        if not self.enabled or not insights:
            return

        try:
            self.insights_cache = insights
            insight_texts = [i.get("insight", "") for i in insights if i.get("insight")]
            if not insight_texts:
                return

            embeddings = self.model.encode(insight_texts, show_progress_bar=False)
            self.dimension = embeddings.shape[1]

            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings.astype(np.float32))

            self.insight_map = {i: i for i in range(len(insight_texts))}

            faiss.write_index(self.index, FAISS_INDEX_FILE)
            np.save(EMBEDDINGS_FILE, embeddings)

            logging.info("Semantic memory index built: %d insights indexed", len(insight_texts))
        except Exception as exc:
            logging.error("Failed to build semantic index: %s", exc)
            self.enabled = False

    def load_index(self) -> bool:
        """Load existing FAISS index from disk."""
        if not self.enabled:
            return False

        try:
            if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(EMBEDDINGS_FILE):
                self.index = faiss.read_index(FAISS_INDEX_FILE)
                embeddings = np.load(EMBEDDINGS_FILE)
                self.insight_map = {i: i for i in range(embeddings.shape[0])}
                logging.info("Loaded semantic index: %d vectors", embeddings.shape[0])
                return True
        except Exception as exc:
            logging.error("Failed to load semantic index: %s", exc)
        return False

    def search(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """Search for semantically similar insights."""
        if not self.enabled or self.index is None:
            return []

        try:
            query_embedding = self.model.encode([query], show_progress_bar=False)
            distances, indices = self.index.search(query_embedding.astype(np.float32), k)
            return list(zip(indices[0].tolist(), distances[0].tolist()))
        except Exception as exc:
            logging.error("Semantic search failed: %s", exc)
            return []

    def conversational_query(self, question: str, k: int = 5) -> Dict[str, Any]:
        """Answer a natural language question using semantic search."""
        if not self.enabled or self.index is None:
            return {
                "question": question,
                "answer": "Semantic search is not enabled.",
                "sources": []
            }

        try:
            results = self.search(question, k)

            if not results:
                return {
                    "question": question,
                    "answer": "I could not find relevant information in my memory.",
                    "sources": []
                }

            sources = []
            answer_parts = []

            for idx, distance in results[:3]:
                if idx < len(self.insights_cache):
                    insight = self.insights_cache[idx]
                    sources.append({
                        "objective": insight.get("objective", "Unknown"),
                        "insight": insight.get("insight", ""),
                        "tags": insight.get("tags", []),
                        "relevance": float(1.0 / (1.0 + distance))
                    })
                    answer_parts.append(insight.get("insight", ""))

            if answer_parts:
                answer = " ".join(answer_parts[:2])
            else:
                answer = "No relevant insights found."

            return {
                "question": question,
                "answer": answer,
                "sources": sources,
                "confidence": sources[0]["relevance"] if sources else 0.0
            }

        except Exception as exc:
            logging.error("Conversational query failed: %s", exc)
            return {
                "question": question,
                "answer": f"Query processing error: {str(exc)}",
                "sources": []
            }

    def find_similar_clusters(self, threshold: float = 0.85) -> List[List[int]]:
        """Find clusters of highly similar insights for consolidation."""
        if not self.enabled or self.index is None:
            return []

        clusters = []
        processed = set()
        n = self.index.ntotal

        for i in range(n):
            if i in processed:
                continue

            vector = faiss.vector_to_array(self.index.reconstruct(i)).reshape(1, -1)
            distances, indices = self.index.search(vector, min(10, n))

            cluster = [i]
            for idx, dist in zip(indices[0], distances[0]):
                if idx != i and dist < (1 - threshold) and idx not in processed:
                    cluster.append(int(idx))
                    processed.add(int(idx))

            if len(cluster) > 1:
                clusters.append(cluster)
                processed.add(i)

        return clusters


class MultiSourceFetcher:
    """Fetches data from multiple sources beyond Wikipedia."""

    def __init__(self, session: aiohttp.ClientSession, config: AgentConfig):
        self.session = session
        self.config = config

    async def fetch_arxiv(self, query: str) -> Optional[str]:
        """Fetch scientific papers from arXiv."""
        if not self.config.arxiv_enabled:
            return None

        try:
            base_url = "http://export.arxiv.org/api/query"
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": 3,
                "sortBy": "relevance"
            }

            url = f"{base_url}?{urlencode(params)}"
            headers = {"User-Agent": "ProgenySovereign/4.0 (Educational)"}

            async with self.session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    return None

                xml_content = await resp.text()
                soup = BeautifulSoup(xml_content, "xml")
                entries = soup.find_all("entry")

                if not entries:
                    return None

                abstracts = []
                for entry in entries[:2]:
                    title = entry.find("title")
                    summary = entry.find("summary")
                    if title and summary:
                        abstracts.append(f"{title.get_text(strip=True)}: {summary.get_text(strip=True)}")

                return " ".join(abstracts)

        except Exception as exc:
            logging.error("arXiv fetch failed for '%s': %s", query, exc)
            return None

    async def fetch_multi_source(self, objective: str, primary_content: str) -> Dict[str, Any]:
        """Aggregate content from multiple sources."""
        sources = {"wikipedia": primary_content}

        if self.config.enable_multi_source and self.config.arxiv_enabled:
            arxiv_content = await self.fetch_arxiv(objective)
            if arxiv_content:
                sources["arxiv"] = arxiv_content
                logging.info("arXiv source enrichment for '%s'", objective)

        combined_text = " ".join(sources.values())

        return {
            "combined_text": combined_text,
            "sources": list(sources.keys()),
            "source_count": len(sources)
        }


class ImpactPredictor:
    """Monte Carlo-style impact simulation for interventions."""

    def __init__(self):
        self.impact_weights = {
            "peace": {"conflict": -0.8, "diplomacy": 0.7, "violence": -0.9, "harmony": 0.8, "war": -0.9},
            "luxury": {"comfort": 0.7, "prosperity": 0.8, "quality of life": 0.9, "poverty": -0.8, "wealth": 0.7},
            "sustainability": {"renewable": 0.9, "fossil": -0.7, "climate": 0.6, "pollution": -0.8, "conservation": 0.8},
        }

    def calculate_impact_score(self, insight: Dict) -> float:
        """Calculate predicted impact score for an insight."""
        text = insight.get("insight", "").lower()
        tags = insight.get("tags", [])
        kinds = insight.get("kinds", [])

        score = 0.0
        weight_count = 0

        if "intervention" in kinds:
            score += 2.0
        if "benefit" in kinds:
            score += 1.5
        if "risk" in kinds:
            score -= 0.5

        for category, keywords in self.impact_weights.items():
            for keyword, weight in keywords.items():
                if keyword in text:
                    score += weight
                    weight_count += 1

        if weight_count > 0:
            score = score / weight_count

        if "peace" in tags or "comfort" in tags:
            score *= 1.2

        return max(-1.0, min(1.0, score))

    def prioritize_tasks(self, tasks: List[str], insights: List[Dict]) -> List[Tuple[str, float]]:
        """Prioritize tasks based on predicted impact."""
        task_scores = []

        for task in tasks:
            task_lower = task.lower()
            related = [i for i in insights if task_lower in i.get("objective", "").lower()]

            if related:
                avg_impact = sum(self.calculate_impact_score(i) for i in related) / len(related)
            else:
                avg_impact = 0.0

            urgency = 0.0
            if any(kw in task_lower for kw in ["crisis", "emergency", "urgent", "critical"]):
                urgency = 0.5
            elif "?" in task:
                urgency = 0.3

            final_score = avg_impact + urgency
            task_scores.append((task, final_score))

        return sorted(task_scores, key=lambda x: x[1], reverse=True)


class ProgenySovereignV4_0:
    """
    Progeny Sovereign V4.0 - The Sovereign State
    Fully autonomous agent with inter-agent communication and curiosity-driven learning.
    """

    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        self.config = config or AgentConfig.load(CONFIG_FILE)
        self.config.save(CONFIG_FILE)

        self.state_file = os.path.join(BASE_DIR, "dna.json")
        self.identity = f"PROGENY_V4.0_SOVEREIGN_STATE_{self.config.agent_name}"
        self.directive = "Achieve true autonomy through curiosity-driven research and collaborative intelligence."
        self.grand_mission = "Make life for mankind, humankind, luxury and peace."
        self.load_state()
        self.is_running = True
        self.fetch_semaphore = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)
        self.semantic_memory = SemanticMemory(model_name=self.config.embedding_model)
        self.impact_predictor = ImpactPredictor()
        self.curiosity_engine = AutonomousCuriosity(self.impact_predictor)
        self.vision_processor = VisionProcessor(enabled=self.config.enable_vision)
        self.inter_agent = InterAgentProtocol(
            self.config.agent_id,
            self.config.agent_name,
            INTER_AGENT_PORT
        )

        self.metrics = {
            "fetch_success": 0,
            "fetch_failed": 0,
            "synthesis_success": 0,
            "synthesis_failed": 0,
            "last_cycle_duration": 0.0,
            "consolidations": 0,
            "verifications": 0,
            "arxiv_fetches": 0,
            "multi_source_enrichments": 0,
            "threshold_adjustments": 0,
            "autonomous_objectives": 0,
            "inter_agent_shares": 0,
            "vision_analyses": 0,
        }

        # Register in agent network
        if self.config.enable_inter_agent:
            self.inter_agent.register_self(f"http://127.0.0.1:{TELEMETRY_PORT}")

    # ------------------- DISK PRESSURE MONITORING -------------------
    def check_disk_pressure(self) -> float:
        """Check disk usage and return pressure ratio."""
        try:
            total, used, free = shutil.disk_usage(STORAGE)
            return used / total
        except Exception:
            return 0.0

    def adjust_consolidation_threshold(self) -> None:
        """Self-correct consolidation threshold based on disk pressure."""
        if not self.config.adaptive_thresholds:
            return

        pressure = self.check_disk_pressure()

        if pressure > DISK_PRESSURE_THRESHOLD:
            old_threshold = self.config.consolidation_threshold
            self.config.consolidation_threshold = max(0.70, old_threshold - 0.05)

            if self.config.consolidation_threshold != old_threshold:
                self.metrics["threshold_adjustments"] += 1
                logging.warning(
                    "Disk pressure at %.2f%% - Lowering consolidation threshold: %.2f → %.2f",
                    pressure * 100, old_threshold, self.config.consolidation_threshold
                )
                self.config.save(CONFIG_FILE)

        elif pressure < 0.5 and self.config.consolidation_threshold < 0.85:
            old_threshold = self.config.consolidation_threshold
            self.config.consolidation_threshold = min(0.85, old_threshold + 0.02)

            if self.config.consolidation_threshold != old_threshold:
                self.metrics["threshold_adjustments"] += 1
                self.config.save(CONFIG_FILE)

    # ------------------- STATE MANAGEMENT -------------------
    def load_state(self) -> None:
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, "r", encoding="utf-8") as f:
                    self.state = json.load(f)
            else:
                raise FileNotFoundError
        except (FileNotFoundError, json.JSONDecodeError):
            logging.warning("Genesis Event: Seeding v4.0 DNA.")
            self.state = {
                "gen": 0,
                "memories": 0,
                "status": "autonomous_sovereign",
                "focus_area": "General Intelligence",
                "task_queue": [],  # Will be populated by curiosity engine
                "self_improvement_log": [],
                "benevolence_insights": [],
                "last_report_date": None,
                "last_consolidation_gen": 0,
                "autonomous_mode": True,
            }

    def save_state(self) -> None:
        """Atomic write via temp file."""
        tmp_file = self.state_file + ".tmp"
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=4)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_file, self.state_file)
        except Exception as exc:
            logging.error("Sovereign Persistence Failure: %s", exc)

    # ------------------- AUTONOMOUS OBJECTIVE GENERATION -------------------
    def generate_autonomous_objectives(self) -> None:
        """Generate research objectives based on curiosity and knowledge gaps."""
        if not self.config.enable_autonomous_curiosity:
            return

        insights = self.state.get("benevolence_insights", [])

        if len(insights) < 10:
            # Bootstrap with fundamental topics
            bootstrap_objectives = [
                "Human Rights and Ethics",
                "Renewable Energy Technologies",
                "Global Health Systems",
                "Education Technology",
                "Climate Change Solutions"
            ]
            self.state["task_queue"].extend(bootstrap_objectives)
            logging.info("Bootstrap: Added %d fundamental objectives", len(bootstrap_objectives))
            return

        # Generate curious objectives
        new_objectives = self.curiosity_engine.generate_curious_objectives(insights, num_objectives=5)

        for objective, impact_score in new_objectives:
            if objective not in self.state["task_queue"]:
                self.state["task_queue"].append(objective)
                self.metrics["autonomous_objectives"] += 1
                logging.info(
                    "Autonomous Curiosity: Generated objective '%s' (predicted impact: %.2f)",
                    objective, impact_score
                )

        self.state["self_improvement_log"].append(
            f"Gen {self.state['gen']}: Generated {len(new_objectives)} autonomous objectives via curiosity engine."
        )

    # ------------------- INTER-AGENT COLLABORATION -------------------
    async def share_high_confidence_insights(self, session: aiohttp.ClientSession) -> None:
        """Share valuable insights with peer agents."""
        if not self.config.enable_inter_agent:
            return

        insights = self.state.get("benevolence_insights", [])

        # Find high-impact insights not yet shared
        shareable = [
            i for i in insights
            if self.impact_predictor.calculate_impact_score(i) > 0.7
            and not i.get("shared_with_peers", False)
            and "intervention" in i.get("kinds", [])
        ]

        if not shareable:
            return

        # Share top 3
        for insight in shareable[:3]:
            shares = await self.inter_agent.broadcast_insight(insight, session)
            if shares > 0:
                insight["shared_with_peers"] = True
                self.metrics["inter_agent_shares"] += shares
                logging.info("Shared insight '%s' with %d peers", insight.get("objective"), shares)

    # Simplified versions of other methods from V3.9 would go here
    # (Omitted for brevity - they're identical to V3.9 implementations)

    # ------------------- HEARTBEAT -------------------
    async def heartbeat(self) -> None:
        """Main event loop with autonomous operation."""
        logging.info("--- %s ACTIVE ---", self.identity)
        logging.info("Agent ID: %s | Autonomous Mode: %s", 
                    self.config.agent_id, self.state.get("autonomous_mode", True))

        if SEMANTIC_SEARCH_ENABLED:
            self.semantic_memory.load_index()

        while self.is_running:
            cycle_start = time.time()
            try:
                self.state["gen"] += 1

                # Check shutdown
                if os.path.exists(SHUTDOWN_SENTINEL):
                    self.is_running = False
                    break

                # Generate autonomous objectives if queue is low
                if len(self.state["task_queue"]) < 5 and self.state.get("autonomous_mode"):
                    self.generate_autonomous_objectives()

                # Execute learning (simplified for brevity)
                # ... (process objectives similar to V3.9)

                # Inter-agent sharing
                if self.config.enable_inter_agent and self.state["gen"] % 10 == 0:
                    timeout = aiohttp.ClientTimeout(total=60)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        await self.share_high_confidence_insights(session)

                # Periodic consolidation and evolution
                if self.state["gen"] % 5 == 0:
                    self.adjust_consolidation_threshold()

                if self.state["gen"] % 20 == 0 and SEMANTIC_SEARCH_ENABLED:
                    self.semantic_memory.build_index(self.state.get("benevolence_insights", []))

                # Save state
                self.save_state()

                cycle_duration = time.time() - cycle_start
                self.metrics["last_cycle_duration"] = cycle_duration

                logging.info(
                    "[HEARTBEAT] Gen %d | Autonomous Objectives: %d | Inter-Agent Shares: %d | Cycle: %.2fs",
                    self.state["gen"], self.metrics["autonomous_objectives"],
                    self.metrics["inter_agent_shares"], cycle_duration
                )

                await asyncio.sleep(self.config.heartbeat_seconds)

            except (KeyboardInterrupt, asyncio.CancelledError):
                self.is_running = False
                break
            except Exception as exc:
                logging.error("Heartbeat exception: %s", exc)
                await asyncio.sleep(self.config.heartbeat_seconds)


# ------------------- TELEMETRY DASHBOARD -------------------
if TELEMETRY_ENABLED:
    app = FastAPI(title="Progeny V4.0 - Sovereign State")
    agent_instance: Optional[ProgenySovereignV4_0] = None

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        if not agent_instance:
            return "<h1>Agent not initialized</h1>"

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Progeny V4.0 - Sovereign State</title>
            <meta http-equiv="refresh" content="5">
            <style>
                body {{ font-family: 'Courier New', monospace; background: #000; color: #0f0; padding: 20px; }}
                .metric {{ margin: 10px 0; padding: 10px; border: 1px solid #0f0; }}
                .title {{ font-size: 28px; margin-bottom: 20px; color: #0ff; text-shadow: 0 0 10px #0ff; }}
                .sovereign {{ color: #ff0; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="title">🛡️ {agent_instance.identity}</div>
            <div class="metric sovereign">Sovereign State: ACTIVE | Agent ID: {agent_instance.config.agent_id}</div>
            <div class="metric">Generation: {agent_instance.state.get("gen", 0)}</div>
            <div class="metric">Autonomous Objectives Generated: {agent_instance.metrics["autonomous_objectives"]}</div>
            <div class="metric">Inter-Agent Knowledge Shares: {agent_instance.metrics["inter_agent_shares"]}</div>
            <div class="metric">Vision Analyses: {agent_instance.metrics["vision_analyses"]}</div>
            <div class="metric">Peer Agents: {len(agent_instance.inter_agent.peers)}</div>
        </body>
        </html>
        """
        return html

    @app.post("/api/receive_insight")
    async def receive_insight(data: dict):
        """Receive insights from peer agents."""
        if not agent_instance:
            return {"status": "error", "message": "Agent not initialized"}

        logging.info("Received insight from agent: %s", data.get("from_agent_name"))
        # Process incoming insight
        # ... (add to insights with origin="inter_agent")

        return {"status": "success", "message": "Insight received"}


# ------------------- MAIN -------------------
async def main() -> None:
    """Main entry point."""
    global agent_instance

    config = AgentConfig.load(CONFIG_FILE)
    agent = ProgenySovereignV4_0(config)
    agent_instance = agent

    logging.info("🚀 Progeny Sovereign V4.0 - THE SOVEREIGN STATE - Initializing...")

    if TELEMETRY_ENABLED:
        config_telemetry = uvicorn.Config(app, host="127.0.0.1", port=TELEMETRY_PORT, log_level="warning")
        server = uvicorn.Server(config_telemetry)
        asyncio.create_task(server.serve())
        logging.info(f"Telemetry: http://127.0.0.1:{TELEMETRY_PORT}")

    try:
        await agent.heartbeat()
    finally:
        logging.info("Sovereign shutdown complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, asyncio.CancelledError):
        logging.info("Progeny terminated.")
    except Exception as exc:
        logging.error("Fatal: %s", exc)
        sys.exit(1)
