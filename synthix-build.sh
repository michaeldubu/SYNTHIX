#!/bin/bash
# SYNTHIX-SAM OS v2.0 - Advanced Build System
# State-of-the-art AI Universe with Embodied SAM Agents
# Full SIMS-like visual experience with neural weight persistence

set -euo pipefail
IFS=$'\n\t'

# Advanced error handling with stack traces
set -E
trap 'error_handler $? $LINENO' ERR

error_handler() {
    local exit_code=$1
    local line_no=$2
    echo -e "${RED}[CRITICAL ERROR]${NC} Exit code $exit_code at line $line_no"
    echo "Stack trace:"
    local frame=0
    while caller $frame; do
        ((frame++))
    done
    cleanup_all
    exit $exit_code
}

# Enhanced variables
readonly WORK_DIR="$(pwd)/synthix-v2-build"
readonly OUTPUT_DIR="$(pwd)/synthix-v2-output"
readonly ISO_NAME="SYNTHIX-SAM-OS-v2.0-ULTIMATE.iso"
readonly BASE_DISTRO="ubuntu-22.04"
readonly ARCH="amd64"
readonly CHROOT_ENV="${WORK_DIR}/chroot"
readonly CACHE_DIR="${WORK_DIR}/cache"
readonly MODELS_DIR="${WORK_DIR}/models"
readonly ASSETS_DIR="${WORK_DIR}/assets"

# Advanced color system
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly WHITE='\033[1;37m'
readonly NC='\033[0m'

# Logging with levels
readonly LOG_FILE="${WORK_DIR}/build.log"
log_debug() { echo -e "${WHITE}[DEBUG]${NC} $1" | tee -a "$LOG_FILE"; }
log_info() { echo -e "${GREEN}[INFO]${NC} $1" | tee -a "$LOG_FILE"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1" | tee -a "$LOG_FILE"; }
log_success() { echo -e "${PURPLE}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"; }

# Enhanced dependency checking
check_dependencies() {
    log_step "Checking advanced dependencies..."
    
    local deps=(
        "debootstrap:debootstrap"
        "xorriso:xorriso"
        "grub-mkrescue:grub-pc-bin"
        "python3:python3"
        "pip3:python3-pip"
        "docker:docker.io"
        "nodejs:nodejs"
        "npm:npm"
        "git:git"
        "curl:curl"
        "wget:wget"
        "qemu-img:qemu-utils"
        "mksquashfs:squashfs-tools"
        "genisoimage:genisoimage"
    )
    
    local missing=()
    for dep in "${deps[@]}"; do
        local cmd="${dep%%:*}"
        local pkg="${dep##*:}"
        if ! command -v "$cmd" &> /dev/null; then
            missing+=("$pkg")
        fi
    done
    
    if [ ${#missing[@]} -ne 0 ]; then
        log_error "Missing dependencies: ${missing[*]}"
        log_info "Install with: sudo apt-get install -y ${missing[*]}"
        exit 1
    fi
    
    # Check Python version
    local python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if [[ $(echo "$python_version < 3.8" | bc) -eq 1 ]]; then
        log_error "Python 3.8+ required (found $python_version)"
        exit 1
    fi
    
    # Check Node.js version
    local node_version=$(node -v | sed 's/v//' | cut -d. -f1)
    if [[ $node_version -lt 14 ]]; then
        log_error "Node.js 14+ required"
        exit 1
    fi
    
    log_success "All dependencies satisfied"
}

# Advanced directory setup with asset preparation
setup_dirs() {
    log_step "Setting up advanced directory structure..."
    
    # Clean previous build
    if [ -d "$WORK_DIR" ]; then
        log_warn "Cleaning previous build..."
        cleanup_all
    fi
    
    # Create comprehensive directory structure
    local dirs=(
        "${CHROOT_ENV}"
        "${WORK_DIR}/iso/boot/grub"
        "${WORK_DIR}/iso/isolinux"
        "${WORK_DIR}/iso/casper"
        "${WORK_DIR}/iso/.disk"
        "${OUTPUT_DIR}"
        "${CACHE_DIR}"
        "${MODELS_DIR}"
        "${ASSETS_DIR}/sprites"
        "${ASSETS_DIR}/audio"
        "${ASSETS_DIR}/shaders"
        "${ASSETS_DIR}/models"
        "${CHROOT_ENV}/usr/lib/synthix/core"
        "${CHROOT_ENV}/usr/lib/synthix/universe"
        "${CHROOT_ENV}/usr/lib/synthix/agents"
        "${CHROOT_ENV}/usr/lib/synthix/physics"
        "${CHROOT_ENV}/usr/lib/synthix/render"
        "${CHROOT_ENV}/usr/lib/sam/core"
        "${CHROOT_ENV}/usr/lib/sam/models"
        "${CHROOT_ENV}/usr/lib/sam/weights"
        "${CHROOT_ENV}/usr/share/synthix/web/dist"
        "${CHROOT_ENV}/usr/share/synthix/assets"
        "${CHROOT_ENV}/etc/synthix/config"
        "${CHROOT_ENV}/var/lib/synthix/saves"
        "${CHROOT_ENV}/var/lib/synthix/cache"
        "${CHROOT_ENV}/var/lib/sam/checkpoints"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
    done
    
    # Initialize build log
    touch "$LOG_FILE"
    
    log_success "Directory structure created"
}

# Create Ubuntu base with GPU support
create_base_system() {
    log_step "Creating Ubuntu 22.04 base system with GPU support..."
    
    # Use cache if available
    if [ -f "${CACHE_DIR}/ubuntu-base.tar" ]; then
        log_info "Using cached base system..."
        tar -xf "${CACHE_DIR}/ubuntu-base.tar" -C "${CHROOT_ENV}"
    else
        # Bootstrap Ubuntu
        sudo debootstrap \
            --arch=${ARCH} \
            --variant=minbase \
            --include=ubuntu-minimal,ubuntu-standard \
            jammy \
            "${CHROOT_ENV}" \
            http://archive.ubuntu.com/ubuntu/
        
        # Cache for future builds
        sudo tar -cf "${CACHE_DIR}/ubuntu-base.tar" -C "${CHROOT_ENV}" .
    fi
    
    # Configure apt sources
    cat << EOF | sudo tee "${CHROOT_ENV}/etc/apt/sources.list"
deb http://archive.ubuntu.com/ubuntu/ jammy main restricted universe multiverse
deb http://archive.ubuntu.com/ubuntu/ jammy-updates main restricted universe multiverse
deb http://archive.ubuntu.com/ubuntu/ jammy-security main restricted universe multiverse
deb http://archive.ubuntu.com/ubuntu/ jammy-backports main restricted universe multiverse
EOF
    
    log_success "Ubuntu base system created"
}

# Mount system directories for chroot
mount_chroot() {
    log_debug "Mounting chroot filesystems..."
    
    # Mount with error checking
    sudo mount -t proc /proc "${CHROOT_ENV}/proc" || true
    sudo mount -t sysfs /sys "${CHROOT_ENV}/sys" || true
    sudo mount -o bind /dev "${CHROOT_ENV}/dev" || true
    sudo mount -o bind /dev/pts "${CHROOT_ENV}/dev/pts" || true
    sudo mount -t tmpfs tmpfs "${CHROOT_ENV}/run" || true
    
    # Copy resolv.conf for network access
    sudo cp -L /etc/resolv.conf "${CHROOT_ENV}/etc/resolv.conf"
    
    # Set up policy-rc.d to prevent services starting in chroot
    echo "exit 101" | sudo tee "${CHROOT_ENV}/usr/sbin/policy-rc.d" > /dev/null
    sudo chmod +x "${CHROOT_ENV}/usr/sbin/policy-rc.d"
}

# Unmount chroot filesystems
unmount_chroot() {
    log_debug "Unmounting chroot filesystems..."
    
    # Unmount in reverse order
    sudo umount -lf "${CHROOT_ENV}/run" 2>/dev/null || true
    sudo umount -lf "${CHROOT_ENV}/dev/pts" 2>/dev/null || true
    sudo umount -lf "${CHROOT_ENV}/dev" 2>/dev/null || true
    sudo umount -lf "${CHROOT_ENV}/sys" 2>/dev/null || true
    sudo umount -lf "${CHROOT_ENV}/proc" 2>/dev/null || true
    
    # Remove policy-rc.d
    sudo rm -f "${CHROOT_ENV}/usr/sbin/policy-rc.d"
}

# Execute command in chroot safely
chroot_exec() {
    local cmd="$1"
    log_debug "Executing in chroot: $cmd"
    sudo chroot "${CHROOT_ENV}" /bin/bash -c "$cmd"
}

# Install advanced packages in chroot
install_packages() {
    log_step "Installing advanced packages..."
    
    mount_chroot
    
    # Update package lists
    chroot_exec "apt-get update"
    
    # Install kernel and essential packages
    chroot_exec "DEBIAN_FRONTEND=noninteractive apt-get install -y \
        linux-image-generic \
        linux-headers-generic \
        grub-efi-amd64 \
        grub-pc-bin \
        systemd \
        systemd-sysv \
        network-manager \
        openssh-server \
        build-essential \
        python3 \
        python3-pip \
        python3-dev \
        python3-venv \
        nodejs \
        npm \
        git \
        curl \
        wget \
        nano \
        vim \
        htop \
        mesa-utils \
        libgl1-mesa-glx \
        libglu1-mesa \
        libegl1-mesa \
        libgles2-mesa \
        vulkan-tools \
        libvulkan1 \
        libvulkan-dev \
        libsdl2-2.0-0 \
        libsdl2-dev \
        ffmpeg \
        pulseaudio \
        libasound2-dev \
        libpulse-dev \
        fonts-liberation \
        fonts-dejavu-core \
        xserver-xorg-core \
        xserver-xorg-video-all \
        xinit \
        openbox \
        chromium-browser \
        plymouth \
        plymouth-theme-ubuntu-logo"
    
    # Install Python packages
    chroot_exec "pip3 install --upgrade pip setuptools wheel"
    chroot_exec "pip3 install \
        torch==2.1.0 \
        torchvision \
        torchaudio \
        transformers==4.35.0 \
        numpy \
        scipy \
        pandas \
        scikit-learn \
        matplotlib \
        opencv-python \
        pillow \
        flask \
        flask-socketio \
        python-socketio \
        eventlet \
        redis \
        celery \
        psutil \
        pydantic \
        fastapi \
        uvicorn"
    
    # Install Node.js packages globally
    chroot_exec "npm install -g \
        typescript \
        webpack \
        webpack-cli \
        @babel/core \
        @babel/preset-env \
        @babel/preset-react \
        react \
        react-dom \
        three \
        @react-three/fiber \
        @react-three/drei \
        socket.io-client \
        axios"
    
    unmount_chroot
    log_success "Packages installed"
}

# Create advanced SAM Core with weight persistence
create_advanced_sam_core() {
    log_step "Creating Advanced SAM Core v2.0..."
    
    cat > "${CHROOT_ENV}/usr/lib/sam/core/sam_core_v2.py" << 'EOFPYTHON'
#!/usr/bin/env python3
"""
SAM Core v2.0 - Advanced Implementation with Weight Persistence
Full neural architecture with dynamic growth and memory consolidation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import pickle
import logging
import threading
import time
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

# Configure advanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - SAM-%(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/sam/sam_core_v2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Core-v2')

class MemoryType(Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"

@dataclass
class Memory:
    """Advanced memory structure"""
    content: torch.Tensor
    type: MemoryType
    timestamp: float
    importance: float
    access_count: int = 0
    last_access: float = 0.0
    associations: List[int] = field(default_factory=list)
    decay_rate: float = 0.99

class DynamicConceptMemory(nn.Module):
    """Advanced concept memory with dynamic growth and consolidation"""
    
    def __init__(self, initial_size=10000, embed_dim=768, growth_rate=1.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.growth_rate = growth_rate
        self.current_size = initial_size
        
        # Multi-level embeddings
        self.char_embeddings = nn.Embedding(512, embed_dim)
        self.subword_embeddings = nn.Embedding(initial_size, embed_dim)
        self.word_embeddings = nn.Embedding(initial_size, embed_dim)
        self.phrase_embeddings = nn.Embedding(initial_size // 2, embed_dim)
        
        # Concept metadata
        self.concept_registry = {}
        self.concept_graph = defaultdict(set)
        self.concept_frequencies = torch.zeros(initial_size)
        self.concept_importance = torch.ones(initial_size)
        
        # Attention mechanisms
        self.self_attention = nn.MultiheadAttention(embed_dim, 8, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(embed_dim, 8, batch_first=True)
        
        # Memory consolidation network
        self.consolidation_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Initialize with linguistic priors
        self._initialize_linguistic_priors()
    
    def _initialize_linguistic_priors(self):
        """Initialize with linguistic knowledge"""
        # Common patterns
        patterns = [
            "the", "a", "an", "is", "are", "was", "were",
            "in", "on", "at", "to", "for", "with", "by",
            "and", "or", "but", "if", "then", "else"
        ]
        
        for idx, pattern in enumerate(patterns):
            self.concept_registry[pattern] = {
                "id": idx,
                "type": "linguistic",
                "frequency": 1000,
                "importance": 0.8
            }
    
    def grow_memory(self):
        """Dynamically grow memory capacity"""
        new_size = int(self.current_size * self.growth_rate)
        
        # Grow embeddings
        new_subword = nn.Embedding(new_size, self.embed_dim)
        new_word = nn.Embedding(new_size, self.embed_dim)
        
        # Copy existing weights
        with torch.no_grad():
            new_subword.weight[:self.current_size] = self.subword_embeddings.weight
            new_word.weight[:self.current_size] = self.word_embeddings.weight
        
        self.subword_embeddings = new_subword
        self.word_embeddings = new_word
        self.current_size = new_size
        
        # Extend tracking tensors
        self.concept_frequencies = F.pad(self.concept_frequencies, (0, new_size - len(self.concept_frequencies)))
        self.concept_importance = F.pad(self.concept_importance, (0, new_size - len(self.concept_importance)), value=1.0)
        
        logger.info(f"Memory grown to size: {new_size}")
    
    def consolidate_memory(self, memories: List[Memory]) -> torch.Tensor:
        """Consolidate multiple memories into unified representation"""
        if not memories:
            return torch.zeros(1, self.embed_dim)
        
        # Stack memory contents
        memory_tensors = torch.stack([m.content for m in memories])
        
        # Weight by importance and recency
        weights = torch.tensor([
            m.importance * (m.decay_rate ** (time.time() - m.timestamp))
            for m in memories
        ])
        weights = F.softmax(weights, dim=0)
        
        # Weighted average
        consolidated = (memory_tensors * weights.unsqueeze(-1)).sum(dim=0)
        
        # Apply consolidation network
        consolidated = self.consolidation_net(
            torch.cat([consolidated, consolidated], dim=-1)
        )
        
        return consolidated

class NeuralProcessor(nn.Module):
    """Advanced neural processor with modular architecture"""
    
    def __init__(self, embed_dim=768, num_layers=12, num_heads=12, ff_dim=3072):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Modular transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        
        # Specialized processors
        self.language_processor = LanguageProcessor(embed_dim)
        self.vision_processor = VisionProcessor(embed_dim)
        self.reasoning_processor = ReasoningProcessor(embed_dim)
        
        # Gating mechanisms
        self.processor_gates = nn.Linear(embed_dim, 3)
        
        # Output projections
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x, mask=None):
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Compute processor weights
        gates = F.softmax(self.processor_gates(x.mean(dim=1)), dim=-1)
        
        # Apply specialized processors
        lang_out = self.language_processor(x)
        vis_out = self.vision_processor(x)
        reason_out = self.reasoning_processor(x)
        
        # Weighted combination
        x = (gates[:, 0:1].unsqueeze(1) * lang_out +
             gates[:, 1:2].unsqueeze(1) * vis_out +
             gates[:, 2:3].unsqueeze(1) * reason_out)
        
        # Final projection
        x = self.output_norm(x)
        x = self.output_proj(x)
        
        return x

class TransformerBlock(nn.Module):
    """Enhanced transformer block with adaptive computation"""
    
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, mask=None):
        # Self-attention
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # Feed-forward
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        
        return x

class LanguageProcessor(nn.Module):
    """Specialized language processing module"""
    
    def __init__(self, embed_dim):
        super().__init__()
        self.syntax_encoder = nn.LSTM(embed_dim, embed_dim // 2, 2, 
                                     batch_first=True, bidirectional=True)
        self.semantic_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, 8, batch_first=True),
            num_layers=3
        )
        self.pragmatic_encoder = nn.GRU(embed_dim, embed_dim, 2, batch_first=True)
    
    def forward(self, x):
        # Syntactic processing
        syntax_out, _ = self.syntax_encoder(x)
        
        # Semantic processing
        semantic_out = self.semantic_encoder(x)
        
        # Pragmatic processing
        pragmatic_out, _ = self.pragmatic_encoder(x)
        
        # Combine
        return (syntax_out + semantic_out + pragmatic_out) / 3

class VisionProcessor(nn.Module):
    """Specialized vision processing module"""
    
    def __init__(self, embed_dim):
        super().__init__()
        self.spatial_encoder = nn.Conv1d(embed_dim, embed_dim, 3, padding=1)
        self.temporal_encoder = nn.Conv1d(embed_dim, embed_dim, 5, padding=2)
        self.object_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, 8, batch_first=True),
            num_layers=2
        )
    
    def forward(self, x):
        # Spatial processing
        spatial = self.spatial_encoder(x.transpose(1, 2)).transpose(1, 2)
        
        # Temporal processing
        temporal = self.temporal_encoder(x.transpose(1, 2)).transpose(1, 2)
        
        # Object processing
        objects = self.object_encoder(x)
        
        return (spatial + temporal + objects) / 3

class ReasoningProcessor(nn.Module):
    """Specialized reasoning module"""
    
    def __init__(self, embed_dim):
        super().__init__()
        self.logic_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.causal_encoder = nn.LSTM(embed_dim, embed_dim, 2, batch_first=True)
        self.abstraction_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, 8, batch_first=True),
            num_layers=2
        )
    
    def forward(self, x):
        # Logical reasoning
        logic = self.logic_encoder(x)
        
        # Causal reasoning
        causal, _ = self.causal_encoder(x)
        
        # Abstract reasoning
        abstract = self.abstraction_encoder(x)
        
        return (logic + causal + abstract) / 3

class SAMAgentCore(nn.Module):
    """Complete SAM agent with embodied cognition"""
    
    def __init__(self, agent_id: str, embed_dim=768, memory_size=10000):
        super().__init__()
        self.agent_id = agent_id
        self.embed_dim = embed_dim
        
        # Core components
        self.concept_memory = DynamicConceptMemory(memory_size, embed_dim)
        self.processor = NeuralProcessor(embed_dim)
        
        # Memory systems
        self.episodic_memory = deque(maxlen=1000)
        self.semantic_memory = {}
        self.procedural_memory = {}
        self.working_memory = deque(maxlen=50)
        
        # Embodiment
        self.sensory_encoder = nn.ModuleDict({
            'vision': nn.Linear(2048, embed_dim),
            'audio': nn.Linear(1024, embed_dim),
            'touch': nn.Linear(256, embed_dim),
            'proprioception': nn.Linear(128, embed_dim)
        })
        
        self.motor_decoder = nn.ModuleDict({
            'movement': nn.Linear(embed_dim, 64),
            'manipulation': nn.Linear(embed_dim, 32),
            'expression': nn.Linear(embed_dim, 128),
            'vocalization': nn.Linear(embed_dim, 256)
        })
        
        # Emotion and personality
        self.emotion_state = torch.zeros(8)  # 8 basic emotions
        self.personality_traits = torch.randn(5)  # Big 5 personality
        
        # Social cognition
        self.social_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, 8, batch_first=True),
            num_layers=3
        )
        
        # Learning and adaptation
        self.meta_learner = nn.LSTM(embed_dim, embed_dim, 2, batch_first=True)
        self.adaptation_rate = 0.01
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize with pretrained weights if available"""
        checkpoint_path = f"/var/lib/sam/checkpoints/{self.agent_id}.pth"
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            self.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded weights for agent {self.agent_id}")
    
    def perceive(self, sensory_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process sensory input"""
        encoded_inputs = []
        
        for modality, encoder in self.sensory_encoder.items():
            if modality in sensory_input:
                encoded = encoder(sensory_input[modality])
                encoded_inputs.append(encoded)
        
        if encoded_inputs:
            # Combine sensory inputs
            perception = torch.stack(encoded_inputs).mean(dim=0)
        else:
            perception = torch.zeros(1, self.embed_dim)
        
        # Store in working memory
        self.working_memory.append(Memory(
            content=perception,
            type=MemoryType.WORKING,
            timestamp=time.time(),
            importance=0.8
        ))
        
        return perception
    
    def think(self, perception: torch.Tensor) -> torch.Tensor:
        """Process perception through cognition"""
        # Retrieve relevant memories
        relevant_memories = self._retrieve_memories(perception)
        
        # Consolidate memories
        memory_context = self.concept_memory.consolidate_memory(relevant_memories)
        
        # Combine with current perception
        thought_input = torch.cat([perception, memory_context], dim=-1)
        thought_input = self.concept_memory.consolidation_net(thought_input)
        
        # Process through neural processor
        thought = self.processor(thought_input.unsqueeze(0))
        
        # Update emotion state
        self._update_emotions(thought)
        
        return thought.squeeze(0)
    
    def act(self, thought: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate motor actions"""
        actions = {}
        
        for action_type, decoder in self.motor_decoder.items():
            action = decoder(thought)
            actions[action_type] = action
        
        # Modulate by personality and emotion
        personality_factor = torch.sigmoid(self.personality_traits).mean()
        emotion_factor = torch.softmax(self.emotion_state, dim=0).max()
        
        for action_type in actions:
            actions[action_type] *= (personality_factor + emotion_factor) / 2
        
        return actions
    
    def communicate(self, thought: torch.Tensor, target_agent: Optional[str] = None) -> str:
        """Generate communication"""
        # Process through social cognition
        social_context = self.social_encoder(thought.unsqueeze(0))
        
        # Generate vocalization
        vocal_output = self.motor_decoder['vocalization'](social_context.squeeze(0))
        
        # Convert to text (simplified)
        message = self._vocalization_to_text(vocal_output)
        
        return message
    
    def learn(self, experience: Dict[str, Any]):
        """Learn from experience"""
        # Create episodic memory
        memory = Memory(
            content=torch.randn(self.embed_dim),  # Simplified
            type=MemoryType.EPISODIC,
            timestamp=time.time(),
            importance=experience.get('importance', 0.5)
        )
        
        self.episodic_memory.append(memory)
        
        # Update semantic memory
        if 'concept' in experience:
            concept = experience['concept']
            if concept not in self.semantic_memory:
                self.semantic_memory[concept] = []
            self.semantic_memory[concept].append(memory)
        
        # Meta-learning update
        with torch.no_grad():
            meta_input = torch.stack([m.content for m in list(self.episodic_memory)[-10:]])
            meta_output, _ = self.meta_learner(meta_input.unsqueeze(0))
            
            # Adaptive learning rate
            self.adaptation_rate *= 0.999  # Decay
            self.adaptation_rate = max(self.adaptation_rate, 0.001)
    
    def _retrieve_memories(self, query: torch.Tensor, k: int = 5) -> List[Memory]:
        """Retrieve relevant memories"""
        memories = []
        
        # Search episodic memory
        for memory in self.episodic_memory:
            similarity = F.cosine_similarity(query, memory.content, dim=0)
            memories.append((similarity.item(), memory))
        
        # Sort by relevance and recency
        memories.sort(key=lambda x: x[0] * (0.9 ** (time.time() - x[1].timestamp)), reverse=True)
        
        return [m[1] for m in memories[:k]]
    
    def _update_emotions(self, thought: torch.Tensor):
        """Update emotional state"""
        # Simple emotion dynamics
        emotion_delta = torch.randn(8) * 0.1
        self.emotion_state = torch.clamp(self.emotion_state + emotion_delta, -1, 1)
    
    def _vocalization_to_text(self, vocal_output: torch.Tensor) -> str:
        """Convert vocalization to text (simplified)"""
        # In reality, this would use a sophisticated decoder
        templates = [
            "Hello! I'm exploring this world.",
            "I sense something interesting nearby.",
            "Would you like to collaborate?",
            "I'm learning new patterns.",
            "This environment feels different.",
            "I'm adapting my neural pathways.",
            "Let's share our experiences.",
            "My cognition is evolving."
        ]
        
        idx = int(torch.argmax(vocal_output[:len(templates)]).item())
        return templates[idx % len(templates)]
    
    def save_state(self):
        """Save agent state"""
        checkpoint = {
            'state_dict': self.state_dict(),
            'agent_id': self.agent_id,
            'personality': self.personality_traits,
            'memories': {
                'episodic': list(self.episodic_memory),
                'semantic': self.semantic_memory
            }
        }
        
        path = f"/var/lib/sam/checkpoints/{self.agent_id}.pth"
        torch.save(checkpoint, path)
        logger.info(f"Saved state for agent {self.agent_id}")

# Global SAM instance manager
class SAMManager:
    """Manage multiple SAM agents"""
    
    def __init__(self):
        self.agents = {}
        self.shared_memory = {}
        self.communication_bus = deque(maxlen=1000)
        self.lock = threading.Lock()
    
    def create_agent(self, agent_id: str, **kwargs) -> SAMAgentCore:
        """Create new SAM agent"""
        with self.lock:
            if agent_id in self.agents:
                return self.agents[agent_id]
            
            agent = SAMAgentCore(agent_id, **kwargs)
            self.agents[agent_id] = agent
            
            logger.info(f"Created SAM agent: {agent_id}")
            return agent
    
    def get_agent(self, agent_id: str) -> Optional[SAMAgentCore]:
        """Get existing agent"""
        return self.agents.get(agent_id)
    
    def broadcast_message(self, sender_id: str, message: str, target_id: Optional[str] = None):
        """Broadcast message between agents"""
        msg = {
            'sender': sender_id,
            'target': target_id,
            'message': message,
            'timestamp': time.time()
        }
        
        with self.lock:
            self.communication_bus.append(msg)
            
            # Direct message
            if target_id and target_id in self.agents:
                agent = self.agents[target_id]
                # Process message (simplified)
                logger.info(f"Message from {sender_id} to {target_id}: {message}")
    
    def share_memory(self, agent_id: str, memory_key: str, memory_content: Any):
        """Share memory between agents"""
        with self.lock:
            if memory_key not in self.shared_memory:
                self.shared_memory[memory_key] = {}
            
            self.shared_memory[memory_key][agent_id] = memory_content
    
    def get_shared_memory(self, memory_key: str) -> Dict[str, Any]:
        """Get shared memory"""
        return self.shared_memory.get(memory_key, {})

# Global instance
sam_manager = SAMManager()

# API functions
def initialize_sam_agent(agent_id: str, **kwargs) -> bool:
    """Initialize a SAM agent"""
    try:
        sam_manager.create_agent(agent_id, **kwargs)
        return True
    except Exception as e:
        logger.error(f"Failed to initialize agent {agent_id}: {e}")
        return False

def get_sam_response(agent_id: str, input_text: str) -> str:
    """Get response from specific SAM agent"""
    agent = sam_manager.get_agent(agent_id)
    if not agent:
        return "Agent not found"
    
    try:
        # Simplified processing
        perception = torch.randn(1, agent.embed_dim)
        thought = agent.think(perception)
        response = agent.communicate(thought)
        return response
    except Exception as e:
        logger.error(f"Error getting response from {agent_id}: {e}")
        return "Processing error"

if __name__ == "__main__":
    # Test advanced SAM
    print("Testing SAM Core v2.0...")
    
    # Create test agent
    agent = sam_manager.create_agent("test_agent")
    
    # Test perception
    sensory_input = {
        'vision': torch.randn(1, 2048),
        'audio': torch.randn(1, 1024)
    }
    
    perception = agent.perceive(sensory_input)
    print(f"Perception shape: {perception.shape}")
    
    # Test thinking
    thought = agent.think(perception)
    print(f"Thought shape: {thought.shape}")
    
    # Test action
    actions = agent.act(thought)
    print(f"Actions: {list(actions.keys())}")
    
    # Test communication
    message = agent.communicate(thought)
    print(f"Message: {message}")
    
    # Save state
    agent.save_state()
    print("Agent state saved")
EOFPYTHON

    log_success "Advanced SAM Core v2.0 created"
}

# Create SYNTHIX Universe Engine with full physics
create_universe_engine_v2() {
    log_step "Creating SYNTHIX Universe Engine v2.0..."
    
    cat > "${CHROOT_ENV}/usr/lib/synthix/universe/universe_engine_v2.py" << 'EOFPYTHON'
#!/usr/bin/env python3
"""
SYNTHIX Universe Engine v2.0 - Full Physics Simulation
SIMS-like world with embodied SAM agents
"""

import numpy as np
import torch
import json
import time
import threading
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import websockets
from collections import defaultdict
import math

# Import SAM
import sys
sys.path.append('/usr/lib/sam/core')
from sam_core_v2 import sam_manager, SAMAgentCore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Universe-v2')

class PhysicsConstants:
    """Physical constants for the universe"""
    GRAVITY = 9.81
    AIR_RESISTANCE = 0.1
    FRICTION = 0.3
    LIGHT_SPEED = 299792458
    TIME_SCALE = 1.0
    COLLISION_THRESHOLD = 0.5
    INTERACTION_RANGE = 5.0

@dataclass
class Vector3:
    """3D vector for physics calculations"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self):
        mag = self.magnitude()
        if mag > 0:
            return Vector3(self.x/mag, self.y/mag, self.z/mag)
        return Vector3()
    
    def to_dict(self):
        return {'x': self.x, 'y': self.y, 'z': self.z}

@dataclass
class Transform:
    """Transform component for entities"""
    position: Vector3 = field(default_factory=Vector3)
    rotation: Vector3 = field(default_factory=Vector3)
    scale: Vector3 = field(default_factory=lambda: Vector3(1, 1, 1))

@dataclass
class PhysicsBody:
    """Physics body component"""
    velocity: Vector3 = field(default_factory=Vector3)
    acceleration: Vector3 = field(default_factory=Vector3)
    mass: float = 1.0
    drag: float = 0.1
    restitution: float = 0.5
    is_static: bool = False
    is_kinematic: bool = False

@dataclass
class Collider:
    """Collision component"""
    shape: str = "sphere"  # sphere, box, capsule
    size: Vector3 = field(default_factory=lambda: Vector3(1, 1, 1))
    is_trigger: bool = False
    layers: int = 1  # Collision layers bitmask

class EntityType(Enum):
    SAM_AGENT = "sam_agent"
    OBJECT = "object"
    FURNITURE = "furniture"
    FOOD = "food"
    TOOL = "tool"
    BUILDING = "building"
    NATURE = "nature"
    PARTICLE = "particle"

@dataclass
class Entity:
    """Base entity in the universe"""
    id: str
    type: EntityType
    name: str
    transform: Transform = field(default_factory=Transform)
    physics: Optional[PhysicsBody] = None
    collider: Optional[Collider] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    components: Dict[str, Any] = field(default_factory=dict)
    active: bool = True
    visible: bool = True

class SAMEntity(Entity):
    """SAM agent entity with full embodiment"""
    
    def __init__(self, id: str, name: str, position: Vector3):
        super().__init__(
            id=id,
            type=EntityType.SAM_AGENT,
            name=name,
            transform=Transform(position=position),
            physics=PhysicsBody(mass=70.0, drag=0.5),
            collider=Collider(shape="capsule", size=Vector3(0.5, 1.8, 0.5))
        )
        
        # Create SAM agent core
        self.sam_agent = sam_manager.create_agent(id)
        
        # Agent state
        self.health = 100.0
        self.energy = 100.0
        self.hunger = 0.0
        self.happiness = 50.0
        self.social_need = 30.0
        
        # Inventory
        self.inventory = []
        self.equipped_tool = None
        
        # Current action
        self.current_action = None
        self.action_target = None
        
        # Relationships
        self.relationships = {}
        
        # Visual properties
        self.properties['appearance'] = {
            'model': 'humanoid',
            'skin_color': np.random.choice(['light', 'medium', 'dark']),
            'hair_style': np.random.choice(['short', 'long', 'medium']),
            'hair_color': np.random.choice(['black', 'brown', 'blonde', 'red']),
            'clothing': 'casual',
            'accessories': []
        }
        
        # Animation state
        self.animation_state = 'idle'
        self.animation_blend = {}
    
    def update_needs(self, dt: float):
        """Update agent needs"""
        self.energy -= dt * 0.5
        self.hunger += dt * 0.3
        self.social_need += dt * 0.2
        
        # Clamp values
        self.energy = max(0, min(100, self.energy))
        self.hunger = max(0, min(100, self.hunger))
        self.social_need = max(0, min(100, self.social_need))
        
        # Update happiness based on needs
        self.happiness = 100 - (self.hunger * 0.3 + (100 - self.energy) * 0.2 + self.social_need * 0.5)
        self.happiness = max(0, min(100, self.happiness))
    
    def perceive_environment(self, nearby_entities: List[Entity]) -> Dict[str, torch.Tensor]:
        """Generate sensory input from environment"""
        # Vision
        vision_data = []
        for entity in nearby_entities[:20]:  # Limit to 20 nearest
            relative_pos = entity.transform.position - self.transform.position
            distance = relative_pos.magnitude()
            
            vision_data.extend([
                relative_pos.x, relative_pos.y, relative_pos.z,
                distance,
                float(entity.type.value == 'sam_agent'),
                float(entity.type.value == 'food'),
                float(entity.type.value == 'tool')
            ])
        
        # Pad to fixed size
        vision_data.extend([0] * (140 - len(vision_data)))
        vision_tensor = torch.tensor(vision_data[:140]).unsqueeze(0)
        
        # Audio (simplified - based on nearby agents)
        audio_data = []
        for entity in nearby_entities:
            if entity.type == EntityType.SAM_AGENT and entity.id != self.id:
                distance = (entity.transform.position - self.transform.position).magnitude()
                if distance < 10:
                    audio_data.append(1.0 / (1 + distance))
        
        audio_tensor = torch.tensor(audio_data[:32] + [0] * (32 - len(audio_data))).unsqueeze(0)
        
        # Proprioception
        proprio_data = [
            self.health / 100,
            self.energy / 100,
            self.hunger / 100,
            self.happiness / 100,
            self.physics.velocity.x,
            self.physics.velocity.y,
            self.physics.velocity.z,
            len(self.inventory) / 10
        ]
        proprio_tensor = torch.tensor(proprio_data).unsqueeze(0)
        
        return {
            'vision': vision_tensor,
            'audio': audio_tensor,
            'proprioception': proprio_tensor
        }
    
    def decide_action(self, perception: torch.Tensor) -> Dict[str, Any]:
        """Decide next action based on perception and needs"""
        # Get thought from SAM
        thought = self.sam_agent.think(perception)
        
        # Get motor actions
        actions = self.sam_agent.act(thought)
        
        # Priority-based action selection
        if self.hunger > 70:
            return {'type': 'seek_food', 'urgency': self.hunger / 100}
        elif self.energy < 30:
            return {'type': 'rest', 'urgency': (100 - self.energy) / 100}
        elif self.social_need > 70:
            return {'type': 'socialize', 'urgency': self.social_need / 100}
        else:
            # Explore or work
            if np.random.random() < 0.3:
                return {'type': 'explore', 'urgency': 0.5}
            else:
                return {'type': 'work', 'urgency': 0.6}
    
    def execute_action(self, action: Dict[str, Any], dt: float, nearby_entities: List[Entity]):
        """Execute decided action"""
        action_type = action['type']
        
        if action_type == 'seek_food':
            # Find nearest food
            food_entities = [e for e in nearby_entities if e.type == EntityType.FOOD]
            if food_entities:
                nearest_food = min(food_entities, 
                    key=lambda e: (e.transform.position - self.transform.position).magnitude())
                self.move_towards(nearest_food.transform.position, dt)
                
                # Eat if close enough
                if (nearest_food.transform.position - self.transform.position).magnitude() < 1.0:
                    self.hunger = max(0, self.hunger - 50)
                    self.energy = min(100, self.energy + 20)
                    # Remove food entity
                    nearest_food.active = False
        
        elif action_type == 'rest':
            # Find a place to rest
            self.animation_state = 'sitting'
            self.energy = min(100, self.energy + dt * 10)
        
        elif action_type == 'socialize':
            # Find nearest agent
            agents = [e for e in nearby_entities if e.type == EntityType.SAM_AGENT and e.id != self.id]
            if agents:
                nearest_agent = min(agents,
                    key=lambda e: (e.transform.position - self.transform.position).magnitude())
                self.move_towards(nearest_agent.transform.position, dt)
                
                # Interact if close enough
                if (nearest_agent.transform.position - self.transform.position).magnitude() < 2.0:
                    self.interact_with(nearest_agent)
        
        elif action_type == 'explore':
            # Random walk
            if not hasattr(self, 'explore_target') or np.random.random() < 0.01:
                angle = np.random.random() * 2 * np.pi
                distance = np.random.random() * 20 + 5
                self.explore_target = Vector3(
                    self.transform.position.x + np.cos(angle) * distance,
                    self.transform.position.y,
                    self.transform.position.z + np.sin(angle) * distance
                )
            
            self.move_towards(self.explore_target, dt)
        
        elif action_type == 'work':
            # Find something to work on
            self.animation_state = 'working'
            # Generate resources or improve environment
            pass
    
    def move_towards(self, target: Vector3, dt: float):
        """Move towards target position"""
        direction = (target - self.transform.position).normalize()
        
        # Set velocity
        move_speed = 5.0 if self.energy > 50 else 3.0
        self.physics.velocity = direction * move_speed
        
        # Update animation
        self.animation_state = 'walking' if move_speed < 4 else 'running'
        
        # Face direction
        if direction.magnitude() > 0:
            self.transform.rotation.y = math.atan2(direction.x, direction.z)
    
    def interact_with(self, other_agent: 'SAMEntity'):
        """Social interaction with another agent"""
        # Generate communication
        message = self.sam_agent.communicate(
            torch.randn(1, self.sam_agent.embed_dim),
            target_agent=other_agent.id
        )
        
        # Update relationship
        if other_agent.id not in self.relationships:
            self.relationships[other_agent.id] = 0
        
        self.relationships[other_agent.id] += 5
        self.social_need = max(0, self.social_need - 20)
        
        # Share experience
        sam_manager.broadcast_message(self.id, message, other_agent.id)
        
        # Animation
        self.animation_state = 'talking'

class UniverseGrid:
    """Spatial partitioning for efficient physics"""
    
    def __init__(self, size: Tuple[float, float, float], cell_size: float = 10.0):
        self.size = size
        self.cell_size = cell_size
        self.grid = defaultdict(list)
    
    def clear(self):
        self.grid.clear()
    
    def add_entity(self, entity: Entity):
        """Add entity to grid"""
        cell = self._get_cell(entity.transform.position)
        self.grid[cell].append(entity)
    
    def get_nearby_entities(self, position: Vector3, radius: float) -> List[Entity]:
        """Get entities near position"""
        nearby = []
        
        # Check cells in radius
        min_cell = self._get_cell(position - Vector3(radius, radius, radius))
        max_cell = self._get_cell(position + Vector3(radius, radius, radius))
        
        for x in range(min_cell[0], max_cell[0] + 1):
            for y in range(min_cell[1], max_cell[1] + 1):
                for z in range(min_cell[2], max_cell[2] + 1):
                    cell = (x, y, z)
                    if cell in self.grid:
                        for entity in self.grid[cell]:
                            if (entity.transform.position - position).magnitude() <= radius:
                                nearby.append(entity)
        
        return nearby
    
    def _get_cell(self, position: Vector3) -> Tuple[int, int, int]:
        """Get grid cell for position"""
        return (
            int(position.x / self.cell_size),
            int(position.y / self.cell_size),
            int(position.z / self.cell_size)
        )

class Universe:
    """SYNTHIX Universe with full physics and SAM agents"""
    
    def __init__(self, name: str, size: Tuple[float, float, float] = (1000, 100, 1000)):
        self.name = name
        self.size = size
        self.entities = {}
        self.sam_entities = {}
        self.grid = UniverseGrid(size)
        
        # Physics settings
        self.gravity = Vector3(0, -PhysicsConstants.GRAVITY, 0)
        self.time_scale = PhysicsConstants.TIME_SCALE
        
        # Environment
        self.time_of_day = 0.5  # 0-1, 0.5 = noon
        self.weather = 'clear'
        self.temperature = 20.0
        
        # Statistics
        self.stats = {
            'total_entities': 0,
            'sam_agents': 0,
            'interactions': 0,
            'births': 0,
            'deaths': 0
        }
        
        # Event system
        self.events = []
        
        # Running state
        self.running = False
        self.thread = None
        self.last_update = time.time()
        
        logger.info(f"Created universe: {name}")
    
    def add_entity(self, entity: Entity):
        """Add entity to universe"""
        self.entities[entity.id] = entity
        
        if isinstance(entity, SAMEntity):
            self.sam_entities[entity.id] = entity
            self.stats['sam_agents'] += 1
        
        self.stats['total_entities'] += 1
        
        logger.info(f"Added entity: {entity.id} ({entity.type.value})")
    
    def remove_entity(self, entity_id: str):
        """Remove entity from universe"""
        if entity_id in self.entities:
            entity = self.entities[entity_id]
            
            if isinstance(entity, SAMEntity):
                del self.sam_entities[entity_id]
                self.stats['sam_agents'] -= 1
            
            del self.entities[entity_id]
            self.stats['total_entities'] -= 1
    
    def spawn_sam_agent(self, name: str, position: Vector3) -> SAMEntity:
        """Spawn new SAM agent"""
        agent_id = f"sam_{len(self.sam_entities)}_{int(time.time())}"
        agent = SAMEntity(agent_id, name, position)
        
        self.add_entity(agent)
        self.stats['births'] += 1
        
        self.events.append({
            'type': 'agent_spawn',
            'agent_id': agent_id,
            'position': position.to_dict(),
            'timestamp': time.time()
        })
        
        return agent
    
    def spawn_object(self, obj_type: EntityType, position: Vector3, properties: Dict[str, Any] = None):
        """Spawn object in universe"""
        obj_id = f"{obj_type.value}_{len(self.entities)}_{int(time.time())}"
        
        entity = Entity(
            id=obj_id,
            type=obj_type,
            name=f"{obj_type.value}_{len(self.entities)}",
            transform=Transform(position=position),
            properties=properties or {}
        )
        
        # Add physics for some objects
        if obj_type in [EntityType.FOOD, EntityType.TOOL]:
            entity.physics = PhysicsBody(mass=1.0)
            entity.collider = Collider(shape="sphere", size=Vector3(0.5, 0.5, 0.5))
        
        self.add_entity(entity)
        return entity
    
    def update(self, dt: float):
        """Update universe simulation"""
        # Update time
        self.time_of_day = (self.time_of_day + dt / 86400) % 1.0
        
        # Clear spatial grid
        self.grid.clear()
        
        # Add all entities to grid
        for entity in self.entities.values():
            if entity.active:
                self.grid.add_entity(entity)
        
        # Update physics
        self._update_physics(dt)
        
        # Update SAM agents
        self._update_agents(dt)
        
        # Handle collisions
        self._handle_collisions()
        
        # Spawn new entities
        self._spawn_entities(dt)
        
        # Update environment
        self._update_environment(dt)
    
    def _update_physics(self, dt: float):
        """Update physics for all entities"""
        for entity in self.entities.values():
            if not entity.active or not entity.physics:
                continue
            
            if entity.physics.is_static:
                continue
            
            # Apply gravity
            if not entity.physics.is_kinematic:
                entity.physics.acceleration = self.gravity * entity.physics.mass
            
            # Apply drag
            drag_force = entity.physics.velocity * (-entity.physics.drag)
            entity.physics.acceleration = entity.physics.acceleration + drag_force
            
            # Update velocity
            entity.physics.velocity = entity.physics.velocity + (entity.physics.acceleration * dt)
            
            # Update position
            entity.transform.position = entity.transform.position + (entity.physics.velocity * dt)
            
            # Ground collision (simple)
            if entity.transform.position.y < 0:
                entity.transform.position.y = 0
                entity.physics.velocity.y = 0
    
    def _update_agents(self, dt: float):
        """Update all SAM agents"""
        for agent in self.sam_entities.values():
            if not agent.active:
                continue
            
            # Update needs
            agent.update_needs(dt)
            
            # Get nearby entities
            nearby = self.grid.get_nearby_entities(
                agent.transform.position,
                PhysicsConstants.INTERACTION_RANGE * 3
            )
            
            # Perceive environment
            sensory_input = agent.perceive_environment(nearby)
            perception = agent.sam_agent.perceive(sensory_input)
            
            # Decide action
            action = agent.decide_action(perception)
            
            # Execute action
            agent.execute_action(action, dt, nearby)
            
            # Learn from experience
            if np.random.random() < 0.1:  # Learn periodically
                experience = {
                    'perception': perception,
                    'action': action,
                    'reward': agent.happiness / 100,
                    'importance': action.get('urgency', 0.5)
                }
                agent.sam_agent.learn(experience)
    
    def _handle_collisions(self):
        """Handle entity collisions"""
        checked_pairs = set()
        
        for entity_id, entity in self.entities.items():
            if not entity.active or not entity.collider:
                continue
            
            # Get nearby entities
            nearby = self.grid.get_nearby_entities(
                entity.transform.position,
                PhysicsConstants.COLLISION_THRESHOLD * 2
            )
            
            for other in nearby:
                if other.id == entity_id or not other.collider:
                    continue
                
                # Skip if already checked
                pair = tuple(sorted([entity_id, other.id]))
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)
                
                # Check collision
                distance = (entity.transform.position - other.transform.position).magnitude()
                collision_distance = (entity.collider.size.x + other.collider.size.x) / 2
                
                if distance < collision_distance:
                    self._resolve_collision(entity, other)
    
    def _resolve_collision(self, entity1: Entity, entity2: Entity):
        """Resolve collision between entities"""
        # Calculate collision normal
        normal = (entity2.transform.position - entity1.transform.position).normalize()
        
        # Separate entities
        overlap = ((entity1.collider.size.x + entity2.collider.size.x) / 2) - \
                  (entity2.transform.position - entity1.transform.position).magnitude()
        
        if entity1.physics and not entity1.physics.is_static:
            entity1.transform.position = entity1.transform.position - (normal * (overlap / 2))
        
        if entity2.physics and not entity2.physics.is_static:
            entity2.transform.position = entity2.transform.position + (normal * (overlap / 2))
        
        # Apply impulse if both have physics
        if entity1.physics and entity2.physics:
            # Calculate relative velocity
            relative_velocity = entity2.physics.velocity - entity1.physics.velocity
            velocity_along_normal = relative_velocity.x * normal.x + \
                                   relative_velocity.y * normal.y + \
                                   relative_velocity.z * normal.z
            
            # Don't resolve if velocities are separating
            if velocity_along_normal > 0:
                return
            
            # Calculate impulse scalar
            e = min(entity1.physics.restitution, entity2.physics.restitution)
            j = -(1 + e) * velocity_along_normal
            j /= 1/entity1.physics.mass + 1/entity2.physics.mass
            
            # Apply impulse
            impulse = normal * j
            entity1.physics.velocity = entity1.physics.velocity - (impulse * (1/entity1.physics.mass))
            entity2.physics.velocity = entity2.physics.velocity + (impulse * (1/entity2.physics.mass))
    
    def _spawn_entities(self, dt: float):
        """Spawn new entities periodically"""
        # Spawn food
        if np.random.random() < 0.01:
            pos = Vector3(
                np.random.uniform(0, self.size[0]),
                0,
                np.random.uniform(0, self.size[2])
            )
            self.spawn_object(EntityType.FOOD, pos, {'calories': 50})
        
        # Spawn new agents occasionally
        if len(self.sam_entities) < 20 and np.random.random() < 0.001:
            pos = Vector3(
                np.random.uniform(0, self.size[0]),
                0,
                np.random.uniform(0, self.size[2])
            )
            self.spawn_sam_agent(f"Agent_{len(self.sam_entities)}", pos)
    
    def _update_environment(self, dt: float):
        """Update environmental conditions"""
        # Weather changes
        if np.random.random() < 0.0001:
            self.weather = np.random.choice(['clear', 'cloudy', 'rain', 'storm'])
        
        # Temperature fluctuation
        target_temp = 20 + 10 * math.sin(self.time_of_day * 2 * math.pi)
        self.temperature += (target_temp - self.temperature) * dt * 0.1
    
    def get_state(self) -> Dict[str, Any]:
        """Get current universe state for rendering"""
        state = {
            'name': self.name,
            'time_of_day': self.time_of_day,
            'weather': self.weather,
            'temperature': self.temperature,
            'stats': self.stats.copy(),
            'entities': []
        }
        
        # Add entity states
        for entity in self.entities.values():
            if not entity.active:
                continue
            
            entity_data = {
                'id': entity.id,
                'type': entity.type.value,
                'name': entity.name,
                'position': entity.transform.position.to_dict(),
                'rotation': entity.transform.rotation.to_dict(),
                'scale': entity.transform.scale.to_dict(),
                'properties': entity.properties
            }
            
            # Add SAM-specific data
            if isinstance(entity, SAMEntity):
                entity_data.update({
                    'health': entity.health,
                    'energy': entity.energy,
                    'hunger': entity.hunger,
                    'happiness': entity.happiness,
                    'animation_state': entity.animation_state,
                    'relationships': len(entity.relationships)
                })
            
            state['entities'].append(entity_data)
        
        return state
    
    def start(self):
        """Start universe simulation"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run_loop)
        self.thread.start()
        
        logger.info(f"Universe {self.name} started")
    
    def stop(self):
        """Stop universe simulation"""
        self.running = False
        if self.thread:
            self.thread.join()
        
        # Save all agent states
        for agent in self.sam_entities.values():
            agent.sam_agent.save_state()
        
        logger.info(f"Universe {self.name} stopped")
    
    def _run_loop(self):
        """Main simulation loop"""
        while self.running:
            current_time = time.time()
            dt = (current_time - self.last_update) * self.time_scale
            self.last_update = current_time
            
            # Update simulation
            self.update(dt)
            
            # Target 60 FPS
            sleep_time = max(0, (1/60) - (time.time() - current_time))
            time.sleep(sleep_time)

# WebSocket server for real-time updates
class UniverseServer:
    """WebSocket server for universe visualization"""
    
    def __init__(self, universe: Universe, port: int = 8765):
        self.universe = universe
        self.port = port
        self.clients = set()
    
    async def handler(self, websocket, path):
        """Handle WebSocket connections"""
        self.clients.add(websocket)
        try:
            # Send initial state
            await websocket.send(json.dumps({
                'type': 'init',
                'data': self.universe.get_state()
            }))
            
            # Keep connection alive
            async for message in websocket:
                data = json.loads(message)
                await self.handle_message(websocket, data)
        finally:
            self.clients.remove(websocket)
    
    async def handle_message(self, websocket, data):
        """Handle client messages"""
        msg_type = data.get('type')
        
        if msg_type == 'spawn_agent':
            pos = Vector3(
                data.get('x', 0),
                data.get('y', 0),
                data.get('z', 0)
            )
            agent = self.universe.spawn_sam_agent(data.get('name', 'NewAgent'), pos)
            
            await websocket.send(json.dumps({
                'type': 'agent_spawned',
                'data': {'id': agent.id}
            }))
    
    async def broadcast_updates(self):
        """Broadcast universe updates to all clients"""
        while True:
            if self.clients:
                state = self.universe.get_state()
                message = json.dumps({
                    'type': 'update',
                    'data': state
                })
                
                await asyncio.gather(
                    *[client.send(message) for client in self.clients],
                    return_exceptions=True
                )
            
            await asyncio.sleep(1/30)  # 30 FPS updates
    
    async def start(self):
        """Start WebSocket server"""
        async with websockets.serve(self.handler, "localhost", self.port):
            await self.broadcast_updates()

# Main entry point
def main():
    # Create universe
    universe = Universe("Genesis", size=(100, 50, 100))
    
    # Spawn initial agents
    for i in range(5):
        pos = Vector3(
            np.random.uniform(10, 90),
            0,
            np.random.uniform(10, 90)
        )
        universe.spawn_sam_agent(f"Pioneer_{i}", pos)
    
    # Spawn initial objects
    for i in range(20):
        pos = Vector3(
            np.random.uniform(0, 100),
            0,
            np.random.uniform(0, 100)
        )
        universe.spawn_object(EntityType.FOOD, pos)
    
    # Start simulation
    universe.start()
    
    # Start WebSocket server
    server = UniverseServer(universe)
    asyncio.run(server.start())

if __name__ == "__main__":
    main()
EOFPYTHON

    log_success "Universe Engine v2.0 created"
}

# Create advanced 3D visualization interface
create_3d_visualization() {
    log_step "Creating advanced 3D visualization interface..."
    
    # Create main HTML file
    cat > "${CHROOT_ENV}/usr/share/synthix/web/index.html" << 'EOFHTML'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SYNTHIX-SAM Universe v2.0 - Living AI World</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            overflow: hidden;
            background: #000;
            color: #fff;
        }
        
        #loading-screen {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, #1a1a2e 0%, #0f0f23 100%);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 10000;
            transition: opacity 1s ease;
        }
        
        .loading-logo {
            width: 200px;
            height: 200px;
            position: relative;
            margin-bottom: 40px;
        }
        
        .loading-logo::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 100%;
            height: 100%;
            border: 3px solid transparent;
            border-top-color: #00ff88;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            transform: translate(-50%, -50%);
        }
        
        .loading-logo::after {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 80%;
            height: 80%;
            border: 3px solid transparent;
            border-bottom-color: #ff6b35;
            border-radius: 50%;
            animation: spin 1.5s linear reverse infinite;
            transform: translate(-50%, -50%);
        }
        
        @keyframes spin {
            0% { transform: translate(-50%, -50%) rotate(0deg); }
            100% { transform: translate(-50%, -50%) rotate(360deg); }
        }
        
        .loading-text {
            font-size: 2em;
            font-weight: 300;
            margin-bottom: 20px;
            background: linear-gradient(45deg, #00ff88, #4ecdc4, #ff6b35);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradient 3s ease infinite;
        }
        
        @keyframes gradient {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
        
        .loading-progress {
            width: 300px;
            height: 4px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 2px;
            overflow: hidden;
        }
        
        .loading-progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #00ff88, #4ecdc4, #ff6b35);
            border-radius: 2px;
            animation: loading 3s ease-in-out infinite;
        }
        
        @keyframes loading {
            0% { width: 0%; }
            50% { width: 70%; }
            100% { width: 100%; }
        }
        
        #universe-container {
            width: 100vw;
            height: 100vh;
            position: relative;
        }
        
        #render-canvas {
            width: 100%;
            height: 100%;
            display: block;
        }
        
        .ui-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 100;
        }
        
        .ui-element {
            position: absolute;
            pointer-events: auto;
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            color: #fff;
        }
        
        .top-bar {
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 30px;
            align-items: center;
        }
        
        .universe-name {
            font-size: 1.5em;
            font-weight: 600;
            background: linear-gradient(45deg, #00ff88, #4ecdc4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .time-display {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .time-icon {
            width: 24px;
            height: 24px;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="%2300ff88"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm.5-13H11v6l5.25 3.15.75-1.23-4.5-2.67z"/></svg>') center/contain no-repeat;
        }
        
        .stats-panel {
            top: 20px;
            right: 20px;
            min-width: 300px;
        }
        
        .stats-title {
            font-size: 1.2em;
            margin-bottom: 15px;
            color: #00ff88;
        }
        
        .stat-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .stat-label {
            color: rgba(255, 255, 255, 0.7);
        }
        
        .stat-value {
            font-weight: 600;
            color: #4ecdc4;
        }
        
        .agent-inspector {
            bottom: 20px;
            left: 20px;
            min-width: 350px;
            max-width: 400px;
            max-height: 50vh;
            overflow-y: auto;
            display: none;
        }
        
        .agent-header {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .agent-avatar {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: linear-gradient(135deg, #00ff88, #4ecdc4);
            margin-right: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5em;
            font-weight: bold;
        }
        
        .agent-info {
            flex: 1;
        }
        
        .agent-name {
            font-size: 1.2em;
            font-weight: 600;
            margin-bottom: 5px;
        }
        
        .agent-id {
            font-size: 0.9em;
            color: rgba(255, 255, 255, 0.5);
        }
        
        .needs-container {
            margin-bottom: 20px;
        }
        
        .needs-title {
            font-size: 1.1em;
            margin-bottom: 10px;
            color: #ff6b35;
        }
        
        .need-bar {
            margin-bottom: 10px;
        }
        
        .need-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
            font-size: 0.9em;
        }
        
        .need-progress {
            width: 100%;
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            overflow: hidden;
        }
        
        .need-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }
        
        .need-fill.health { background: #00ff88; }
        .need-fill.energy { background: #4ecdc4; }
        .need-fill.hunger { background: #ff6b35; }
        .need-fill.happiness { background: #ffd93d; }
        
        .controls-panel {
            bottom: 20px;
            right: 20px;
            display: flex;
            gap: 10px;
        }
        
        .control-btn {
            padding: 12px 24px;
            background: rgba(0, 255, 136, 0.1);
            border: 1px solid #00ff88;
            border-radius: 8px;
            color: #00ff88;
            font-size: 1em;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .control-btn:hover {
            background: rgba(0, 255, 136, 0.2);
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(0, 255, 136, 0.3);
        }
        
        .control-btn.active {
            background: #00ff88;
            color: #000;
        }
        
        .minimap {
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            width: 200px;
            height: 200px;
            border: 2px solid rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            overflow: hidden;
        }
        
        #minimap-canvas {
            width: 100%;
            height: 100%;
        }
        
        .chat-panel {
            top: 50%;
            right: 20px;
            transform: translateY(-50%);
            width: 300px;
            max-height: 400px;
            display: flex;
            flex-direction: column;
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            margin-bottom: 10px;
            padding-right: 10px;
        }
        
        .chat-message {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 10px;
            animation: slideIn 0.3s ease;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        .chat-sender {
            font-weight: 600;
            color: #4ecdc4;
            margin-bottom: 5px;
        }
        
        .chat-text {
            font-size: 0.9em;
            line-height: 1.4;
        }
        
        .weather-indicator {
            position: absolute;
            top: 80px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            align-items: center;
            gap: 10px;
            pointer-events: auto;
        }
        
        .weather-icon {
            width: 40px;
            height: 40px;
        }
        
        .weather-text {
            font-size: 1.1em;
            text-transform: capitalize;
        }
        
        .tooltip {
            position: absolute;
            background: rgba(0, 0, 0, 0.9);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            padding: 10px 15px;
            font-size: 0.9em;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.3s ease;
            z-index: 1000;
        }
        
        .tooltip.visible {
            opacity: 1;
        }
        
        /* Mobile responsive */
        @media (max-width: 768px) {
            .stats-panel {
                top: auto;
                bottom: 100px;
                right: 10px;
                min-width: 250px;
            }
            
            .agent-inspector {
                left: 10px;
                right: 10px;
                min-width: auto;
                max-width: none;
            }
            
            .chat-panel {
                display: none;
            }
        }
    </style>
</head>
<body>
    <div id="loading-screen">
        <div class="loading-logo"></div>
        <div class="loading-text">SYNTHIX-SAM Universe</div>
        <div class="loading-progress">
            <div class="loading-progress-bar"></div>
        </div>
        <div style="margin-top: 20px; color: #666;">Initializing living AI world...</div>
    </div>
    
    <div id="universe-container">
        <canvas id="render-canvas"></canvas>
        
        <div class="ui-overlay">
            <div class="ui-element top-bar">
                <div class="universe-name">Genesis Universe</div>
                <div class="time-display">
                    <div class="time-icon"></div>
                    <span id="time-text">12:00 PM</span>
                </div>
            </div>
            
            <div class="weather-indicator">
                <canvas class="weather-icon" id="weather-icon"></canvas>
                <span class="weather-text" id="weather-text">Clear</span>
            </div>
            
            <div class="ui-element stats-panel">
                <div class="stats-title">Universe Statistics</div>
                <div class="stat-row">
                    <span class="stat-label">SAM Agents</span>
                    <span class="stat-value" id="stat-agents">0</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Total Entities</span>
                    <span class="stat-value" id="stat-entities">0</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Interactions</span>
                    <span class="stat-value" id="stat-interactions">0</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Temperature</span>
                    <span class="stat-value" id="stat-temperature">20°C</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">FPS</span>
                    <span class="stat-value" id="stat-fps">0</span>
                </div>
            </div>
            
            <div class="ui-element agent-inspector" id="agent-inspector">
                <div class="agent-header">
                    <div class="agent-avatar" id="agent-avatar">SA</div>
                    <div class="agent-info">
                        <div class="agent-name" id="agent-name">SAM Agent</div>
                        <div class="agent-id" id="agent-id">ID: sam_0</div>
                    </div>
                </div>
                
                <div class="needs-container">
                    <div class="needs-title">Needs & Status</div>
                    <div class="need-bar">
                        <div class="need-label">
                            <span>Health</span>
                            <span id="health-value">100%</span>
                        </div>
                        <div class="need-progress">
                            <div class="need-fill health" id="health-bar" style="width: 100%"></div>
                        </div>
                    </div>
                    <div class="need-bar">
                        <div class="need-label">
                            <span>Energy</span>
                            <span id="energy-value">100%</span>
                        </div>
                        <div class="need-progress">
                            <div class="need-fill energy" id="energy-bar" style="width: 100%"></div>
                        </div>
                    </div>
                    <div class="need-bar">
                        <div class="need-label">
                            <span>Hunger</span>
                            <span id="hunger-value">0%</span>
                        </div>
                        <div class="need-progress">
                            <div class="need-fill hunger" id="hunger-bar" style="width: 0%"></div>
                        </div>
                    </div>
                    <div class="need-bar">
                        <div class="need-label">
                            <span>Happiness</span>
                            <span id="happiness-value">50%</span>
                        </div>
                        <div class="need-progress">
                            <div class="need-fill happiness" id="happiness-bar" style="width: 50%"></div>
                        </div>
                    </div>
                </div>
                
                <div class="agent-details">
                    <div class="stat-row">
                        <span class="stat-label">Current Action</span>
                        <span class="stat-value" id="agent-action">Idle</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Relationships</span>
                        <span class="stat-value" id="agent-relationships">0</span>
                    </div>
                </div>
            </div>
            
            <div class="ui-element controls-panel">
                <button class="control-btn" id="pause-btn">⏸ Pause</button>
                <button class="control-btn" id="spawn-btn">➕ Spawn Agent</button>
                <button class="control-btn" id="camera-btn">📷 Free Camera</button>
                <button class="control-btn" id="settings-btn">⚙️ Settings</button>
            </div>
            
            <div class="ui-element minimap">
                <canvas id="minimap-canvas"></canvas>
            </div>
            
            <div class="ui-element chat-panel">
                <div class="stats-title">Agent Communications</div>
                <div class="chat-messages" id="chat-messages">
                    <!-- Messages will be added here -->
                </div>
            </div>
        </div>
        
        <div class="tooltip" id="tooltip"></div>
    </div>
    
    <script type="module">
        import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.150.0/build/three.module.js';
        import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.150.0/examples/jsm/controls/OrbitControls.js';
        import { GLTFLoader } from 'https://cdn.jsdelivr.net/npm/three@0.150.0/examples/jsm/loaders/GLTFLoader.js';
        import { EffectComposer } from 'https://cdn.jsdelivr.net/npm/three@0.150.0/examples/jsm/postprocessing/EffectComposer.js';
        import { RenderPass } from 'https://cdn.jsdelivr.net/npm/three@0.150.0/examples/jsm/postprocessing/RenderPass.js';
        import { UnrealBloomPass } from 'https://cdn.jsdelivr.net/npm/three@0.150.0/examples/jsm/postprocessing/UnrealBloomPass.js';
        
        // SYNTHIX Universe 3D Renderer
        class UniverseRenderer {
            constructor() {
                this.container = document.getElementById('universe-container');
                this.canvas = document.getElementById('render-canvas');
                
                // Three.js setup
                this.scene = new THREE.Scene();
                this.camera = new THREE.PerspectiveCamera(
                    75,
                    window.innerWidth / window.innerHeight,
                    0.1,
                    1000
                );
                
                this.renderer = new THREE.WebGLRenderer({
                    canvas: this.canvas,
                    antialias: true,
                    alpha: true
                });
                
                this.renderer.setSize(window.innerWidth, window.innerHeight);
                this.renderer.setPixelRatio(window.devicePixelRatio);
                this.renderer.shadowMap.enabled = true;
                this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
                this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
                this.renderer.toneMappingExposure = 1;
                
                // Post-processing
                this.composer = new EffectComposer(this.renderer);
                this.renderPass = new RenderPass(this.scene, this.camera);
                this.composer.addPass(this.renderPass);
                
                this.bloomPass = new UnrealBloomPass(
                    new THREE.Vector2(window.innerWidth, window.innerHeight),
                    1.5, 0.4, 0.85
                );
                this.composer.addPass(this.bloomPass);
                
                // Camera controls
                this.controls = new OrbitControls(this.camera, this.canvas);
                this.controls.enableDamping = true;
                this.controls.dampingFactor = 0.05;
                this.controls.maxDistance = 200;
                this.controls.minDistance = 5;
                
                // Initial camera position
                this.camera.position.set(50, 30, 50);
                this.camera.lookAt(0, 0, 0);
                
                // Entities
                this.entities = new Map();
                this.selectedEntity = null;
                
                // Animation
                this.clock = new THREE.Clock();
                this.mixer = null;
                this.animations = new Map();
                
                // WebSocket connection
                this.ws = null;
                this.wsUrl = 'ws://localhost:8765';
                
                // UI elements
                this.uiElements = {
                    stats: {
                        agents: document.getElementById('stat-agents'),
                        entities: document.getElementById('stat-entities'),
                        interactions: document.getElementById('stat-interactions'),
                        temperature: document.getElementById('stat-temperature'),
                        fps: document.getElementById('stat-fps')
                    },
                    agent: {
                        inspector: document.getElementById('agent-inspector'),
                        name: document.getElementById('agent-name'),
                        id: document.getElementById('agent-id'),
                        avatar: document.getElementById('agent-avatar'),
                        action: document.getElementById('agent-action'),
                        relationships: document.getElementById('agent-relationships'),
                        health: {
                            value: document.getElementById('health-value'),
                            bar: document.getElementById('health-bar')
                        },
                        energy: {
                            value: document.getElementById('energy-value'),
                            bar: document.getElementById('energy-bar')
                        },
                        hunger: {
                            value: document.getElementById('hunger-value'),
                            bar: document.getElementById('hunger-bar')
                        },
                        happiness: {
                            value: document.getElementById('happiness-value'),
                            bar: document.getElementById('happiness-bar')
                        }
                    },
                    chat: document.getElementById('chat-messages'),
                    time: document.getElementById('time-text'),
                    weather: {
                        text: document.getElementById('weather-text'),
                        icon: document.getElementById('weather-icon')
                    }
                };
                
                // State
                this.paused = false;
                this.freeCamera = false;
                this.stats = {
                    fps: 0,
                    frameTime: 0,
                    lastTime: 0
                };
                
                this.init();
            }
            
            async init() {
                // Setup scene
                this.setupEnvironment();
                this.setupLighting();
                this.setupTerrain();
                this.loadAssets();
                
                // Connect WebSocket
                this.connectWebSocket();
                
                // Setup events
                this.setupEvents();
                
                // Start render loop
                this.animate();
                
                // Hide loading screen
                setTimeout(() => {
                    const loadingScreen = document.getElementById('loading-screen');
                    loadingScreen.style.opacity = '0';
                    setTimeout(() => loadingScreen.style.display = 'none', 1000);
                }, 2000);
            }
            
            setupEnvironment() {
                // Fog
                this.scene.fog = new THREE.Fog(0x87CEEB, 50, 300);
                
                // Skybox
                const skyGeometry = new THREE.SphereGeometry(500, 32, 32);
                const skyMaterial = new THREE.ShaderMaterial({
                    uniforms: {
                        topColor: { value: new THREE.Color(0x0077be) },
                        bottomColor: { value: new THREE.Color(0xffffff) },
                        offset: { value: 33 },
                        exponent: { value: 0.6 },
                        time: { value: 0 }
                    },
                    vertexShader: `
                        varying vec3 vWorldPosition;
                        void main() {
                            vec4 worldPosition = modelMatrix * vec4(position, 1.0);
                            vWorldPosition = worldPosition.xyz;
                            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                        }
                    `,
                    fragmentShader: `
                        uniform vec3 topColor;
                        uniform vec3 bottomColor;
                        uniform float offset;
                        uniform float exponent;
                        uniform float time;
                        varying vec3 vWorldPosition;
                        
                        void main() {
                            float h = normalize(vWorldPosition + offset).y;
                            float t = max(pow(max(h, 0.0), exponent), 0.0);
                            
                            // Day/night cycle
                            float dayNight = sin(time * 0.1) * 0.5 + 0.5;
                            vec3 skyColor = mix(bottomColor, topColor, t);
                            skyColor = mix(skyColor * 0.2, skyColor, dayNight);
                            
                            gl_FragColor = vec4(skyColor, 1.0);
                        }
                    `,
                    side: THREE.BackSide
                });
                
                this.sky = new THREE.Mesh(skyGeometry, skyMaterial);
                this.scene.add(this.sky);
            }
            
            setupLighting() {
                // Ambient light
                this.ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
                this.scene.add(this.ambientLight);
                
                // Directional light (sun)
                this.sunLight = new THREE.DirectionalLight(0xffffff, 1);
                this.sunLight.position.set(100, 100, 50);
                this.sunLight.castShadow = true;
                this.sunLight.shadow.mapSize.width = 2048;
                this.sunLight.shadow.mapSize.height = 2048;
                this.sunLight.shadow.camera.near = 0.5;
                this.sunLight.shadow.camera.far = 500;
                this.sunLight.shadow.camera.left = -100;
                this.sunLight.shadow.camera.right = 100;
                this.sunLight.shadow.camera.top = 100;
                this.sunLight.shadow.camera.bottom = -100;
                this.scene.add(this.sunLight);
                
                // Hemisphere light
                this.hemiLight = new THREE.HemisphereLight(0x87CEEB, 0x545454, 0.6);
                this.scene.add(this.hemiLight);
            }
            
            setupTerrain() {
                // Create terrain with noise
                const terrainSize = 200;
                const segments = 100;
                const geometry = new THREE.PlaneGeometry(terrainSize, terrainSize, segments, segments);
                
                // Add height variation
                const vertices = geometry.attributes.position.array;
                for (let i = 0; i < vertices.length; i += 3) {
                    const x = vertices[i];
                    const y = vertices[i + 1];
                    
                    // Simple noise function
                    const height = Math.sin(x * 0.05) * Math.cos(y * 0.05) * 2 +
                                  Math.sin(x * 0.1) * Math.cos(y * 0.1) * 1;
                    
                    vertices[i + 2] = height;
                }
                
                geometry.computeVertexNormals();
                
                // Terrain material
                const material = new THREE.MeshStandardMaterial({
                    color: 0x3a8c3a,
                    roughness: 0.8,
                    metalness: 0.2
                });
                
                this.terrain = new THREE.Mesh(geometry, material);
                this.terrain.rotation.x = -Math.PI / 2;
                this.terrain.receiveShadow = true;
                this.scene.add(this.terrain);
                
                // Add trees
                this.addTrees();
                
                // Add buildings
                this.addBuildings();
            }
            
            addTrees() {
                const treeGroup = new THREE.Group();
                
                for (let i = 0; i < 50; i++) {
                    const tree = new THREE.Group();
                    
                    // Trunk
                    const trunkGeometry = new THREE.CylinderGeometry(0.5, 0.7, 4);
                    const trunkMaterial = new THREE.MeshStandardMaterial({ color: 0x8B4513 });
                    const trunk = new THREE.Mesh(trunkGeometry, trunkMaterial);
                    trunk.position.y = 2;
                    trunk.castShadow = true;
                    tree.add(trunk);
                    
                    // Leaves
                    const leavesGeometry = new THREE.ConeGeometry(3, 6, 8);
                    const leavesMaterial = new THREE.MeshStandardMaterial({ color: 0x228B22 });
                    const leaves = new THREE.Mesh(leavesGeometry, leavesMaterial);
                    leaves.position.y = 6;
                    leaves.castShadow = true;
                    tree.add(leaves);
                    
                    // Position
                    tree.position.x = (Math.random() - 0.5) * 180;
                    tree.position.z = (Math.random() - 0.5) * 180;
                    tree.position.y = 0;
                    
                    // Random scale
                    const scale = 0.8 + Math.random() * 0.4;
                    tree.scale.set(scale, scale, scale);
                    
                    treeGroup.add(tree);
                }
                
                this.scene.add(treeGroup);
            }
            
            addBuildings() {
                const buildingGroup = new THREE.Group();
                
                // Simple houses
                for (let i = 0; i < 10; i++) {
                    const house = new THREE.Group();
                    
                    // Base
                    const baseGeometry = new THREE.BoxGeometry(8, 6, 8);
                    const baseMaterial = new THREE.MeshStandardMaterial({ color: 0xD2691E });
                    const base = new THREE.Mesh(baseGeometry, baseMaterial);
                    base.position.y = 3;
                    base.castShadow = true;
                    base.receiveShadow = true;
                    house.add(base);
                    
                    // Roof
                    const roofGeometry = new THREE.ConeGeometry(6, 4, 4);
                    const roofMaterial = new THREE.MeshStandardMaterial({ color: 0x8B0000 });
                    const roof = new THREE.Mesh(roofGeometry, roofMaterial);
                    roof.position.y = 8;
                    roof.rotation.y = Math.PI / 4;
                    roof.castShadow = true;
                    house.add(roof);
                    
                    // Position
                    house.position.x = (Math.random() - 0.5) * 150;
                    house.position.z = (Math.random() - 0.5) * 150;
                    
                    buildingGroup.add(house);
                }
                
                this.scene.add(buildingGroup);
            }
            
            async loadAssets() {
                // Create simple agent model
                this.agentModel = this.createAgentModel();
                
                // Load textures
                const textureLoader = new THREE.TextureLoader();
                
                // Food model
                this.foodModel = new THREE.Mesh(
                    new THREE.SphereGeometry(0.3, 16, 16),
                    new THREE.MeshStandardMaterial({
                        color: 0xff6b35,
                        emissive: 0xff6b35,
                        emissiveIntensity: 0.2
                    })
                );
            }
            
            createAgentModel() {
                const group = new THREE.Group();
                
                // Body
                const bodyGeometry = new THREE.CapsuleGeometry(0.4, 1.2, 8, 16);
                const bodyMaterial = new THREE.MeshStandardMaterial({ color: 0x4444ff });
                const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
                body.position.y = 1;
                body.castShadow = true;
                group.add(body);
                
                // Head
                const headGeometry = new THREE.SphereGeometry(0.4, 16, 16);
                const headMaterial = new THREE.MeshStandardMaterial({ color: 0xffdbac });
                const head = new THREE.Mesh(headGeometry, headMaterial);
                head.position.y = 2.2;
                head.castShadow = true;
                group.add(head);
                
                // Eyes
                const eyeGeometry = new THREE.SphereGeometry(0.05, 8, 8);
                const eyeMaterial = new THREE.MeshStandardMaterial({ color: 0x000000 });
                
                const leftEye = new THREE.Mesh(eyeGeometry, eyeMaterial);
                leftEye.position.set(-0.15, 2.2, 0.35);
                group.add(leftEye);
                
                const rightEye = new THREE.Mesh(eyeGeometry, eyeMaterial);
                rightEye.position.set(0.15, 2.2, 0.35);
                group.add(rightEye);
                
                // Name label
                const canvas = document.createElement('canvas');
                canvas.width = 256;
                canvas.height = 64;
                const context = canvas.getContext('2d');
                context.fillStyle = 'rgba(0, 0, 0, 0.5)';
                context.fillRect(0, 0, 256, 64);
                context.fillStyle = 'white';
                context.font = '32px Arial';
                context.textAlign = 'center';
                context.fillText('SAM Agent', 128, 40);
                
                const texture = new THREE.CanvasTexture(canvas);
                const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
                const sprite = new THREE.Sprite(spriteMaterial);
                sprite.position.y = 3;
                sprite.scale.set(2, 0.5, 1);
                group.add(sprite);
                
                return group;
            }
            
            connectWebSocket() {
                this.ws = new WebSocket(this.wsUrl);
                
                this.ws.onopen = () => {
                    console.log('Connected to SYNTHIX Universe');
                };
                
                this.ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    this.handleServerMessage(data);
                };
                
                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                };
                
                this.ws.onclose = () => {
                    console.log('Disconnected from universe');
                    // Attempt reconnect
                    setTimeout(() => this.connectWebSocket(), 5000);
                };
            }
            
            handleServerMessage(data) {
                switch (data.type) {
                    case 'init':
                    case 'update':
                        this.updateUniverse(data.data);
                        break;
                    case 'agent_spawned':
                        this.addChatMessage('System', `New agent spawned: ${data.data.id}`);
                        break;
                }
            }
            
            updateUniverse(state) {
                // Update stats
                this.uiElements.stats.agents.textContent = state.stats.sam_agents;
                this.uiElements.stats.entities.textContent = state.stats.total_entities;
                this.uiElements.stats.interactions.textContent = state.stats.interactions;
                this.uiElements.stats.temperature.textContent = `${Math.round(state.temperature)}°C`;
                
                // Update time
                const hours = Math.floor(state.time_of_day * 24);
                const minutes = Math.floor((state.time_of_day * 24 - hours) * 60);
                const ampm = hours >= 12 ? 'PM' : 'AM';
                const displayHours = hours % 12 || 12;
                this.uiElements.time.textContent = `${displayHours}:${minutes.toString().padStart(2, '0')} ${ampm}`;
                
                // Update weather
                this.uiElements.weather.text.textContent = state.weather;
                this.updateWeatherIcon(state.weather);
                
                // Update sky
                if (this.sky) {
                    this.sky.material.uniforms.time.value = state.time_of_day * Math.PI * 2;
                }
                
                // Update sun position
                const sunAngle = state.time_of_day * Math.PI * 2 - Math.PI / 2;
                this.sunLight.position.x = Math.cos(sunAngle) * 100;
                this.sunLight.position.y = Math.sin(sunAngle) * 100 + 50;
                this.sunLight.intensity = Math.max(0, Math.sin(sunAngle)) * 1.5;
                
                // Update entities
                const existingIds = new Set(this.entities.keys());
                
                for (const entityData of state.entities) {
                    existingIds.delete(entityData.id);
                    
                    if (this.entities.has(entityData.id)) {
                        this.updateEntity(entityData);
                    } else {
                        this.createEntity(entityData);
                    }
                }
                
                // Remove entities that no longer exist
                for (const id of existingIds) {
                    this.removeEntity(id);
                }
            }
            
            createEntity(data) {
                let mesh;
                
                if (data.type === 'sam_agent') {
                    mesh = this.agentModel.clone();
                    
                    // Customize appearance
                    const body = mesh.children[0];
                    const head = mesh.children[1];
                    
                    // Random colors
                    body.material = body.material.clone();
                    body.material.color = new THREE.Color(Math.random(), Math.random(), Math.random());
                    
                    // Update name
                    const sprite = mesh.children[mesh.children.length - 1];
                    const canvas = document.createElement('canvas');
                    canvas.width = 256;
                    canvas.height = 64;
                    const context = canvas.getContext('2d');
                    context.fillStyle = 'rgba(0, 0, 0, 0.5)';
                    context.fillRect(0, 0, 256, 64);
                    context.fillStyle = 'white';
                    context.font = '32px Arial';
                    context.textAlign = 'center';
                    context.fillText(data.name, 128, 40);
                    sprite.material.map = new THREE.CanvasTexture(canvas);
                    
                } else if (data.type === 'food') {
                    mesh = this.foodModel.clone();
                } else {
                    // Generic object
                    mesh = new THREE.Mesh(
                        new THREE.BoxGeometry(1, 1, 1),
                        new THREE.MeshStandardMaterial({ color: 0x888888 })
                    );
                }
                
                // Set position
                mesh.position.set(data.position.x, data.position.y, data.position.z);
                
                // Add to scene
                this.scene.add(mesh);
                this.entities.set(data.id, {
                    mesh: mesh,
                    data: data,
                    mixer: null
                });
            }
            
            updateEntity(data) {
                const entity = this.entities.get(data.id);
                if (!entity) return;
                
                // Update position
                entity.mesh.position.set(data.position.x, data.position.y, data.position.z);
                
                // Update rotation
                entity.mesh.rotation.y = data.rotation.y;
                
                // Update data
                entity.data = data;
                
                // Update selected entity UI if this is selected
                if (this.selectedEntity === data.id) {
                    this.updateAgentInspector(data);
                }
            }
            
            removeEntity(id) {
                const entity = this.entities.get(id);
                if (!entity) return;
                
                this.scene.remove(entity.mesh);
                this.entities.delete(id);
                
                if (this.selectedEntity === id) {
                    this.selectedEntity = null;
                    this.uiElements.agent.inspector.style.display = 'none';
                }
            }
            
            updateWeatherIcon(weather) {
                const canvas = this.uiElements.weather.icon;
                const ctx = canvas.getContext('2d');
                canvas.width = 40;
                canvas.height = 40;
                
                ctx.clearRect(0, 0, 40, 40);
                
                switch (weather) {
                    case 'clear':
                        // Sun
                        ctx.fillStyle = '#FFD700';
                        ctx.beginPath();
                        ctx.arc(20, 20, 10, 0, Math.PI * 2);
                        ctx.fill();
                        
                        // Rays
                        for (let i = 0; i < 8; i++) {
                            const angle = (i / 8) * Math.PI * 2;
                            ctx.beginPath();
                            ctx.moveTo(20 + Math.cos(angle) * 13, 20 + Math.sin(angle) * 13);
                            ctx.lineTo(20 + Math.cos(angle) * 18, 20 + Math.sin(angle) * 18);
                            ctx.strokeStyle = '#FFD700';
                            ctx.lineWidth = 2;
                            ctx.stroke();
                        }
                        break;
                        
                    case 'cloudy':
                        // Cloud
                        ctx.fillStyle = '#CCCCCC';
                        ctx.beginPath();
                        ctx.arc(15, 22, 6, 0, Math.PI * 2);
                        ctx.arc(25, 22, 6, 0, Math.PI * 2);
                        ctx.arc(20, 18, 7, 0, Math.PI * 2);
                        ctx.fill();
                        break;
                        
                    case 'rain':
                        // Cloud
                        ctx.fillStyle = '#888888';
                        ctx.beginPath();
                        ctx.arc(15, 15, 5, 0, Math.PI * 2);
                        ctx.arc(25, 15, 5, 0, Math.PI * 2);
                        ctx.arc(20, 12, 6, 0, Math.PI * 2);
                        ctx.fill();
                        
                        // Rain drops
                        ctx.strokeStyle = '#4169E1';
                        ctx.lineWidth = 2;
                        for (let i = 0; i < 5; i++) {
                            ctx.beginPath();
                            ctx.moveTo(10 + i * 5, 22);
                            ctx.lineTo(8 + i * 5, 28);
                            ctx.stroke();
                        }
                        break;
                }
            }
            
            updateAgentInspector(data) {
                this.uiElements.agent.name.textContent = data.name;
                this.uiElements.agent.id.textContent = `ID: ${data.id}`;
                this.uiElements.agent.avatar.textContent = data.name.substring(0, 2).toUpperCase();
                
                // Update needs
                const health = Math.round(data.health || 100);
                this.uiElements.agent.health.value.textContent = `${health}%`;
                this.uiElements.agent.health.bar.style.width = `${health}%`;
                
                const energy = Math.round(data.energy || 100);
                this.uiElements.agent.energy.value.textContent = `${energy}%`;
                this.uiElements.agent.energy.bar.style.width = `${energy}%`;
                
                const hunger = Math.round(data.hunger || 0);
                this.uiElements.agent.hunger.value.textContent = `${hunger}%`;
                this.uiElements.agent.hunger.bar.style.width = `${hunger}%`;
                
                const happiness = Math.round(data.happiness || 50);
                this.uiElements.agent.happiness.value.textContent = `${happiness}%`;
                this.uiElements.agent.happiness.bar.style.width = `${happiness}%`;
                
                // Update other info
                this.uiElements.agent.action.textContent = data.animation_state || 'idle';
                this.uiElements.agent.relationships.textContent = data.relationships || 0;
            }
            
            addChatMessage(sender, message) {
                const messageEl = document.createElement('div');
                messageEl.className = 'chat-message';
                
                const senderEl = document.createElement('div');
                senderEl.className = 'chat-sender';
                senderEl.textContent = sender;
                
                const textEl = document.createElement('div');
                textEl.className = 'chat-text';
                textEl.textContent = message;
                
                messageEl.appendChild(senderEl);
                messageEl.appendChild(textEl);
                
                this.uiElements.chat.appendChild(messageEl);
                
                // Auto-scroll
                this.uiElements.chat.scrollTop = this.uiElements.chat.scrollHeight;
                
                // Remove old messages
                while (this.uiElements.chat.children.length > 50) {
                    this.uiElements.chat.removeChild(this.uiElements.chat.firstChild);
                }
            }
            
            setupEvents() {
                // Window resize
                window.addEventListener('resize', () => this.onWindowResize());
                
                // Mouse events
                this.canvas.addEventListener('click', (e) => this.onMouseClick(e));
                this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
                
                // Control buttons
                document.getElementById('pause-btn').addEventListener('click', () => {
                    this.paused = !this.paused;
                    document.getElementById('pause-btn').textContent = this.paused ? '▶ Play' : '⏸ Pause';
                });
                
                document.getElementById('spawn-btn').addEventListener('click', () => {
                    const pos = {
                        x: (Math.random() - 0.5) * 100,
                        y: 0,
                        z: (Math.random() - 0.5) * 100
                    };
                    
                    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                        this.ws.send(JSON.stringify({
                            type: 'spawn_agent',
                            name: `Agent_${Date.now()}`,
                            ...pos
                        }));
                    }
                });
                
                document.getElementById('camera-btn').addEventListener('click', () => {
                    this.freeCamera = !this.freeCamera;
                    this.controls.enabled = this.freeCamera;
                    document.getElementById('camera-btn').classList.toggle('active', this.freeCamera);
                });
            }
            
            onWindowResize() {
                this.camera.aspect = window.innerWidth / window.innerHeight;
                this.camera.updateProjectionMatrix();
                this.renderer.setSize(window.innerWidth, window.innerHeight);
                this.composer.setSize(window.innerWidth, window.innerHeight);
            }
            
            onMouseClick(event) {
                const mouse = new THREE.Vector2();
                mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
                mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
                
                const raycaster = new THREE.Raycaster();
                raycaster.setFromCamera(mouse, this.camera);
                
                // Check for entity clicks
                const meshes = Array.from(this.entities.values()).map(e => e.mesh);
                const intersects = raycaster.intersectObjects(meshes, true);
                
                if (intersects.length > 0) {
                    // Find which entity was clicked
                    for (const [id, entity] of this.entities) {
                        if (entity.mesh === intersects[0].object.parent || 
                            entity.mesh === intersects[0].object) {
                            this.selectEntity(id);
                            break;
                        }
                    }
                } else {
                    // Deselect
                    this.selectedEntity = null;
                    this.uiElements.agent.inspector.style.display = 'none';
                }
            }
            
            onMouseMove(event) {
                // Update tooltip
                const mouse = new THREE.Vector2();
                mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
                mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
                
                const raycaster = new THREE.Raycaster();
                raycaster.setFromCamera(mouse, this.camera);
                
                const meshes = Array.from(this.entities.values()).map(e => e.mesh);
                const intersects = raycaster.intersectObjects(meshes, true);
                
                const tooltip = document.getElementById('tooltip');
                
                if (intersects.length > 0) {
                    for (const [id, entity] of this.entities) {
                        if (entity.mesh === intersects[0].object.parent || 
                            entity.mesh === intersects[0].object) {
                            tooltip.textContent = entity.data.name;
                            tooltip.style.left = event.clientX + 10 + 'px';
                            tooltip.style.top = event.clientY - 30 + 'px';
                            tooltip.classList.add('visible');
                            break;
                        }
                    }
                } else {
                    tooltip.classList.remove('visible');
                }
            }
            
            selectEntity(id) {
                const entity = this.entities.get(id);
                if (!entity || entity.data.type !== 'sam_agent') return;
                
                this.selectedEntity = id;
                this.uiElements.agent.inspector.style.display = 'block';
                this.updateAgentInspector(entity.data);
                
                // Focus camera on entity
                if (!this.freeCamera) {
                    const targetPos = entity.mesh.position.clone();
                    targetPos.y += 10;
                    
                    this.camera.position.lerp(
                        new THREE.Vector3(
                            targetPos.x + 20,
                            targetPos.y + 15,
                            targetPos.z + 20
                        ),
                        0.1
                    );
                    
                    this.controls.target.lerp(targetPos, 0.1);
                }
            }
            
            animate() {
                requestAnimationFrame(() => this.animate());
                
                // Calculate FPS
                const currentTime = performance.now();
                if (currentTime >= this.stats.lastTime + 1000) {
                    this.uiElements.stats.fps.textContent = Math.round(this.stats.fps);
                    this.stats.fps = 0;
                    this.stats.lastTime = currentTime;
                }
                this.stats.fps++;
                
                const deltaTime = this.clock.getDelta();
                
                // Update controls
                this.controls.update();
                
                // Update animations
                for (const entity of this.entities.values()) {
                    if (entity.mixer) {
                        entity.mixer.update(deltaTime);
                    }
                    
                    // Simple animation for agents
                    if (entity.data && entity.data.type === 'sam_agent') {
                        // Bobbing animation
                        const time = performance.now() * 0.001;
                        entity.mesh.position.y = entity.data.position.y + Math.sin(time * 2) * 0.1;
                        
                        // Rotation towards movement
                        if (entity.data.animation_state === 'walking' || entity.data.animation_state === 'running') {
                            entity.mesh.rotation.y = entity.data.rotation.y;
                        }
                    }
                }
                
                // Render scene
                this.composer.render();
                
                // Update minimap
                this.updateMinimap();
            }
            
            updateMinimap() {
                const canvas = document.getElementById('minimap-canvas');
                const ctx = canvas.getContext('2d');
                
                canvas.width = 200;
                canvas.height = 200;
                
                // Clear
                ctx.fillStyle = '#111';
                ctx.fillRect(0, 0, 200, 200);
                
                // Draw entities
                for (const entity of this.entities.values()) {
                    if (!entity.data) continue;
                    
                    // Map position to minimap
                    const x = (entity.data.position.x + 100) / 200 * 200;
                    const z = (entity.data.position.z + 100) / 200 * 200;
                    
                    ctx.fillStyle = entity.data.type === 'sam_agent' ? '#00ff88' : '#ff6b35';
                    ctx.beginPath();
                    ctx.arc(x, z, entity.data.type === 'sam_agent' ? 4 : 2, 0, Math.PI * 2);
                    ctx.fill();
                }
                
                // Draw camera position
                const camX = (this.camera.position.x + 100) / 200 * 200;
                const camZ = (this.camera.position.z + 100) / 200 * 200;
                
                ctx.strokeStyle = '#4ecdc4';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(camX - 5, camZ);
                ctx.lineTo(camX + 5, camZ);
                ctx.moveTo(camX, camZ - 5);
                ctx.lineTo(camX, camZ + 5);
                ctx.stroke();
            }
        }
        
        // Initialize and start the universe renderer
        const universe = new UniverseRenderer();
    </script>
</body>
</html>
EOFHTML

    log_success "Advanced 3D visualization created"
}

# Create systemd services
create_services() {
    log_step "Creating systemd services..."
    
    # SAM Core Service
    cat > "${CHROOT_ENV}/etc/systemd/system/sam-core.service" << 'EOF'
[Unit]
Description=SAM Core v2.0 - Advanced AI Agent System
After=network.target

[Service]
Type=simple
User=sam
Group=sam
WorkingDirectory=/usr/lib/sam/core
ExecStart=/usr/bin/python3 /usr/lib/sam/core/sam_core_v2.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    # SYNTHIX Universe Service
    cat > "${CHROOT_ENV}/etc/systemd/system/synthix-universe.service" << 'EOF'
[Unit]
Description=SYNTHIX Universe Engine v2.0
After=network.target sam-core.service

[Service]
Type=simple
User=synthix
Group=synthix
WorkingDirectory=/usr/lib/synthix/universe
ExecStart=/usr/bin/python3 /usr/lib/synthix/universe/universe_engine_v2.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    # Web Interface Service
    cat > "${CHROOT_ENV}/etc/systemd/system/synthix-web.service" << 'EOF'
[Unit]
Description=SYNTHIX Web Interface
After=network.target synthix-universe.service

[Service]
Type=simple
User=synthix
Group=synthix
WorkingDirectory=/usr/share/synthix/web
ExecStart=/usr/bin/python3 -m http.server 8080
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    # Create service users
    mount_chroot
    chroot_exec "useradd -r -s /bin/false sam || true"
    chroot_exec "useradd -r -s /bin/false synthix || true"
    
    # Set permissions
    chroot_exec "chown -R sam:sam /usr/lib/sam /var/lib/sam"
    chroot_exec "chown -R synthix:synthix /usr/lib/synthix /usr/share/synthix /var/lib/synthix"
    
    # Enable services
    chroot_exec "systemctl enable sam-core.service"
    chroot_exec "systemctl enable synthix-universe.service"
    chroot_exec "systemctl enable synthix-web.service"
    
    unmount_chroot
    log_success "Services created and enabled"
}

# Configure auto-start
configure_autostart() {
    log_step "Configuring auto-start..."
    
    # Create X11 session
    cat > "${CHROOT_ENV}/usr/share/xsessions/synthix.desktop" << 'EOF'
[Desktop Entry]
Name=SYNTHIX-SAM Universe
Comment=Living AI World
Exec=/usr/bin/synthix-session
Type=Application
EOF

    # Create session script
    cat > "${CHROOT_ENV}/usr/bin/synthix-session" << 'EOF'
#!/bin/bash
# Start SYNTHIX session

# Configure display
export DISPLAY=:0
export XDG_RUNTIME_DIR="/run/user/$(id -u)"

# Start window manager
openbox &

# Wait for services
sleep 5

# Launch browser in fullscreen
chromium-browser \
    --kiosk \
    --no-sandbox \
    --disable-web-security \
    --disable-features=TranslateUI \
    --disable-features=IsolateOrigins,site-per-process \
    --autoplay-policy=no-user-gesture-required \
    --start-maximized \
    --window-position=0,0 \
    --window-size=1920,1080 \
    "http://localhost:8080" &

# Keep session alive
wait
EOF

    chmod +x "${CHROOT_ENV}/usr/bin/synthix-session"
    
    # Configure auto-login
    cat > "${CHROOT_ENV}/etc/systemd/system/getty@tty1.service.d/override.conf" << 'EOF'
[Service]
ExecStart=
ExecStart=-/sbin/agetty --autologin synthix --noclear %I $TERM
EOF

    # Configure automatic X start
    cat >> "${CHROOT_ENV}/home/synthix/.bash_profile" << 'EOF'
# Auto-start X if on tty1
if [[ -z $DISPLAY ]] && [[ $(tty) = /dev/tty1 ]]; then
    exec startx
fi
EOF

    log_success "Auto-start configured"
}

# Build ISO
build_iso() {
    log_step "Building ISO image..."
    
    # Create squashfs
    log_info "Creating squashfs filesystem..."
    sudo mksquashfs "${CHROOT_ENV}" "${WORK_DIR}/iso/casper/filesystem.squashfs" \
        -comp xz -noappend
    
    # Create filesystem manifest
    mount_chroot
    chroot_exec "dpkg-query -W --showformat='${Package} ${Version}\n'" > \
        "${WORK_DIR}/iso/casper/filesystem.manifest"
    unmount_chroot
    
    # Create disk info
    echo "SYNTHIX-SAM OS v2.0" > "${WORK_DIR}/iso/.disk/info"
    echo "SYNTHIX-SAM OS v2.0 - Living AI Universe" > "${WORK_DIR}/iso/README.txt"
    
    # Create GRUB configuration
    cat > "${WORK_DIR}/iso/boot/grub/grub.cfg" << 'EOF'
set timeout=10
set default=0

menuentry "SYNTHIX-SAM OS v2.0 - Living AI Universe" {
    linux /casper/vmlinuz boot=casper quiet splash ---
    initrd /casper/initrd
}

menuentry "SYNTHIX-SAM OS v2.0 (Safe Mode)" {
    linux /casper/vmlinuz boot=casper xforcevesa quiet splash ---
    initrd /casper/initrd
}
EOF

    # Copy kernel and initrd
    cp "${CHROOT_ENV}/boot/vmlinuz-"* "${WORK_DIR}/iso/casper/vmlinuz"
    cp "${CHROOT_ENV}/boot/initrd.img-"* "${WORK_DIR}/iso/casper/initrd"
    
    # Create ISO
    log_info "Creating ISO image..."
    cd "${WORK_DIR}"
    
    xorriso -as mkisofs \
        -D -r -V "SYNTHIX-SAM-v2" \
        -cache-inodes -J -l \
        -b boot/grub/i386-pc/eltorito.img \
        -c boot/grub/boot.cat \
        -no-emul-boot -boot-load-size 4 -boot-info-table \
        -eltorito-alt-boot \
        -e boot/grub/efi.img \
        -no-emul-boot \
        -isohybrid-mbr /usr/lib/ISOLINUX/isohdpfx.bin \
        -o "${OUTPUT_DIR}/${ISO_NAME}" \
        "${WORK_DIR}/iso"
    
    log_success "ISO created: ${OUTPUT_DIR}/${ISO_NAME}"
    
    # Calculate checksum
    cd "${OUTPUT_DIR}"
    sha256sum "${ISO_NAME}" > "${ISO_NAME}.sha256"
    
    log_info "ISO size: $(du -h ${ISO_NAME} | cut -f1)"
    log_info "SHA256: $(cat ${ISO_NAME}.sha256)"
}

# Cleanup function
cleanup_all() {
    log_debug "Cleaning up..."
    unmount_chroot
    if [ -d "$WORK_DIR" ]; then
        sudo rm -rf "$WORK_DIR"
    fi
}

# Main build process
main() {
    log_step "SYNTHIX-SAM OS v2.0 Build Process Starting..."
    
    # Ensure running with proper permissions
    if [ "$EUID" -ne 0 ]; then
        log_error "This script must be run with sudo"
        exit 1
    fi
    
    # Build stages
    check_dependencies
    setup_dirs
    create_base_system
    install_packages
    create_advanced_sam_core
    create_universe_engine_v2
    create_3d_visualization
    create_services
    configure_autostart
    build_iso
    
    log_success "Build complete! ISO available at: ${OUTPUT_DIR}/${ISO_NAME}"
    log_info "To test the ISO:"
    log_info "  qemu-system-x86_64 -m 4096 -cdrom ${OUTPUT_DIR}/${ISO_NAME}"
    
    cleanup_all
}

# Handle script arguments
case "${1:-}" in
    --debug)
        set -x
        main
        ;;
    --clean)
        cleanup_all
        ;;
    *)
        main
        ;;
esac
