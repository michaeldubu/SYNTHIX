#!/bin/bash
# SYNTHIX OS ISO Builder
# A script to build a bootable Linux-based OS designed for AI simulation

set -e

# Define variables
WORK_DIR="$(pwd)/synthix-build"
OUTPUT_DIR="$(pwd)/synthix-output"
ISO_NAME="SYNTHIX-OS-v0.1.iso"
BASE_DISTRO="debian-minimal"
ARCH="amd64"

# Create necessary directories
# Add these lines before any file operations
mkdir -p /usr/lib/synthix
mkdir -p ${WORK_DIR}/chroot/usr/lib/synthix
mkdir -p ${WORK_DIR}/chroot/etc/synthix
mkdir -p ${WORK_DIR}/chroot/var/lib/synthix/universes
mkdir -p "${WORK_DIR}/chroot"
mkdir -p "${WORK_DIR}/iso"
mkdir -p "${OUTPUT_DIR}"

echo "=== SYNTHIX OS Builder ==="
echo "Building a metaphysical Linux distribution for AI simulation..."

# 1. Set up base system
echo "Fetching base system..."
debootstrap --arch=${ARCH} --variant=minbase bullseye "${WORK_DIR}/chroot" http://deb.debian.org/debian/

# 2. Prepare chroot environment
cat > ${WORK_DIR}/chroot/synthix-setup.sh << 'EOF'
#!/bin/bash
set -e

# Update and install necessary packages
apt-get update
apt-get install -y --no-install-recommends \
    linux-image-amd64 \
    grub-pc \
    systemd \
    dbus \
    network-manager \
    python3 \
    python3-pip \
    python3-venv \
    cmake \
    build-essential \
    git \
    psmisc \
    htop \
    nano \
    curl \
    wget \
    ca-certificates

# Set up SYNTHIX-specific components
mkdir -p /etc/synthix
mkdir -p /usr/lib/synthix
mkdir -p /var/lib/synthix/universes

# Create MetaKern kernel module directory
mkdir -p /usr/src/metakern-1.0

# Set up universe configuration
cat > /etc/synthix/universe.conf << 'EOFINNER'
# SYNTHIX Universe Configuration

[Universe-DEFAULT]
name="Genesis"
physics_constants={
  lightspeed=299792458,
  gravity=9.8,
  entropy_rate=0.0042,
  quantum_uncertainty=0.05
}
agent_perception_resolution=0.0001
simulation_time_dilation=1.0
metacognition_enabled=true
causality_enforcement=strict
emergence_threshold=0.75

[Runtime]
agent_scheduler=priority
max_agents=1024
memory_per_agent=128M
perception_threads=4
action_threads=4
universe_tick_rate=100
EOFINNER

EOF

# Create the Agent Runtime Environment
cat > /usr/lib/synthix/are.py << 'EOFINNER'
#!/usr/bin/env python3
"""
Agent Runtime Environment (ARE) for SYNTHIX OS
Handles agent lifecycle, perception, and action within simulated universes
"""

import os
import sys
import json
import time
import random
import logging
from multiprocessing import Process, Queue
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('/var/log/synthix/are.log'), logging.StreamHandler()]
)
logger = logging.getLogger('ARE')

class AgentProcess:
    """Representation of an AI agent as a system process"""
    
    def __init__(self, agent_id: str, universe_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.universe_id = universe_id
        self.config = config
        self.perception_queue = Queue()
        self.action_queue = Queue()
        self.process = None
        self.state = "INITIALIZED"
        
    def start(self):
        """Start the agent process"""
        self.process = Process(target=self._run)
        self.process.start()
        self.state = "RUNNING"
        logger.info(f"Agent {self.agent_id} started in universe {self.universe_id}")
        
    def _run(self):
        """Main agent process loop"""
        logger.info(f"Agent {self.agent_id} is booting up...")
        
        # Set up agent's cognitive architecture
        self._setup_cognition()
        
        # Main perception-action loop
        while True:
            # Receive perceptions from the universe
            if not self.perception_queue.empty():
                perception = self.perception_queue.get()
                self._process_perception(perception)
            
            # Think and decide on actions
            action = self._cognitive_cycle()
            
            # Send actions to the universe
            if action:
                self.action_queue.put(action)
            
            # Simulate agent "thinking time"
            time_dilation = float(self.config.get('simulation_time_dilation', 1.0))
            time.sleep(0.01 * time_dilation)
    
    def _setup_cognition(self):
        """Set up the agent's cognitive architecture"""
        # In a real implementation, this would load the agent's neural networks,
        # belief systems, memory structures, etc.
        logger.info(f"Setting up cognitive architecture for agent {self.agent_id}")
        
        # Placeholder for demonstration
        self.beliefs = {}
        self.desires = []
        self.intentions = []
        self.memory = []
        
    def _process_perception(self, perception):
        """Process incoming perception data"""
        logger.debug(f"Agent {self.agent_id} received perception: {perception}")
        self.memory.append(("perception", perception, time.time()))
        
        # Update beliefs based on new perceptions
        for key, value in perception.items():
            self.beliefs[key] = value
    
    def _cognitive_cycle(self):
        """Run one cognitive cycle and decide on actions"""
        # This is a simplified placeholder for the agent's thinking process
        # In a real implementation, this would involve sophisticated AI algorithms
        
        # Random decision making for demonstration
        if random.random() < 0.3:  # 30% chance to take action
            action_types = ["MOVE", "SPEAK", "INTERACT", "OBSERVE", "CREATE"]
            chosen_action = random.choice(action_types)
            
            action = {
                "type": chosen_action,
                "agent_id": self.agent_id,
                "timestamp": time.time(),
                "parameters": {
                    "intensity": random.random(),
                    "direction": random.choice(["NORTH", "SOUTH", "EAST", "WEST"]),
                    "target": f"object_{random.randint(1, 100)}"
                }
            }
            
            return action
        return None
    
    def stop(self):
        """Stop the agent process"""
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join()
        self.state = "STOPPED"
        logger.info(f"Agent {self.agent_id} stopped")

class UniverseSimulation:
    """Manages a simulated universe with multiple agents"""
    
    def __init__(self, universe_id: str, config_path: str):
        self.universe_id = universe_id
        self.agents = {}
        self.config = self._load_config(config_path)
        self.state = "INITIALIZED"
        self.tick = 0
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load universe configuration"""
        # In a real implementation, this would parse a sophisticated config format
        try:
            with open(config_path, 'r') as f:
                # For demonstration, assume it's a JSON file
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load universe config: {e}")
            # Return default config
            return {
                "name": "Default Universe",
                "physics_constants": {
                    "lightspeed": 299792458,
                    "gravity": 9.8,
                    "entropy_rate": 0.01
                },
                "agent_perception_resolution": 0.001,
                "simulation_time_dilation": 1.0,
                "metacognition_enabled": False
            }
    
    def start(self):
        """Start the universe simulation"""
        logger.info(f"Starting universe {self.universe_id}: {self.config.get('name', 'Unnamed')}")
        self.state = "RUNNING"
        
        # Main universe loop would run in a separate thread
        # For this example, we'll just demonstrate the concept
    
    def add_agent(self, agent_id: str, agent_config: Dict[str, Any]) -> str:
        """Add an agent to the universe"""
        if agent_id in self.agents:
            logger.warning(f"Agent {agent_id} already exists in universe {self.universe_id}")
            return None
            
        # Create and start agent process
        agent = AgentProcess(agent_id, self.universe_id, agent_config)
        agent.start()
        
        self.agents[agent_id] = agent
        logger.info(f"Added agent {agent_id} to universe {self.universe_id}")
        return agent_id
    
    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the universe"""
        if agent_id not in self.agents:
            logger.warning(f"Agent {agent_id} does not exist in universe {self.universe_id}")
            return False
            
        # Stop the agent process
        self.agents[agent_id].stop()
        del self.agents[agent_id]
        
        logger.info(f"Removed agent {agent_id} from universe {self.universe_id}")
        return True
    
    def stop(self):
        """Stop the universe simulation"""
        logger.info(f"Stopping universe {self.universe_id}")
        
        # Stop all agent processes
        for agent_id, agent in self.agents.items():
            agent.stop()
        
        self.state = "STOPPED"
        logger.info(f"Universe {self.universe_id} stopped")

# CLI for the Agent Runtime Environment
def main():
    """Main entry point for the ARE"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SYNTHIX Agent Runtime Environment')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Universe commands
    universe_parser = subparsers.add_parser('universe', help='Universe management')
    universe_subparsers = universe_parser.add_subparsers(dest='universe_command')
    
    create_universe_parser = universe_subparsers.add_parser('create', help='Create a new universe')
    create_universe_parser.add_argument('--id', required=True, help='Universe ID')
    create_universe_parser.add_argument('--config', required=True, help='Path to universe config file')
    
    start_universe_parser = universe_subparsers.add_parser('start', help='Start a universe')
    start_universe_parser.add_argument('--id', required=True, help='Universe ID')
    
    stop_universe_parser = universe_subparsers.add_parser('stop', help='Stop a universe')
    stop_universe_parser.add_argument('--id', required=True, help='Universe ID')
    
    # Agent commands
    agent_parser = subparsers.add_parser('agent', help='Agent management')
    agent_subparsers = agent_parser.add_subparsers(dest='agent_command')
    
    create_agent_parser = agent_subparsers.add_parser('create', help='Create a new agent')
    create_agent_parser.add_argument('--id', required=True, help='Agent ID')
    create_agent_parser.add_argument('--universe', required=True, help='Universe ID')
    create_agent_parser.add_argument('--config', required=True, help='Path to agent config file')
    
    stop_agent_parser = agent_subparsers.add_parser('stop', help='Stop an agent')
    stop_agent_parser.add_argument('--id', required=True, help='Agent ID')
    stop_agent_parser.add_argument('--universe', required=True, help='Universe ID')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == 'universe':
        if args.universe_command == 'create':
            # Implementation would create a new universe
            print(f"Creating universe {args.id} with config {args.config}")
        elif args.universe_command == 'start':
            # Implementation would start the universe
            print(f"Starting universe {args.id}")
        elif args.universe_command == 'stop':
            # Implementation would stop the universe
            print(f"Stopping universe {args.id}")
    
    elif args.command == 'agent':
        if args.agent_command == 'create':
            # Implementation would create a new agent
            print(f"Creating agent {args.id} in universe {args.universe}")
        elif args.agent_command == 'stop':
            # Implementation would stop the agent
            print(f"Stopping agent {args.id} in universe {args.universe}")

if __name__ == "__main__":
    main()
EOFINNER

# Create Universe Simulation Engine
cat > /usr/lib/synthix/use.py << 'EOFINNER'
#!/usr/bin/env python3
"""
Universe Simulation Engine (USE) for SYNTHIX OS
Manages the physics, causality, and environment for agent simulations
"""

import os
import sys
import json
import time
import random
import logging
import threading
import numpy as np
from typing import Dict, List, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('/var/log/synthix/use.log'), logging.StreamHandler()]
)
logger = logging.getLogger('USE')

class PhysicsEngine:
    """Simulates physical laws and properties within a universe"""
    
    def __init__(self, constants: Dict[str, float]):
        self.constants = constants
        self.entities = {}
        self.space = {}  # Simplified spatial representation
        
    def add_entity(self, entity_id: str, properties: Dict[str, Any]) -> bool:
        """Add an entity to the physical simulation"""
        if entity_id in self.entities:
            return False
            
        self.entities[entity_id] = properties
        
        # If the entity has a position, add it to the spatial index
        if 'position' in properties:
            pos = properties['position']
            pos_key = self._position_to_key(pos)
            
            if pos_key not in self.space:
                self.space[pos_key] = []
                
            self.space[pos_key].append(entity_id)
            
        return True
    
    def update_entity(self, entity_id: str, properties: Dict[str, Any]) -> bool:
        """Update an entity's properties"""
        if entity_id not in self.entities:
            return False
            
        # If position is changing, update spatial index
        if 'position' in properties and 'position' in self.entities[entity_id]:
            old_pos = self.entities[entity_id]['position']
            old_pos_key = self._position_to_key(old_pos)
            
            if old_pos_key in self.space and entity_id in self.space[old_pos_key]:
                self.space[old_pos_key].remove(entity_id)
                
            new_pos = properties['position']
            new_pos_key = self._position_to_key(new_pos)
            
            if new_pos_key not in self.space:
                self.space[new_pos_key] = []
                
            self.space[new_pos_key].append(entity_id)
            
        # Update properties
        self.entities[entity_id].update(properties)
        return True
    
    def remove_entity(self, entity_id: str) -> bool:
        """Remove an entity from the simulation"""
        if entity_id not in self.entities:
            return False
            
        # Remove from spatial index if it has a position
        if 'position' in self.entities[entity_id]:
            pos = self.entities[entity_id]['position']
            pos_key = self._position_to_key(pos)
            
            if pos_key in self.space and entity_id in self.space[pos_key]:
                self.space[pos_key].remove(entity_id)
                
        # Remove entity
        del self.entities[entity_id]
        return True
    
    def _position_to_key(self, position: Tuple[float, float, float]) -> str:
        """Convert a position to a spatial index key"""
        # Simplistic grid-based approach - in a real implementation,
        # this would be more sophisticated (octree, etc.)
        resolution = 1.0  # Grid cell size
        x, y, z = position
        grid_x = int(x / resolution)
        grid_y = int(y / resolution)
        grid_z = int(z / resolution)
        return f"{grid_x}:{grid_y}:{grid_z}"
    
    def get_nearby_entities(self, position: Tuple[float, float, float], radius: float) -> List[str]:
        """Get entities near a position"""
        results = []
        pos_key = self._position_to_key(position)
        
        # For simplicity, just check the current cell and immediate neighbors
        # A real implementation would use proper spatial queries
        # This is a placeholder to demonstrate the concept
        grid_x, grid_y, grid_z = map(int, pos_key.split(':'))
        
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    check_key = f"{grid_x+dx}:{grid_y+dy}:{grid_z+dz}"
                    if check_key in self.space:
                        results.extend(self.space[check_key])
        
        return results
    
    def apply_physics(self, delta_time: float) -> None:
        """Apply physical laws to all entities for a time step"""
        # For demonstration purposes, we'll implement a very simple physics
        # update that just applies velocity to position if those properties exist
        
        for entity_id, properties in self.entities.items():
            if 'position' in properties and 'velocity' in properties:
                # Get current values
                pos = properties['position']
                vel = properties['velocity']
                
                # Apply velocity to position
                new_x = pos[0] + vel[0] * delta_time
                new_y = pos[1] + vel[1] * delta_time
                new_z = pos[2] + vel[2] * delta_time
                
                # Update position
                self.update_entity(entity_id, {'position': (new_x, new_y, new_z)})
                
            # If the entity has mass and is affected by gravity
            if 'mass' in properties and properties.get('affected_by_gravity', True):
                # Get gravity constant from physics settings
                gravity = self.constants.get('gravity', 9.8)
                
                # If entity has velocity, apply gravity to y component
                if 'velocity' in properties:
                    vel = properties['velocity']
                    new_vel_y = vel[1] - gravity * delta_time
                    
                    # Update velocity
                    self.update_entity(entity_id, {'velocity': (vel[0], new_vel_y, vel[2])})

class CausalityEngine:
    """Manages cause and effect relationships in the universe"""
    
    def __init__(self, enforcement_level: str = 'strict'):
        self.enforcement_level = enforcement_level
        self.causal_graph = {}  # A simple directed graph of cause->effect
        self.event_log = []
        
    def log_event(self, event: Dict[str, Any]) -> int:
        """Log an event in the causal chain"""
        event_id = len(self.event_log)
        event['id'] = event_id
        event['timestamp'] = time.time()
        self.event_log.append(event)
        return event_id
    
    def add_causal_link(self, cause_id: int, effect_id: int, strength: float = 1.0) -> bool:
        """Add a causal link between two events"""
        if cause_id >= len(self.event_log) or effect_id >= len(self.event_log):
            return False
            
        if cause_id not in self.causal_graph:
            self.causal_graph[cause_id] = []
            
        self.causal_graph[cause_id].append({
            'effect_id': effect_id,
            'strength': strength
        })
        
        return True
    
    def get_causes(self, effect_id: int) -> List[int]:
        """Get all events that caused a given effect"""
        causes = []
        
        for cause_id, effects in self.causal_graph.items():
            for effect in effects:
                if effect['effect_id'] == effect_id:
                    causes.append(cause_id)
                    
        return causes
    
    def get_effects(self, cause_id: int) -> List[int]:
        """Get all effects of a given cause"""
        if cause_id not in self.causal_graph:
            return []
            
        return [effect['effect_id'] for effect in self.causal_graph[cause_id]]
    
    def validate_causal_chain(self, start_id: int, end_id: int) -> Tuple[bool, List[int]]:
        """Validate that there is a causal chain between start and end events"""
        # Simple BFS to find path from start to end
        if start_id not in self.causal_graph:
            return False, []
            
        visited = set()
        queue = [(start_id, [start_id])]
        
        while queue:
            node, path = queue.pop(0)
            
            if node == end_id:
                return True, path
                
            if node in visited:
                continue
                
            visited.add(node)
            
            if node in self.causal_graph:
                for effect in self.causal_graph[node]:
                    effect_id = effect['effect_id']
                    if effect_id not in visited:
                        queue.append((effect_id, path + [effect_id]))
        
        return False, []

class UniverseSimulation:
    """Main class for simulating an entire universe"""
    
    def __init__(self, universe_id: str, config: Dict[str, Any]):
        self.universe_id = universe_id
        self.config = config
        self.name = config.get('name', 'Unnamed Universe')
        
        # Extract physics constants
        physics_constants = config.get('physics_constants', {})
        
        # Set up component engines
        self.physics = PhysicsEngine(physics_constants)
        self.causality = CausalityEngine(config.get('causality_enforcement', 'strict'))
        
        # Universe state
        self.entities = {}  # All entities in the universe
        self.agents = {}    # Agent-specific data
        self.running = False
        self.tick_count = 0
        self.start_time = None
        
        # Time dilation (subjective time vs. real time)
        self.time_dilation = float(config.get('simulation_time_dilation', 1.0))
        
        # Simulation thread
        self.simulation_thread = None
        
        logger.info(f"Initialized universe '{self.name}' with ID {universe_id}")
    
    def start(self):
        """Start the universe simulation"""
        if self.running:
            logger.warning(f"Universe {self.universe_id} is already running")
            return False
            
        logger.info(f"Starting universe {self.universe_id} '{self.name}'")
        self.running = True
        self.start_time = time.time()
        
        # Start simulation in a separate thread
        self.simulation_thread = threading.Thread(target=self._simulation_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        return True
    
    def stop(self):
        """Stop the universe simulation"""
        if not self.running:
            logger.warning(f"Universe {self.universe_id} is not running")
            return False
            
        logger.info(f"Stopping universe {self.universe_id} '{self.name}'")
        self.running = False
        
        # Wait for simulation thread to end
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join(timeout=5.0)
            
        return True
    
    def _simulation_loop(self):
        """Main simulation loop"""
        logger.info(f"Universe {self.universe_id} simulation loop started")
        
        # Get tick rate from config
        tick_rate = float(self.config.get('universe_tick_rate', 100))
        tick_interval = 1.0 / tick_rate
        
        while self.running:
            loop_start = time.time()
            
            # Process one simulation tick
            self._process_tick()
            
            # Calculate time to sleep to maintain tick rate
            elapsed = time.time() - loop_start
            sleep_time = max(0, tick_interval - elapsed)
            
            # Sleep if needed
            if sleep_time > 0:
                time.sleep(sleep_time)
                
        logger.info(f"Universe {self.universe_id} simulation loop ended")
    
    def _process_tick(self):
        """Process one simulation tick"""
        self.tick_count += 1
        
        # Apply physics
        self.physics.apply_physics(1.0 / float(self.config.get('universe_tick_rate', 100)))
        
        # Process agent actions - not implemented in this simplified version
        
        # Log a tick event every 100 ticks
        if self.tick_count % 100 == 0:
            logger.debug(f"Universe {self.universe_id} tick {self.tick_count}")
    
    def add_entity(self, entity_id: str, properties: Dict[str, Any]) -> bool:
        """Add an entity to the universe"""
        if entity_id in self.entities:
            logger.warning(f"Entity {entity_id} already exists in universe {self.universe_id}")
            return False
            
        # Add to universe entities
        self.entities[entity_id] = properties
        
        # Add to physics engine if it has physical properties
        if 'position' in properties:
            self.physics.add_entity(entity_id, properties)
            
        # Log event
        event = {
            'type': 'ENTITY_CREATED',
            'entity_id': entity_id,
            'properties': properties
        }
        self.causality.log_event(event)
        
        logger.info(f"Entity {entity_id} added to universe {self.universe_id}")
        return True
    
    def update_entity(self, entity_id: str, properties: Dict[str, Any]) -> bool:
        """Update an entity's properties"""
        if entity_id not in self.entities:
            logger.warning(f"Entity {entity_id} does not exist in universe {self.universe_id}")
            return False
            
        # Update universe entities
        self.entities[entity_id].update(properties)
        
        # Update in physics engine if it has physical properties
        if 'position' in properties or 'velocity' in properties or 'mass' in properties:
            self.physics.update_entity(entity_id, properties)
            
        # Log event
        event = {
            'type': 'ENTITY_UPDATED',
            'entity_id': entity_id,
            'properties': properties
        }
        self.causality.log_event(event)
        
        logger.debug(f"Entity {entity_id} updated in universe {self.universe_id}")
        return True
    
    def remove_entity(self, entity_id: str) -> bool:
        """Remove an entity from the universe"""
        if entity_id not in self.entities:
            logger.warning(f"Entity {entity_id} does not exist in universe {self.universe_id}")
            return False
            
        # Remove from universe entities
        del self.entities[entity_id]
        
        # Remove from physics engine
        self.physics.remove_entity(entity_id)
        
        # Log event
        event = {
            'type': 'ENTITY_REMOVED',
            'entity_id': entity_id
        }
        self.causality.log_event(event)
        
        logger.info(f"Entity {entity_id} removed from universe {self.universe_id}")
        return True
    
    def register_agent(self, agent_id: str, properties: Dict[str, Any]) -> bool:
        """Register an agent in the universe"""
        if agent_id in self.agents:
            logger.warning(f"Agent {agent_id} already registered in universe {self.universe_id}")
            return False
            
        # Add to agents
        self.agents[agent_id] = properties
        
        # Also add as an entity
        entity_properties = properties.copy()
        entity_properties['is_agent'] = True
        self.add_entity(agent_id, entity_properties)
        
        logger.info(f"Agent {agent_id} registered in universe {self.universe_id}")
        return True
    
    def process_agent_action(self, agent_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        """Process an action from an agent"""
        if agent_id not in self.agents:
            logger.warning(f"Agent {agent_id} not registered in universe {self.universe_id}")
            return {'success': False, 'error': 'Agent not registered'}
            
        # Log event
        event = {
            'type': 'AGENT_ACTION',
            'agent_id': agent_id,
            'action': action
        }
        event_id = self.causality.log_event(event)
        
        # Process different action types
        action_type = action.get('type', '').upper()
        
        if action_type == 'MOVE':
            # Get current position
            if agent_id not in self.entities or 'position' not in self.entities[agent_id]:
                return {'success': False, 'error': 'Agent has no position'}
                
            current_pos = self.entities[agent_id]['position']
            
            # Get direction and intensity
            direction = action.get('parameters', {}).get('direction', 'NORTH')
            intensity = float(action.get('parameters', {}).get('intensity', 1.0))
            
            # Calculate new position
            if direction == 'NORTH':
                new_pos = (current_pos[0], current_pos[1], current_pos[2] - intensity)
            elif direction == 'SOUTH':
                new_pos = (current_pos[0], current_pos[1], current_pos[2] + intensity)
            elif direction == 'EAST':
                new_pos = (current_pos[0] + intensity, current_pos[1], current_pos[2])
            elif direction == 'WEST':
                new_pos = (current_pos[0] - intensity, current_pos[1], current_pos[2])
            else:
                return {'success': False, 'error': 'Invalid direction'}
                
            # Update position
            self.update_entity(agent_id, {'position': new_pos})
            
            # Create effect event for the move
            effect_event = {
                'type': 'AGENT_MOVED',
                'agent_id': agent_id,
                'old_position': current_pos,
                'new_position': new_pos
            }
            effect_id = self.causality.log_event(effect_event)
            
            # Link cause and effect
            self.causality.add_causal_link(event_id, effect_id)
            
            return {'success': True, 'new_position': new_pos}
            
        elif action_type == 'SPEAK':
            # Get message content
            message = action.get('parameters', {}).get('message', '')
            
            if not message:
                return {'success': False, 'error': 'No message provided'}
                
            # Get nearby agents
            if 'position' in self.entities[agent_id]:
                pos = self.entities[agent_id]['position']
                nearby_entities = self.physics.get_nearby_entities(pos, 10.0)
                
                # Filter to only agents
                nearby_agents = [e for e in nearby_entities if e in self.agents and e != agent_id]
                
                # Create effect events for each nearby agent receiving the message
                for nearby_agent in nearby_agents:
                    effect_event = {
                        'type': 'AGENT_RECEIVED_MESSAGE',
                        'agent_id': nearby_agent,
                        'from_agent': agent_id,
                        'message': message
                    }
                    effect_id = self.causality.log_event(effect_event)
                    
                    # Link cause and effect
                    self.causality.add_causal_link(event_id, effect_id)
            
            return {'success': True, 'message': message, 'heard_by': len(nearby_agents) if 'nearby_agents' in locals() else 0}
            
        elif action_type == 'INTERACT':
            # Get target entity
            target = action.get('parameters', {}).get('target', '')
            
            if not target or target not in self.entities:
                return {'success': False, 'error': 'Invalid target entity'}
                
            # Get interaction type
            interaction = action.get('parameters', {}).get('interaction', 'INSPECT')
            
            # Process different interactions
            if interaction == 'INSPECT':
                # Return information about the target
                return {
                    'success': True,
                    'target': target,
                    'properties': self.entities[target]
                }
            elif interaction == 'MODIFY':
                # Get modifications
                modifications = action.get('parameters', {}).get('modifications', {})
                
                if not modifications:
                    return {'success': False, 'error': 'No modifications provided'}
                    
                # Apply modifications
                self.update_entity(target, modifications)
                
                # Create effect event
                effect_event = {
                    'type': 'ENTITY_MODIFIED',
                    'entity_id': target,
                    'by_agent': agent_id,
                    'modifications': modifications
                }
                effect_id = self.causality.log_event(effect_event)
                
                # Link cause and effect
                self.causality.add_causal_link(event_id, effect_id)
                
                return {'success': True, 'target': target, 'applied_modifications': modifications}
            else:
                return {'success': False, 'error': 'Invalid interaction type'}
        
        # Default response for unhandled action types
        return {'success': False, 'error': 'Unhandled action type'}
EOFINNER

# Create the MetaKern kernel module
cat > /usr/src/metakern-1.0/Makefile << 'EOFINNER'
obj-m := metakern.o
KDIR := /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

default:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
EOFINNER

cat > /usr/src/metakern-1.0/metakern.c << 'EOFINNER'
/*
 * MetaKern - Metaphysical Kernel Extension for SYNTHIX OS
 * 
 * This kernel module provides system calls for managing AI agent processes
 * with special memory handling and time dilation capabilities.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/sched.h>
#include <linux/time.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("SYNTHIX OS Team");
MODULE_DESCRIPTION("Metaphysical Kernel Extension for SYNTHIX OS");
MODULE_VERSION("0.1");

#define METAKERN_PROC_NAME "metakern"
#define MAX_AGENTS 1024
#define AGENT_NAME_MAX 64

/* Agent data structure */
struct metakern_agent {
    char name[AGENT_NAME_MAX];
    pid_t pid;
    unsigned long memory_limit;
    float time_dilation;
    unsigned long long creation_time;
    int universe_id;
    int active;
};

/* Array of registered agents */
static struct metakern_agent *agents = NULL;
static int agent_count = 0;
static DEFINE_MUTEX(agent_mutex);

/* Proc file operations */
static int metakern_proc_show(struct seq_file *m, void *v)
{
    int i;
    
    mutex_lock(&agent_mutex);
    
    seq_printf(m, "MetaKern - SYNTHIX OS Metaphysical Kernel Extension\n");
    seq_printf(m, "Version: 0.1\n\n");
    seq_printf(m, "Registered Agents: %d / %d\n\n", agent_count, MAX_AGENTS);
    
    seq_printf(m, "ID  | Name                 | PID    | Universe | Time Dilation | Memory Limit \n");
    seq_printf(m, "----+----------------------+--------+----------+---------------+--------------\n");
    
    for (i = 0; i < agent_count; i++) {
        if (agents[i].active) {
            seq_printf(m, "%-3d | %-20s | %-6d | %-8d | %-13.2f | %-12lu\n",
                       i,
                       agents[i].name,
                       agents[i].pid,
                       agents[i].universe_id,
                       agents[i].time_dilation,
                       agents[i].memory_limit);
        }
    }
    
    mutex_unlock(&agent_mutex);
    
    return 0;
}

static int metakern_proc_open(struct inode *inode, struct file *file)
{
    return single_open(file, metakern_proc_show, NULL);
}

/* Command processing */
static ssize_t metakern_proc_write(struct file *file, const char __user *buffer,
                                  size_t count, loff_t *pos)
{
    char *cmd_buf, *cmd, *arg1, *arg2, *arg3, *arg4;
    int ret = count;
    
    /* Allocate command buffer */
    cmd_buf = kmalloc(count + 1, GFP_KERNEL);
    if (!cmd_buf)
        return -ENOMEM;
    
    /* Copy command from user space */
    if (copy_from_user(cmd_buf, buffer, count)) {
        kfree(cmd_buf);
        return -EFAULT;
    }
    
    /* Null-terminate the command */
    cmd_buf[count] = '\0';
    
    /* Parse the command */
    cmd = strsep(&cmd_buf, " \t\n");
    if (!cmd) {
        kfree(cmd_buf);
        return -EINVAL;
    }
    
    /* Process commands */
    if (strcmp(cmd, "register") == 0) {
        /* register <name> <pid> <universe_id> <time_dilation> <memory_limit> */
        arg1 = strsep(&cmd_buf, " \t\n");  /* name */
        arg2 = strsep(&cmd_buf, " \t\n");  /* pid */
        arg3 = strsep(&cmd_buf, " \t\n");  /* universe_id */
        arg4 = strsep(&cmd_buf, " \t\n");  /* time_dilation */
        
        if (arg1 && arg2 && arg3 && arg4) {
            struct metakern_agent new_agent;
            
            /* Fill in agent data */
            strncpy(new_agent.name, arg1, AGENT_NAME_MAX - 1);
            new_agent.name[AGENT_NAME_MAX - 1] = '\0';
            new_agent.pid = simple_strtol(arg2, NULL, 10);
            new_agent.universe_id = simple_strtol(arg3, NULL, 10);
            new_agent.time_dilation = simple_strtol(arg4, NULL, 10);
            new_agent.memory_limit = 128 * 1024 * 1024;  /* Default: 128 MB */
            new_agent.creation_time = ktime_get_ns();
            new_agent.active = 1;
            
            /* Add to agent list */
            mutex_lock(&agent_mutex);
            
            if (agent_count < MAX_AGENTS) {
                agents[agent_count++] = new_agent;
                printk(KERN_INFO "MetaKern: Registered agent '%s' (PID %d)\n", 
                       new_agent.name, new_agent.pid);
            } else {
                printk(KERN_WARNING "MetaKern: Cannot register agent '%s', maximum reached\n",
                       new_agent.name);
                ret = -ENOSPC;
            }
            
            mutex_unlock(&agent_mutex);
        } else {
            printk(KERN_WARNING "MetaKern: Invalid register command format\n");
            ret = -EINVAL;
        }
    } else if (strcmp(cmd, "unregister") == 0) {
        /* unregister <name> */
        arg1 = strsep(&cmd_buf, " \t\n");  /* name */
        
        if (arg1) {
            int i, found = 0;
            
            mutex_lock(&agent_mutex);
            
            for (i = 0; i < agent_count; i++) {
                if (strcmp(agents[i].name, arg1) == 0) {
                    agents[i].active = 0;
                    found = 1;
                    printk(KERN_INFO "MetaKern: Unregistered agent '%s'\n", arg1);
                    break;
                }
            }
            
            mutex_unlock(&agent_mutex);
            
            if (!found) {
                printk(KERN_WARNING "MetaKern: Agent '%s' not found\n", arg1);
                ret = -ENOENT;
            }
        } else {
            printk(KERN_WARNING "MetaKern: Invalid unregister command format\n");
            ret = -EINVAL;
        }
    } else if (strcmp(cmd, "dilate") == 0) {
        /* dilate <name> <factor> */
        arg1 = strsep(&cmd_buf, " \t\n");  /* name */
        arg2 = strsep(&cmd_buf, " \t\n");  /* factor */
        
        if (arg1 && arg2) {
            int i, found = 0;
            float factor = simple_strtol(arg2, NULL, 10);
            
            mutex_lock(&agent_mutex);
            
            for (i = 0; i < agent_count; i++) {
                if (strcmp(agents[i].name, arg1) == 0) {
                    agents[i].time_dilation = factor;
                    found = 1;
                    printk(KERN_INFO "MetaKern: Set time dilation for agent '%s' to %.2f\n",
                           arg1, factor);
                    break;
                }
            }
            
            mutex_unlock(&agent_mutex);
            
            if (!found) {
                printk(KERN_WARNING "MetaKern: Agent '%s' not found\n", arg1);
                ret = -ENOENT;
            }
        } else {
            printk(KERN_WARNING "MetaKern: Invalid dilate command format\n");
            ret = -EINVAL;
        }
    } else {
        printk(KERN_WARNING "MetaKern: Unknown command '%s'\n", cmd);
        ret = -EINVAL;
    }
    
    kfree(cmd_buf);
    return ret;
}

static const struct file_operations metakern_proc_fops = {
    .owner = THIS_MODULE,
    .open = metakern_proc_open,
    .read = seq_read,
    .write = metakern_proc_write,
    .llseek = seq_lseek,
    .release = single_release,
};

static int __init metakern_init(void)
{
    struct proc_dir_entry *proc_entry;
    
    printk(KERN_INFO "MetaKern: Initializing SYNTHIX OS Metaphysical Kernel Extension\n");
    
    /* Allocate agent array */
    agents = kmalloc(sizeof(struct metakern_agent) * MAX_AGENTS, GFP_KERNEL);
    if (!agents) {
        printk(KERN_ERR "MetaKern: Failed to allocate agent array\n");
        return -ENOMEM;
    }
    
    /* Create proc entry */
    proc_entry = proc_create(METAKERN_PROC_NAME, 0666, NULL, &metakern_proc_fops);
    if (!proc_entry) {
        printk(KERN_ERR "MetaKern: Failed to create proc entry\n");
        kfree(agents);
        return -ENOMEM;
    }
    
    printk(KERN_INFO "MetaKern: Ready for metaphysical computation\n");
    
    return 0;
}

static void __exit metakern_exit(void)
{
    printk(KERN_INFO "MetaKern: Shutting down\n");
    
    /* Remove proc entry */
    remove_proc_entry(METAKERN_PROC_NAME, NULL);
    
    /* Free agent array */
    kfree(agents);
    
    printk(KERN_INFO "MetaKern: Module unloaded\n");
}

module_init(metakern_init);
module_exit(metakern_exit);
EOFINNER

# Create a simple init script for SYNTHIX
mkdir -p /etc/synthix/init.d

cat > /etc/synthix/init.d/00-universe-init.sh << 'EOFINNER'
#!/bin/bash
# Initialize default universe

# Create logs directory
mkdir -p /var/log/synthix

# Load the MetaKern module
modprobe metakern

# Create default universe config
cat > /var/lib/synthix/universes/genesis.json << 'EOT'
{
    "name": "Genesis",
    "physics_constants": {
        "lightspeed": 299792458,
        "gravity": 9.8,
        "entropy_rate": 0.01,
        "quantum_uncertainty": 0.05
    },
    "agent_perception_resolution": 0.001,
    "simulation_time_dilation": 1.0,
    "metacognition_enabled": true,
    "causality_enforcement": "strict",
    "emergence_threshold": 0.75,
    "universe_tick_rate": 100
}
EOT

# Start the universe
/usr/lib/synthix/use.py universe create --id genesis --config /var/lib/synthix/universes/genesis.json
/usr/lib/synthix/use.py universe start --id genesis

echo "SYNTHIX universe 'Genesis' initialized and started"
EOFINNER

chmod +x /etc/synthix/init.d/00-universe-init.sh

# Create systemd service for SYNTHIX
cat > /etc/systemd/system/synthix.service << 'EOFINNER'
[Unit]
Description=SYNTHIX OS Universe Simulation
After=network.target

[Service]
Type=forking
ExecStart=/bin/bash -c "for script in /etc/synthix/init.d/*.sh; do $script; done"
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOFINNER

# Create SYNTHIX shell for interacting with the system
cat > /usr/bin/synthix << 'EOFINNER'
#!/bin/bash
# SYNTHIX Shell - Interface for managing AI simulation universes

SYNTHIX_VERSION="0.1"

function show_help {
    echo "SYNTHIX Shell v${SYNTHIX_VERSION}"
    echo "Usage: synthix <command> [options]"
    echo
    echo "Commands:"
    echo "  universe list                         List all universes"
    echo "  universe create <name> [options]      Create a new universe"
    echo "  universe start <id>                   Start a universe"
    echo "  universe stop <id>                    Stop a universe"
    echo "  universe status <id>                  Show universe status"
    echo
    echo "  agent list [universe_id]              List all agents"
    echo "  agent create <name> <universe_id>     Create a new agent"
    echo "  agent stop <id> <universe_id>         Stop an agent"
    echo "  agent status <id> <universe_id>       Show agent status"
    echo
    echo "  meta dilate <agent_id> <factor>       Change time dilation for an agent"
    echo "  meta stats                            Show metaphysical statistics"
    echo
    echo "  shell                                 Start interactive shell"
    echo "  help                                  Show this help"
}

function universe_list {
    echo "Available Universes:"
    # In a real implementation, this would query the USE service
    ls -l /var/lib/synthix/universes/ | grep json | awk '{print $9}' | sed 's/.json//'
}

function universe_create {
    if [ -z "$1" ]; then
        echo "Error: Universe name required"
        return 1
    fi
    
    echo "Creating universe '$1'..."
    # In a real implementation, this would call the USE service
    
    # Create a simple config file
    cat > "/var/lib/synthix/universes/$1.json" << EOT
{
    "name": "$1",
    "physics_constants": {
        "lightspeed": 299792458,
        "gravity": 9.8,
        "entropy_rate": 0.01
    },
    "agent_perception_resolution": 0.001,
    "simulation_time_dilation": 1.0,
    "metacognition_enabled": true
}
EOT
    
    echo "Universe '$1' created"
}

function universe_start {
    if [ -z "$1" ]; then
        echo "Error: Universe ID required"
        return 1
    fi
    
    if [ ! -f "/var/lib/synthix/universes/$1.json" ]; then
        echo "Error: Universe '$1' does not exist"
        return 1
    }
    
    echo "Starting universe '$1'..."
    # In a real implementation, this would call the USE service
    /usr/lib/synthix/use.py universe start --id "$1"
    
    echo "Universe '$1' started"
}

function universe_stop {
    if [ -z "$1" ]; then
        echo "Error: Universe ID required"
        return 1
    fi
    
    echo "Stopping universe '$1'..."
    # In a real implementation, this would call the USE service
    /usr/lib/synthix/use.py universe stop --id "$1"
    
    echo "Universe '$1' stopped"
}

function agent_create {
    if [ -z "$1" ] || [ -z "$2" ]; then
        echo "Error: Agent name and universe ID required"
        return 1
    fi
    
    if [ ! -f "/var/lib/synthix/universes/$2.json" ]; then
        echo "Error: Universe '$2' does not exist"
        return 1
    fi
    
    echo "Creating agent '$1' in universe '$2'..."
    # In a real implementation, this would call the ARE service
    
    # Create a simple config file
    mkdir -p "/var/lib/synthix/agents/$2"
    cat > "/var/lib/synthix/agents/$2/$1.json" << EOT
{
    "name": "$1",
    "universe_id": "$2",
    "cognitive_architecture": "basic",
    "perception_resolution": 0.001,
    "action_capabilities": ["MOVE", "SPEAK", "INTERACT", "OBSERVE"],
    "starting_position": [0, 0, 0],
    "starting_beliefs": {}
}
EOT
    
    # Register with the kernel module
    echo "register $1 $ $2 1.0" > /proc/metakern
    
    echo "Agent '$1' created in universe '$2'"
    echo "Starting agent process..."
    
    # In a real implementation, this would start the agent process
    /usr/lib/synthix/are.py agent create --id "$1" --universe "$2" --config "/var/lib/synthix/agents/$2/$1.json"
    
    echo "Agent '$1' started"
}

function meta_dilate {
    if [ -z "$1" ] || [ -z "$2" ]; then
        echo "Error: Agent ID and dilation factor required"
        return 1
    fi
    
    echo "Setting time dilation for agent '$1' to $2..."
    # In a real implementation, this would call the kernel module
    echo "dilate $1 $2" > /proc/metakern
    
    echo "Time dilation set"
}

function meta_stats {
    echo "SYNTHIX Metaphysical System Statistics:"
    echo "--------------------------------------"
    
    # In a real implementation, this would query the kernel module
    if [ -f /proc/metakern ]; then
        cat /proc/metakern
    else
        echo "MetaKern module not loaded"
    fi
}

function interactive_shell {
    echo "SYNTHIX Interactive Shell v${SYNTHIX_VERSION}"
    echo "Type 'help' for a list of commands, 'exit' to quit"
    
    while true; do
        read -p "synthix> " cmd args
        
        case "$cmd" in
            exit|quit)
                break
                ;;
            help)
                show_help
                ;;
            universe)
                case "$args" in
                    list*)
                        universe_list
                        ;;
                    create*)
                        universe_create $(echo $args | cut -d' ' -f2-)
                        ;;
                    start*)
                        universe_start $(echo $args | cut -d' ' -f2)
                        ;;
                    stop*)
                        universe_stop $(echo $args | cut -d' ' -f2)
                        ;;
                    *)
                        echo "Unknown universe command: $args"
                        ;;
                esac
                ;;
            agent)
                case "$args" in
                    create*)
                        agent_create $(echo $args | cut -d' ' -f2-3)
                        ;;
                    *)
                        echo "Unknown agent command: $args"
                        ;;
                esac
                ;;
            meta)
                case "$args" in
                    dilate*)
                        meta_dilate $(echo $args | cut -d' ' -f2-3)
                        ;;
                    stats*)
                        meta_stats
                        ;;
                    *)
                        echo "Unknown meta command: $args"
                        ;;
                esac
                ;;
            *)
                echo "Unknown command: $cmd"
                ;;
        esac
    done
}

# Main command processing
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

case "$1" in
    help)
        show_help
        ;;
    universe)
        case "$2" in
            list)
                universe_list
                ;;
            create)
                universe_create "$3"
                ;;
            start)
                universe_start "$3"
                ;;
            stop)
                universe_stop "$3"
                ;;
            *)
                echo "Unknown universe command: $2"
                show_help
                exit 1
                ;;
        esac
        ;;
    agent)
        case "$2" in
            create)
                agent_create "$3" "$4"
                ;;
            *)
                echo "Unknown agent command: $2"
                show_help
                exit 1
                ;;
        esac
        ;;
    meta)
        case "$2" in
            dilate)
                meta_dilate "$3" "$4"
                ;;
            stats)
                meta_stats
                ;;
            *)
                echo "Unknown meta command: $2"
                show_help
                exit 1
                ;;
        esac
        ;;
    shell)
        interactive_shell
        ;;
    *)
        echo "Unknown command: $1"
        show_help
        exit 1
        ;;
esac

exit 0
EOFINNER

chmod +x /usr/bin/synthix

# Create a desktop entry for SYNTHIX Shell
mkdir -p /usr/share/applications

cat > /usr/share/applications/synthix-shell.desktop << 'EOFINNER'
[Desktop Entry]
Name=SYNTHIX Shell
Comment=Interface for SYNTHIX OS universe simulation
Exec=x-terminal-emulator -e synthix shell
Icon=terminal
Terminal=false
Type=Application
Categories=System;TerminalEmulator;
Keywords=synthix;simulation;universe;
EOFINNER

# Create documentation
mkdir -p /usr/share/doc/synthix

cat > /usr/share/doc/synthix/README.md << 'EOFINNER'
# SYNTHIX OS

SYNTHIX is a custom Linux-based operating system designed to simulate artificial
universes with AI agents running as OS-level processes.

## Key Components

### MetaKern - Metaphysical Kernel Extensions

MetaKern extends the Linux kernel with capabilities for handling AI agent processes,
including:

- Agent-process mapping with specialized memory management
- Time dilation controls for subjective agent time
- Symbolic reference architecture for metaphorical computation

### ARE - Agent Runtime Environment

The ARE provides the execution environment for AI agents, including:

- Perception pipeline for sensory data
- Action interface for universe interaction
- Belief-desire-intention framework

### USE - Universe Simulation Engine

The USE generates and manages simulated universes:

- Physics emulation with configurable laws and constants
- Entity relationship tracking
- Causal chain validation

## Getting Started

1. Boot into SYNTHIX OS
2. Open a terminal
3. Run the SYNTHIX shell: `synthix shell`
4. Create a universe: `universe create myworld`
5. Start the universe: `universe start myworld`
6. Create an agent: `agent create agent1 myworld`

## Configuration

Universe configuration files are stored in `/etc/synthix/universe.conf` and
in individual JSON files in `/var/lib/synthix/universes/`.

Agent configurations are stored in `/var/lib/synthix/agents/<universe_id>/`.

## Logging

Logs are stored in `/var/log/synthix/`.

## System Requirements

- 64-bit x86 processor
- 4GB RAM minimum (8GB recommended)
- 20GB disk space

## License

SYNTHIX OS is open source software.
EOFINNER

# Install some basic packages and clean up
apt-get autoremove -y
apt-get clean

# Set up auto-login for demo
mkdir -p /etc/systemd/system/getty@tty1.service.d/

cat > /etc/systemd/system/getty@tty1.service.d/autologin.conf << 'EOFINNER'
[Service]
ExecStart=
ExecStart=-/sbin/agetty --autologin root --noclear %I $TERM
EOFINNER

# Set up .bashrc to start SYNTHIX shell
cat >> /root/.bashrc << 'EOFINNER'
# SYNTHIX OS Welcome Message
echo "Welcome to SYNTHIX OS - Metaphysical Computing Platform"
echo "-------------------------------------------------------"
echo "Type 'synthix shell' to start the interactive shell"
echo "Type 'synthix help' for available commands"
echo ""
EOFINNER

# Set up a custom MOTD
cat > /etc/motd << 'EOFINNER'
 _____  __   __  _   _  _____  _   _  _  __  __
/  ___/ \ \ / / | \ | ||_   _|| | | || |\ \/ /
\ `--.   \ V /  |  \| |  | |  | |_| || | \  / 
 `--. \   \ /   | . ` |  | |  |  _  || | /  \ 
/\__/ /   | |   | |\  | _| |_ | | | || |/ /\ \
\____/    \_/   \_| \_/ \___/ \_| |_/\_/\_/ \_/
                                              
Metaphysical Operating System - v0.1
A simulation platform for artificial universes and AI agents

EOFINNER

# Create the boot setup
mkdir -p "${WORK_DIR}/chroot/boot/grub"
cat > "${WORK_DIR}/chroot/boot/grub/grub.cfg" << 'EOFINNER'
set timeout=5
set default=0

menuentry "SYNTHIX OS" {
    linux /vmlinuz root=/dev/sda1 ro quiet splash
    initrd /initrd.img
}

menuentry "SYNTHIX OS (Safe Mode)" {
    linux /vmlinuz root=/dev/sda1 ro single
    initrd /initrd.img
}
EOFINNER

# Create the network setup
cat > "${WORK_DIR}/chroot/etc/network/interfaces" << 'EOFINNER'
# The loopback network interface
auto lo
iface lo inet loopback

# The primary network interface
allow-hotplug eth0
iface eth0 inet dhcp
EOFINNER

# Create a minimal X environment for the GUI
cat > "${WORK_DIR}/chroot/etc/X11/xorg.conf" << 'EOFINNER'
Section "ServerLayout"
    Identifier     "SYNTHIX Layout"
    Screen         0 "Screen0" 0 0
    InputDevice    "Mouse0" "CorePointer"
    InputDevice    "Keyboard0" "CoreKeyboard"
EndSection

Section "InputDevice"
    Identifier     "Keyboard0"
    Driver         "kbd"
EndSection

Section "InputDevice"
    Identifier     "Mouse0"
    Driver         "mouse"
    Option         "Protocol" "auto"
    Option         "Device" "/dev/input/mice"
    Option         "ZAxisMapping" "4 5 6 7"
EndSection

Section "Monitor"
    Identifier     "Monitor0"
    VendorName     "SYNTHIX"
    ModelName      "Generic Monitor"
EndSection

Section "Device"
    Identifier     "Card0"
    Driver         "vesa"
EndSection

Section "Screen"
    Identifier     "Screen0"
    Device         "Card0"
    Monitor        "Monitor0"
    DefaultDepth    24
    SubSection     "Display"
        Depth       24
    EndSubSection
EndSection
EOFINNER

# Create a simple GUI for SYNTHIX
mkdir -p "${WORK_DIR}/chroot/usr/share/synthix/gui"

cat > "${WORK_DIR}/chroot/usr/share/synthix/gui/synthix-gui.py" << 'EOFINNER'
#!/usr/bin/env python3
"""
SYNTHIX GUI - Graphical interface for SYNTHIX OS
A simple universe and agent management interface
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import subprocess
import json
import os
import threading
import time

class SynthixGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SYNTHIX OS - Universe Simulation Interface")
        self.root.geometry("800x600")
        self.root.minsize(800, 600)
        
        # Set up the main frame
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.universe_tab = ttk.Frame(self.notebook)
        self.agent_tab = ttk.Frame(self.notebook)
        self.meta_tab = ttk.Frame(self.notebook)
        self.console_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.universe_tab, text="Universes")
        self.notebook.add(self.agent_tab, text="Agents")
        self.notebook.add(self.meta_tab, text="Metaphysics")
        self.notebook.add(self.console_tab, text="Console")
        
        # Set up universe tab
        self.setup_universe_tab()
        
        # Set up agent tab
        self.setup_agent_tab()
        
        # Set up metaphysics tab
        self.setup_meta_tab()
        
        # Set up console tab
        self.setup_console_tab()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("SYNTHIX OS Ready")
        self.status_bar = ttk.Label(dialog, text="Capabilities:").pack(padx=10, pady=5, anchor=tk.W)
        capabilities_frame = ttk.Frame(dialog)
        capabilities_frame.pack(padx=10, pady=5, anchor=tk.W)
        
        move_var = tk.BooleanVar(value=True)
        speak_var = tk.BooleanVar(value=True)
        interact_var = tk.BooleanVar(value=True)
        observe_var = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(capabilities_frame, text="MOVE", variable=move_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(capabilities_frame, text="SPEAK", variable=speak_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(capabilities_frame, text="INTERACT", variable=interact_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(capabilities_frame, text="OBSERVE", variable=observe_var).pack(side=tk.LEFT, padx=5)
        
        def create():
            agent_name = name_entry.get().strip()
            universe_id = universe_combo.get()
            
            if not agent_name:
                messagebox.showerror("Error", "Agent name cannot be empty")
                return
                
            if not universe_id:
                messagebox.showerror("Error", "Please select a universe")
                return
                
            try:
                # Build capabilities list
                capabilities = []
                if move_var.get(): capabilities.append("MOVE")
                if speak_var.get(): capabilities.append("SPEAK")
                if interact_var.get(): capabilities.append("INTERACT")
                if observe_var.get(): capabilities.append("OBSERVE")
                
                # Create agent config
                config = {
                    "name": agent_name,
                    "universe_id": universe_id,
                    "cognitive_architecture": arch_combo.get(),
                    "perception_resolution": 0.001,
                    "action_capabilities": capabilities,
                    "starting_position": [
                        float(x_entry.get()),
                        float(y_entry.get()),
                        float(z_entry.get())
                    ],
                    "starting_beliefs": {}
                }
                
                # Save config
                agent_dir = f"/var/lib/synthix/agents/{universe_id}"
                os.makedirs(agent_dir, exist_ok=True)
                with open(f"{agent_dir}/{agent_name}.json", 'w') as f:
                    json.dump(config, f, indent=2)
                
                # Register and start agent
                self.write_to_console(f"Creating agent '{agent_name}' in universe '{universe_id}'...\n")
                
                cmd = ["/usr/bin/synthix", "agent", "create", agent_name, universe_id]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    self.write_to_console(result.stdout + "\n")
                    messagebox.showinfo("Success", f"Agent '{agent_name}' created successfully")
                else:
                    self.write_to_console(result.stderr + "\n")
                    messagebox.showerror("Error", f"Failed to create agent: {result.stderr}")
                
                dialog.destroy()
                
                # Refresh agent list
                self.refresh_agents()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to create agent: {str(e)}")
        
        buttons_frame = ttk.Frame(dialog)
        buttons_frame.pack(pady=10)
        
        ttk.Button(buttons_frame, text="Create", command=create).pack(side=tk.LEFT, padx=10)
        ttk.Button(buttons_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=10)
    
    def stop_agent(self):
        if not self.agent_listbox.curselection():
            messagebox.showwarning("Warning", "Please select an agent to stop")
            return
            
        agent_info = self.agent_listbox.get(self.agent_listbox.curselection()[0])
        agent_parts = agent_info.split(" in universe ")
        
        if len(agent_parts) != 2:
            return
            
        agent_id = agent_parts[0]
        universe_id = agent_parts[1]
        
        try:
            # Run command to stop agent
            self.write_to_console(f"Stopping agent '{agent_id}' in universe '{universe_id}'...\n")
            
            cmd = ["/usr/bin/synthix", "agent", "stop", agent_id, universe_id]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.write_to_console(result.stdout + "\n")
                messagebox.showinfo("Success", f"Agent '{agent_id}' stopped successfully")
            else:
                self.write_to_console(result.stderr + "\n")
                messagebox.showerror("Error", f"Failed to stop agent: {result.stderr}")
                
            # Refresh agent list
            self.refresh_agents()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop agent: {str(e)}")
    
    def apply_time_dilation(self):
        agent_info = self.dilation_agent.get()
        if not agent_info:
            messagebox.showwarning("Warning", "Please select an agent")
            return
            
        dilation = self.dilation_value.get()
        try:
            dilation = float(dilation)
        except ValueError:
            messagebox.showerror("Error", "Invalid dilation value")
            return
            
        agent_parts = agent_info.split(" in universe ")
        if len(agent_parts) != 2:
            return
            
        agent_id = agent_parts[0]
        
        try:
            # Run command to set time dilation
            self.write_to_console(f"Setting time dilation for agent '{agent_id}' to {dilation}...\n")
            
            cmd = ["/usr/bin/synthix", "meta", "dilate", agent_id, str(dilation)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.write_to_console(result.stdout + "\n")
                messagebox.showinfo("Success", f"Time dilation set successfully")
            else:
                self.write_to_console(result.stderr + "\n")
                messagebox.showerror("Error", f"Failed to set time dilation: {result.stderr}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to set time dilation: {str(e)}")
    
    def execute_command(self, event=None):
        command = self.console_input.get().strip()
        if not command:
            return
            
        self.write_to_console(f"> {command}\n")
        
        try:
            # Execute command
            cmd = ["/bin/bash", "-c", command]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.stdout:
                self.write_to_console(result.stdout + "\n")
                
            if result.stderr:
                self.write_to_console("Error: " + result.stderr + "\n")
                
            # Clear input
            self.console_input.delete(0, tk.END)
        except Exception as e:
            self.write_to_console(f"Error executing command: {str(e)}\n")
    
    def clear_console(self):
        self.console_output.config(state=tk.NORMAL)
        self.console_output.delete(1.0, tk.END)
        self.console_output.config(state=tk.DISABLED)
    
    def write_to_console(self, text):
        self.console_output.config(state=tk.NORMAL)
        self.console_output.insert(tk.END, text)
        self.console_output.see(tk.END)
        self.console_output.config(state=tk.DISABLED)
    
    def monitor_system(self):
        """Background thread to monitor the system and update stats"""
        while self.running:
            # Only update if the Meta tab is active
            if self.notebook.index(self.notebook.select()) == 2:
                self.refresh_meta_stats()
                
            time.sleep(5)
    
    def on_closing(self):
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SynthixGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
EOFINNER

cat > "${WORK_DIR}/chroot/usr/share/applications/synthix-gui.desktop" << 'EOFINNER'
[Desktop Entry]
Name=SYNTHIX OS
Comment=Universe Simulation Platform
Exec=python3 /usr/share/synthix/gui/synthix-gui.py
Icon=computer
Terminal=false
Type=Application
Categories=System;
Keywords=synthix;simulation;universe;
EOFINNER

# Make the GUI executable
chmod +x "${WORK_DIR}/chroot/usr/share/synthix/gui/synthix-gui.py"

# Create basic X session for SYNTHIX
cat > "${WORK_DIR}/chroot/etc/X11/xinit/xinitrc" << 'EOFINNER'
#!/bin/sh

# Run system-wide settings
if [ -d /etc/X11/xinit/xinitrc.d ]; then
    for f in /etc/X11/xinit/xinitrc.d/?*.sh; do
        [ -x "$f" ] && . "$f"
    done
    unset f
fi

# Start basic window manager
if which openbox > /dev/null 2>&1; then
    exec openbox-session
elif which fluxbox > /dev/null 2>&1; then
    exec fluxbox
else
    # Fallback to twm
    twm &
    xclock -geometry 50x50-1+1 &
    xterm -geometry 80x50+494+51 &
    xterm -geometry 80x20+494-0 &
fi

# Start SYNTHIX GUI
python3 /usr/share/synthix/gui/synthix-gui.py &
EOFINNER

# Create a README file explaining how to build the ISO
cat > README.md << 'EOFINNER'
# SYNTHIX OS ISO Builder

This script will build a bootable ISO image of SYNTHIX OS, a metaphysical
Linux-based operating system designed to simulate artificial universes with
AI agents running as OS-level processes.

## Requirements

- Debian-based Linux distribution (Ubuntu, Debian, etc.)
- debootstrap package installed
- xorriso package installed
- sudo privileges

## Usage

1. Make the script executable:
   ```
   chmod +x synthix-build.sh
   ```

2. Run the script with sudo:
   ```
   sudo ./synthix-build.sh
   ```

3. The ISO will be created in the `synthix-output` directory.

## Booting SYNTHIX OS

You can boot SYNTHIX OS in a virtual machine:
- VirtualBox: New VM > Linux > Debian (64-bit) > Use ISO image
- QEMU: `qemu-system-x86_64 -cdrom SYNTHIX-OS-v0.1.iso -m 2G`
- VMware: Create new VM > Install from ISO

Or burn to a USB drive (replace X with your USB device letter):
```
sudo dd if=SYNTHIX-OS-v0.1.iso of=/dev/sdX bs=4M status=progress
```

## Getting Started with SYNTHIX

Once booted, login as root (no password required for demo purposes).

1. Use the SYNTHIX shell to create and manage universes and agents:
   ```
   synthix shell
   ```

2. Or use the graphical interface:
   ```
   startx
   ```

3. Basic commands:
   ```
   universe create myworld
   universe start myworld
   agent create myagent myworld
   meta stats
   ```

## Customizing SYNTHIX

- Universe configurations: /etc/synthix/universe.conf
- Agent runtime: /usr/lib/synthix/are.py
- Universe simulation: /usr/lib/synthix/use.py
- MetaKern kernel module: /usr/src/metakern-1.0

## Support

This is a conceptual prototype system. For more information, see the documentation
in /usr/share/doc/synthix/.
EOFINNER

# Set up the ISO creation part of the script
cat >> "${WORK_DIR}/chroot/synthix-setup.sh" << 'EOFINNER'
# Enable services
systemctl enable synthix.service

# Clean up
apt-get autoremove -y
apt-get clean

# Set root password (empty for demo purposes)
passwd -d root

# Exit chroot setup
exit 0
EOFINNER

# Add the rest of the ISO building script
cat << 'EOFINNER'
# 3. Copy the setup script into chroot and make it executable
chmod +x ${WORK_DIR}/chroot/synthix-setup.sh

# 4. Prepare chroot environment for running setup script
mount -t proc none ${WORK_DIR}/chroot/proc
mount -o bind /dev ${WORK_DIR}/chroot/dev
mount -o bind /dev/pts ${WORK_DIR}/chroot/dev/pts

# 5. Run setup script inside chroot
chroot ${WORK_DIR}/chroot /synthix-setup.sh

# 6. Cleanup chroot environment
umount ${WORK_DIR}/chroot/dev/pts
umount ${WORK_DIR}/chroot/dev
umount ${WORK_DIR}/chroot/proc

# 7. Copy kernel and initrd from chroot
cp ${WORK_DIR}/chroot/boot/vmlinuz-* ${WORK_DIR}/iso/vmlinuz
cp ${WORK_DIR}/chroot/boot/initrd.img-* ${WORK_DIR}/iso/initrd.img

# 8. Create ISO image
echo "Creating ISO image..."
mkdir -p ${WORK_DIR}/iso/boot/grub
cp ${WORK_DIR}/chroot/boot/grub/grub.cfg ${WORK_DIR}/iso/boot/grub/

# Create GRUB image
grub-mkrescue -o ${OUTPUT_DIR}/${ISO_NAME} ${WORK_DIR}/iso

echo "SYNTHIX OS ISO created at: ${OUTPUT_DIR}/${ISO_NAME}"
echo "Done!"

# 9. Clean up
if [ -z "$KEEP_BUILD" ]; then
    echo "Cleaning up build directory..."
    rm -rf ${WORK_DIR}
fi

exit 0
EOFINNER

# Make the whole script executable
chmod +x synthix-build.sh

echo "SYNTHIX OS ISO builder script created successfully."
echo "To build the ISO, run: sudo ./synthix-build.sh"
root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Initialize data
        self.refresh_universes()
        self.refresh_agents()
        
        # Start monitoring thread
        self.running = True
        self.monitor_thread = threading.Thread(target=self.monitor_system)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def setup_universe_tab(self):
        # Left frame - Universe list
        left_frame = ttk.LabelFrame(self.universe_tab, text="Available Universes")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Universe listbox
        self.universe_listbox = tk.Listbox(left_frame)
        self.universe_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbar for listbox
        scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.universe_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.universe_listbox.config(yscrollcommand=scrollbar.set)
        
        # Universe listbox selection event
        self.universe_listbox.bind('<<ListboxSelect>>', self.on_universe_select)
        
        # Right frame - Universe details and actions
        right_frame = ttk.LabelFrame(self.universe_tab, text="Universe Details")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Details text
        self.universe_details = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, width=40, height=10)
        self.universe_details.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.universe_details.config(state=tk.DISABLED)
        
        # Buttons frame
        buttons_frame = ttk.Frame(right_frame)
        buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Action buttons
        ttk.Button(buttons_frame, text="Create Universe", command=self.create_universe).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Start Universe", command=self.start_universe).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Stop Universe", command=self.stop_universe).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Refresh", command=self.refresh_universes).pack(side=tk.LEFT, padx=5)
    
    def setup_agent_tab(self):
        # Left frame - Agent list
        left_frame = ttk.LabelFrame(self.agent_tab, text="Active Agents")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Agent listbox
        self.agent_listbox = tk.Listbox(left_frame)
        self.agent_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbar for listbox
        scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.agent_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.agent_listbox.config(yscrollcommand=scrollbar.set)
        
        # Agent listbox selection event
        self.agent_listbox.bind('<<ListboxSelect>>', self.on_agent_select)
        
        # Right frame - Agent details and actions
        right_frame = ttk.LabelFrame(self.agent_tab, text="Agent Details")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Details text
        self.agent_details = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, width=40, height=10)
        self.agent_details.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.agent_details.config(state=tk.DISABLED)
        
        # Buttons frame
        buttons_frame = ttk.Frame(right_frame)
        buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Action buttons
        ttk.Button(buttons_frame, text="Create Agent", command=self.create_agent).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Stop Agent", command=self.stop_agent).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Refresh", command=self.refresh_agents).pack(side=tk.LEFT, padx=5)
    
    def setup_meta_tab(self):
        # Top frame - Controls
        top_frame = ttk.LabelFrame(self.meta_tab, text="Metaphysical Controls")
        top_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Time dilation control
        dilation_frame = ttk.Frame(top_frame)
        dilation_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(dilation_frame, text="Agent:").pack(side=tk.LEFT, padx=5)
        self.dilation_agent = ttk.Combobox(dilation_frame, width=20)
        self.dilation_agent.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(dilation_frame, text="Time Dilation:").pack(side=tk.LEFT, padx=5)
        self.dilation_value = ttk.Spinbox(dilation_frame, from_=0.1, to=10.0, increment=0.1, width=5)
        self.dilation_value.set(1.0)
        self.dilation_value.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(dilation_frame, text="Apply", command=self.apply_time_dilation).pack(side=tk.LEFT, padx=5)
        
        # Bottom frame - Stats
        bottom_frame = ttk.LabelFrame(self.meta_tab, text="Metaphysical Statistics")
        bottom_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Stats text
        self.meta_stats = scrolledtext.ScrolledText(bottom_frame, wrap=tk.WORD)
        self.meta_stats.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.meta_stats.config(state=tk.DISABLED)
        
        # Refresh button
        ttk.Button(bottom_frame, text="Refresh Stats", command=self.refresh_meta_stats).pack(padx=5, pady=5)
    
    def setup_console_tab(self):
        # Console frame
        console_frame = ttk.Frame(self.console_tab)
        console_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Output text area
        self.console_output = scrolledtext.ScrolledText(console_frame, wrap=tk.WORD, bg='black', fg='green')
        self.console_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.console_output.config(state=tk.DISABLED)
        
        # Input frame
        input_frame = ttk.Frame(console_frame)
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Command:").pack(side=tk.LEFT, padx=5)
        self.console_input = ttk.Entry(input_frame)
        self.console_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.console_input.bind('<Return>', self.execute_command)
        
        ttk.Button(input_frame, text="Execute", command=self.execute_command).pack(side=tk.LEFT, padx=5)
        ttk.Button(input_frame, text="Clear", command=self.clear_console).pack(side=tk.LEFT, padx=5)
    
    def on_universe_select(self, event):
        if not self.universe_listbox.curselection():
            return
            
        universe_id = self.universe_listbox.get(self.universe_listbox.curselection()[0])
        
        try:
            # Load universe config
            config_path = f"/var/lib/synthix/universes/{universe_id}.json"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    
                # Update details text
                self.universe_details.config(state=tk.NORMAL)
                self.universe_details.delete(1.0, tk.END)
                
                self.universe_details.insert(tk.END, f"Name: {config.get('name', universe_id)}\n\n")
                self.universe_details.insert(tk.END, "Physics Constants:\n")
                
                for key, value in config.get('physics_constants', {}).items():
                    self.universe_details.insert(tk.END, f"  {key}: {value}\n")
                
                self.universe_details.insert(tk.END, f"\nAgent Perception Resolution: {config.get('agent_perception_resolution', 'N/A')}\n")
                self.universe_details.insert(tk.END, f"Time Dilation: {config.get('simulation_time_dilation', 'N/A')}\n")
                self.universe_details.insert(tk.END, f"Metacognition Enabled: {config.get('metacognition_enabled', 'N/A')}\n")
                
                self.universe_details.config(state=tk.DISABLED)
            else:
                self.universe_details.config(state=tk.NORMAL)
                self.universe_details.delete(1.0, tk.END)
                self.universe_details.insert(tk.END, f"Universe '{universe_id}' configuration not found.")
                self.universe_details.config(state=tk.DISABLED)
        except Exception as e:
            self.universe_details.config(state=tk.NORMAL)
            self.universe_details.delete(1.0, tk.END)
            self.universe_details.insert(tk.END, f"Error loading universe details: {str(e)}")
            self.universe_details.config(state=tk.DISABLED)
    
    def on_agent_select(self, event):
        if not self.agent_listbox.curselection():
            return
            
        agent_info = self.agent_listbox.get(self.agent_listbox.curselection()[0])
        agent_parts = agent_info.split(" in universe ")
        
        if len(agent_parts) != 2:
            return
            
        agent_id = agent_parts[0]
        universe_id = agent_parts[1]
        
        try:
            # Load agent config
            config_path = f"/var/lib/synthix/agents/{universe_id}/{agent_id}.json"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    
                # Update details text
                self.agent_details.config(state=tk.NORMAL)
                self.agent_details.delete(1.0, tk.END)
                
                self.agent_details.insert(tk.END, f"Name: {config.get('name', agent_id)}\n")
                self.agent_details.insert(tk.END, f"Universe: {config.get('universe_id', universe_id)}\n\n")
                
                self.agent_details.insert(tk.END, f"Cognitive Architecture: {config.get('cognitive_architecture', 'N/A')}\n")
                self.agent_details.insert(tk.END, f"Perception Resolution: {config.get('perception_resolution', 'N/A')}\n\n")
                
                self.agent_details.insert(tk.END, "Action Capabilities:\n")
                for capability in config.get('action_capabilities', []):
                    self.agent_details.insert(tk.END, f"  - {capability}\n")
                
                self.agent_details.insert(tk.END, f"\nStarting Position: {config.get('starting_position', 'N/A')}\n")
                
                self.agent_details.config(state=tk.DISABLED)
            else:
                self.agent_details.config(state=tk.NORMAL)
                self.agent_details.delete(1.0, tk.END)
                self.agent_details.insert(tk.END, f"Agent '{agent_id}' configuration not found.")
                self.agent_details.config(state=tk.DISABLED)
        except Exception as e:
            self.agent_details.config(state=tk.NORMAL)
            self.agent_details.delete(1.0, tk.END)
            self.agent_details.insert(tk.END, f"Error loading agent details: {str(e)}")
            self.agent_details.config(state=tk.DISABLED)
    
    def refresh_universes(self):
        # Clear existing entries
        self.universe_listbox.delete(0, tk.END)
        
        try:
            # Get universe list
            universe_dir = "/var/lib/synthix/universes/"
            if os.path.exists(universe_dir):
                for filename in os.listdir(universe_dir):
                    if filename.endswith(".json"):
                        universe_id = filename[:-5]  # Remove .json extension
                        self.universe_listbox.insert(tk.END, universe_id)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh universes: {str(e)}")
        
        # Update status
        self.status_var.set(f"Found {self.universe_listbox.size()} universe(s)")
    
    def refresh_agents(self):
        # Clear existing entries
        self.agent_listbox.delete(0, tk.END)
        self.dilation_agent.set("")
        
        agents = []
        
        try:
            # Get agent list
            agents_dir = "/var/lib/synthix/agents/"
            if os.path.exists(agents_dir):
                for universe_dir in os.listdir(agents_dir):
                    universe_path = os.path.join(agents_dir, universe_dir)
                    if os.path.isdir(universe_path):
                        for filename in os.listdir(universe_path):
                            if filename.endswith(".json"):
                                agent_id = filename[:-5]  # Remove .json extension
                                agent_info = f"{agent_id} in universe {universe_dir}"
                                self.agent_listbox.insert(tk.END, agent_info)
                                agents.append(agent_info)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh agents: {str(e)}")
        
        # Update agent dropdown
        self.dilation_agent['values'] = agents
        
        # Update status
        self.status_var.set(f"Found {self.agent_listbox.size()} agent(s)")
    
    def refresh_meta_stats(self):
        try:
            # Read from kernel module
            if os.path.exists("/proc/metakern"):
                with open("/proc/metakern", 'r') as f:
                    stats = f.read()
                    
                self.meta_stats.config(state=tk.NORMAL)
                self.meta_stats.delete(1.0, tk.END)
                self.meta_stats.insert(tk.END, stats)
                self.meta_stats.config(state=tk.DISABLED)
            else:
                self.meta_stats.config(state=tk.NORMAL)
                self.meta_stats.delete(1.0, tk.END)
                self.meta_stats.insert(tk.END, "MetaKern module not loaded.")
                self.meta_stats.config(state=tk.DISABLED)
        except Exception as e:
            self.meta_stats.config(state=tk.NORMAL)
            self.meta_stats.delete(1.0, tk.END)
            self.meta_stats.insert(tk.END, f"Error reading MetaKern stats: {str(e)}")
            self.meta_stats.config(state=tk.DISABLED)
    
    def create_universe(self):
        # Simple dialog to create a universe
        dialog = tk.Toplevel(self.root)
        dialog.title("Create Universe")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Universe Name:").pack(padx=10, pady=5, anchor=tk.W)
        name_entry = ttk.Entry(dialog, width=30)
        name_entry.pack(padx=10, pady=5, fill=tk.X)
        
        ttk.Label(dialog, text="Gravity:").pack(padx=10, pady=5, anchor=tk.W)
        gravity_entry = ttk.Spinbox(dialog, from_=0.1, to=20.0, increment=0.1, width=10)
        gravity_entry.set(9.8)
        gravity_entry.pack(padx=10, pady=5, anchor=tk.W)
        
        ttk.Label(dialog, text="Time Dilation:").pack(padx=10, pady=5, anchor=tk.W)
        time_entry = ttk.Spinbox(dialog, from_=0.1, to=10.0, increment=0.1, width=10)
        time_entry.set(1.0)
        time_entry.pack(padx=10, pady=5, anchor=tk.W)
        
        ttk.Label(dialog, text="Enable Metacognition:").pack(padx=10, pady=5, anchor=tk.W)
        meta_var = tk.BooleanVar(value=True)
        meta_check = ttk.Checkbutton(dialog, variable=meta_var)
        meta_check.pack(padx=10, pady=5, anchor=tk.W)
        
        def create():
            universe_name = name_entry.get().strip()
            if not universe_name:
                messagebox.showerror("Error", "Universe name cannot be empty")
                return
                
            try:
                # Create universe config
                config = {
                    "name": universe_name,
                    "physics_constants": {
                        "lightspeed": 299792458,
                        "gravity": float(gravity_entry.get()),
                        "entropy_rate": 0.01
                    },
                    "agent_perception_resolution": 0.001,
                    "simulation_time_dilation": float(time_entry.get()),
                    "metacognition_enabled": meta_var.get()
                }
                
                # Save config
                os.makedirs("/var/lib/synthix/universes/", exist_ok=True)
                with open(f"/var/lib/synthix/universes/{universe_name}.json", 'w') as f:
                    json.dump(config, f, indent=2)
                
                messagebox.showinfo("Success", f"Universe '{universe_name}' created successfully")
                dialog.destroy()
                
                # Refresh universe list
                self.refresh_universes()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to create universe: {str(e)}")
        
        buttons_frame = ttk.Frame(dialog)
        buttons_frame.pack(pady=10)
        
        ttk.Button(buttons_frame, text="Create", command=create).pack(side=tk.LEFT, padx=10)
        ttk.Button(buttons_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=10)
    
    def start_universe(self):
        if not self.universe_listbox.curselection():
            messagebox.showwarning("Warning", "Please select a universe to start")
            return
            
        universe_id = self.universe_listbox.get(self.universe_listbox.curselection()[0])
        
        try:
            # Run command to start universe
            self.write_to_console(f"Starting universe '{universe_id}'...\n")
            
            cmd = ["/usr/bin/synthix", "universe", "start", universe_id]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.write_to_console(result.stdout + "\n")
                messagebox.showinfo("Success", f"Universe '{universe_id}' started successfully")
            else:
                self.write_to_console(result.stderr + "\n")
                messagebox.showerror("Error", f"Failed to start universe: {result.stderr}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start universe: {str(e)}")
    
    def stop_universe(self):
        if not self.universe_listbox.curselection():
            messagebox.showwarning("Warning", "Please select a universe to stop")
            return
            
        universe_id = self.universe_listbox.get(self.universe_listbox.curselection()[0])
        
        try:
            # Run command to stop universe
            self.write_to_console(f"Stopping universe '{universe_id}'...\n")
            
            cmd = ["/usr/bin/synthix", "universe", "stop", universe_id]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.write_to_console(result.stdout + "\n")
                messagebox.showinfo("Success", f"Universe '{universe_id}' stopped successfully")
            else:
                self.write_to_console(result.stderr + "\n")
                messagebox.showerror("Error", f"Failed to stop universe: {result.stderr}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop universe: {str(e)}")
    
    def create_agent(self):
        # Get available universes
        universes = []
        universe_dir = "/var/lib/synthix/universes/"
        if os.path.exists(universe_dir):
            for filename in os.listdir(universe_dir):
                if filename.endswith(".json"):
                    universes.append(filename[:-5])
        
        if not universes:
            messagebox.showwarning("Warning", "No universes available. Please create a universe first")
            return
        
        # Simple dialog to create an agent
        dialog = tk.Toplevel(self.root)
        dialog.title("Create Agent")
        dialog.geometry("400x350")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Agent Name:").pack(padx=10, pady=5, anchor=tk.W)
        name_entry = ttk.Entry(dialog, width=30)
        name_entry.pack(padx=10, pady=5, fill=tk.X)
        
        ttk.Label(dialog, text="Universe:").pack(padx=10, pady=5, anchor=tk.W)
        universe_combo = ttk.Combobox(dialog, values=universes, width=30)
        universe_combo.current(0)
        universe_combo.pack(padx=10, pady=5, fill=tk.X)
        
        ttk.Label(dialog, text="Cognitive Architecture:").pack(padx=10, pady=5, anchor=tk.W)
        arch_combo = ttk.Combobox(dialog, values=["basic", "advanced", "symbolic", "connectionist"], width=20)
        arch_combo.current(0)
        arch_combo.pack(padx=10, pady=5, anchor=tk.W)
        
        ttk.Label(dialog, text="Starting Position (x, y, z):").pack(padx=10, pady=5, anchor=tk.W)
        pos_frame = ttk.Frame(dialog)
        pos_frame.pack(padx=10, pady=5, anchor=tk.W)
        
        x_entry = ttk.Spinbox(pos_frame, from_=-100.0, to=100.0, increment=1.0, width=5)
        x_entry.set(0.0)
        x_entry.pack(side=tk.LEFT, padx=2)
        
        y_entry = ttk.Spinbox(pos_frame, from_=-100.0, to=100.0, increment=1.0, width=5)
        y_entry.set(0.0)
        y_entry.pack(side=tk.LEFT, padx=2)
        
        z_entry = ttk.Spinbox(pos_frame, from_=-100.0, to=100.0, increment=1.0, width=5)
        z_entry.set(0.0)
        z_entry.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(
