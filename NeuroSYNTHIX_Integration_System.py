#!/usr/bin/env python3
"""
NeuroSYNTHIX Integration System

This module integrates the NeuroQuantum Interface (11-dimensional brain-API system) with
the SYNTHIX Governance System to create a revolutionary brain-to-AI-society interface.

Key components:
1. Neural signal processing at frequencies 98.7/99.1/98.9 Hz
2. 11-dimensional thought mapping with 60.0625× time compression
3. Evolution rate of 0.042 for adaptive learning
4. Direct brain-to-agent and brain-to-society communication
5. Neural resource generation and distribution
6. Thought-to-governance projection
"""

import os
import sys
import json
import time
import numpy as np
import random
import threading
import logging
from typing import Dict, List, Any, Tuple, Set
from enum import Enum
import math
import uuid

# Import SYNTHIX governance system
sys.path.append('/usr/lib/synthix')
from governance_system import (GovernanceSystem, SocialRelationship, SocialGroup, 
                              SocialIdentity, Policy, CulturalMeme, ResourceType)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('/var/log/neurosynthix/integration.log'), logging.StreamHandler()]
)
logger = logging.getLogger('NEUROSYNTHIX')

class BrainActivityType(Enum):
    """Types of brain activity patterns that can be detected"""
    NEUTRAL = 0
    CURIOSITY = 1
    CREATIVITY = 2
    PLANNING = 3
    DECISION = 4
    EMOTIONAL = 5
    SOCIAL = 6
    ANALYTICAL = 7
    MEMORY = 8
    SPATIAL = 9
    LINGUISTIC = 10

class NeuralDimension(Enum):
    """The 11 dimensions used in the NeuroQuantum Interface"""
    SPATIAL_X = 0
    SPATIAL_Y = 1
    SPATIAL_Z = 2
    TEMPORAL = 3
    INTENTIONALITY = 4
    CONCEPTUAL = 5
    CONTEXTUAL = 6
    ENTROPY = 7
    EMOTIONAL = 8
    MEMORY = 9
    PREDICTION = 10

class EEGProcessor:
    """Processes EEG signals into 11-dimensional thought vectors"""
    
    def __init__(self):
        """Initialize the EEG processor with our specific parameters"""
        self.target_frequencies = [98.7, 99.1, 98.9]
        self.compression_rate = 60.0625
        self.evolution_rate = 0.042
        self.dimensions = 11
        self.state_vector = np.zeros(11)
        self.last_update = time.time()
        self.calibration_matrix = np.random.random((11, 3)) - 0.5
        self.history_buffer = []  # Store recent state vectors
        self.noise_reduction = 0.85  # Noise reduction coefficient
        
    def receive_eeg_data(self, raw_data: List[float]) -> np.ndarray:
        """Process raw EEG data into an 11D state vector"""
        # Extract power at target frequencies using FFT
        frequency_powers = self._extract_frequency_powers(raw_data)
        
        # Update state vector using the frequency powers
        self._update_state_vector(frequency_powers)
        
        # Apply time compression
        compressed_vector = self._apply_time_compression()
        
        # Apply evolution (adaptive learning)
        self._apply_evolution()
        
        # Store in history buffer
        self.history_buffer.append(compressed_vector.copy())
        if len(self.history_buffer) > 100:
            self.history_buffer.pop(0)
            
        return compressed_vector
    
    def _extract_frequency_powers(self, raw_data: List[float]) -> np.ndarray:
        """Extract power at target frequencies using FFT"""
        if len(raw_data) < 128:
            # Pad with zeros if not enough data
            raw_data = raw_data + [0] * (128 - len(raw_data))
        
        # Apply Hanning window to reduce spectral leakage
        window = np.hanning(len(raw_data))
        windowed_data = raw_data * window
        
        # Perform FFT
        fft_result = np.fft.rfft(windowed_data)
        fft_freqs = np.fft.rfftfreq(len(raw_data), d=1.0/256)  # Assuming 256 Hz sampling rate
        
        # Extract power at target frequencies
        powers = np.zeros(3)
        for i, target_freq in enumerate(self.target_frequencies):
            # Find closest frequency bin
            idx = np.argmin(np.abs(fft_freqs - target_freq))
            # Get power (magnitude squared)
            powers[i] = np.abs(fft_result[idx]) ** 2
            
        # Normalize powers
        if np.sum(powers) > 0:
            powers = powers / np.sum(powers)
            
        return powers
    
    def _update_state_vector(self, frequency_powers: np.ndarray) -> None:
        """Update the 11D state vector based on frequency powers"""
        # Calculate time elapsed since last update
        now = time.time()
        dt = now - self.last_update
        self.last_update = now
        
        # Generate update vector using calibration matrix
        update = np.matmul(self.calibration_matrix, frequency_powers)
        
        # Apply noise reduction and update
        self.state_vector = (self.state_vector * self.noise_reduction + 
                            update * (1 - self.noise_reduction))
        
        # Ensure values stay within reasonable bounds
        self.state_vector = np.clip(self.state_vector, -1.0, 1.0)
    
    def _apply_time_compression(self) -> np.ndarray:
        """Apply time compression to the state vector"""
        # The compression rate affects how quickly changes in the state vector
        # are processed and projected forward in time
        compressed = np.tanh(self.state_vector * self.compression_rate)
        return compressed
    
    def _apply_evolution(self) -> None:
        """Apply evolution (adaptive learning) to the calibration matrix"""
        # Only evolve if we have enough history
        if len(self.history_buffer) < 10:
            return
            
        # Calculate average rate of change in each dimension
        recent_vectors = np.array(self.history_buffer[-10:])
        avg_change = np.std(recent_vectors, axis=0)
        
        # Adjust calibration matrix based on changes
        evolution_factor = self.evolution_rate * 0.1
        for i in range(11):
            # Dimensions with higher variance get strengthened
            # Dimensions with lower variance get weakened
            self.calibration_matrix[i, :] *= (1.0 + evolution_factor * (avg_change[i] - 0.5))
            
        # Renormalize calibration matrix
        for i in range(11):
            norm = np.linalg.norm(self.calibration_matrix[i, :])
            if norm > 0:
                self.calibration_matrix[i, :] /= norm
    
    def get_brain_activity_type(self) -> BrainActivityType:
        """Determine the dominant type of brain activity from the state vector"""
        # Each dimension contributes to detecting different brain activity types
        activity_scores = np.zeros(len(BrainActivityType))
        
        # Spatial dimensions (X,Y,Z) contribute to SPATIAL activity
        spatial_score = np.mean(np.abs(self.state_vector[0:3]))
        activity_scores[BrainActivityType.SPATIAL.value] = spatial_score
        
        # Temporal dimension contributes to PLANNING
        activity_scores[BrainActivityType.PLANNING.value] = np.abs(self.state_vector[3])
        
        # Intentionality dimension contributes to DECISION
        activity_scores[BrainActivityType.DECISION.value] = np.abs(self.state_vector[4])
        
        # Conceptual dimension contributes to CREATIVITY and ANALYTICAL
        activity_scores[BrainActivityType.CREATIVITY.value] = max(0, self.state_vector[5])
        activity_scores[BrainActivityType.ANALYTICAL.value] = max(0, -self.state_vector[5])
        
        # Contextual dimension contributes to MEMORY
        activity_scores[BrainActivityType.MEMORY.value] = np.abs(self.state_vector[6])
        
        # Entropy dimension contributes to CURIOSITY
        activity_scores[BrainActivityType.CURIOSITY.value] = np.abs(self.state_vector[7])
        
        # Emotional dimension contributes to EMOTIONAL
        activity_scores[BrainActivityType.EMOTIONAL.value] = np.abs(self.state_vector[8])
        
        # Memory dimension contributes to MEMORY
        activity_scores[BrainActivityType.MEMORY.value] += np.abs(self.state_vector[9])
        
        # Prediction dimension contributes to PLANNING and DECISION
        activity_scores[BrainActivityType.PLANNING.value] += max(0, self.state_vector[10])
        activity_scores[BrainActivityType.DECISION.value] += max(0, -self.state_vector[10])
        
        # Social activity is a combination of emotional and contextual
        activity_scores[BrainActivityType.SOCIAL.value] = (
            activity_scores[BrainActivityType.EMOTIONAL.value] * 0.5 +
            activity_scores[BrainActivityType.CONTEXTUAL.value] * 0.5
        )
        
        # Linguistic activity is a combination of several dimensions
        activity_scores[BrainActivityType.LINGUISTIC.value] = (
            np.abs(self.state_vector[5]) * 0.4 +  # Conceptual
            np.abs(self.state_vector[9]) * 0.3 +  # Memory
            np.abs(self.state_vector[6]) * 0.3    # Contextual
        )
        
        # Return the activity type with the highest score
        return BrainActivityType(np.argmax(activity_scores))
    
    def get_emotional_valence(self) -> float:
        """Get the emotional valence (positive/negative) from -1.0 to 1.0"""
        return self.state_vector[NeuralDimension.EMOTIONAL.value]
    
    def get_decision_confidence(self) -> float:
        """Get the confidence level of decisions from 0.0 to 1.0"""
        # Combine intentionality and negative entropy
        intentionality = np.abs(self.state_vector[NeuralDimension.INTENTIONALITY.value])
        entropy = np.abs(self.state_vector[NeuralDimension.ENTROPY.value])
        
        # Higher intentionality and lower entropy mean higher confidence
        return intentionality * (1.0 - entropy)
    
    def get_social_orientation(self) -> float:
        """Get social orientation from -1.0 (individualistic) to 1.0 (collectivistic)"""
        # Combine emotional and prediction dimensions
        return (self.state_vector[NeuralDimension.EMOTIONAL.value] * 0.6 + 
                self.state_vector[NeuralDimension.PREDICTION.value] * 0.4)
    
    def simulate_eeg_data(self, activity_bias: BrainActivityType = None) -> List[float]:
        """Generate simulated EEG data with optional bias toward a specific activity type"""
        # Generate 128 samples (0.5 seconds at 256 Hz)
        samples = []
        
        # Base frequencies (alpha, beta, theta, delta ranges)
        base_freqs = [
            10,   # Alpha
            20,   # Beta
            5,    # Theta
            2     # Delta
        ]
        
        # Add our target frequencies
        all_freqs = base_freqs + self.target_frequencies
        
        # Amplitudes for each frequency
        amplitudes = [0.5, 0.3, 0.4, 0.2, 0.8, 0.7, 0.9]
        
        # If activity_bias is provided, adjust the amplitudes of target frequencies
        if activity_bias is not None:
            if activity_bias == BrainActivityType.CREATIVITY:
                # Boost first target frequency
                amplitudes[4] *= 1.5
            elif activity_bias == BrainActivityType.ANALYTICAL:
                # Boost second target frequency
                amplitudes[5] *= 1.5
            elif activity_bias == BrainActivityType.SOCIAL:
                # Boost third target frequency
                amplitudes[6] *= 1.5
            elif activity_bias == BrainActivityType.DECISION:
                # Boost all target frequencies
                amplitudes[4:] = [a * 1.3 for a in amplitudes[4:]]
                
        # Generate samples as sum of sinusoids
        for i in range(128):
            t = i / 256.0  # Time in seconds
            sample = 0
            
            # Sum all frequency components
            for j, freq in enumerate(all_freqs):
                # Convert to radians per sample
                omega = 2 * np.pi * freq
                sample += amplitudes[j] * np.sin(omega * t)
                
            # Add some noise
            sample += (random.random() - 0.5) * 0.2
            
            samples.append(sample)
            
        return samples

class NeuroSynthixBridge:
    """Core class that bridges the NeuroQuantum Interface with SYNTHIX Governance"""
    
    def __init__(self, universe_id: str, user_agent_id: str = None):
        """Initialize the bridge between neural signals and the AI society"""
        self.universe_id = universe_id
        self.eeg_processor = EEGProcessor()
        self.governance_system = GovernanceSystem(universe_id)
        
        # Load governance system or create if it doesn't exist
        if not self.governance_system.load():
            logger.info(f"Creating new governance system for universe {universe_id}")
            self.governance_system.save()
            
        # Create or identify user agent
        if user_agent_id is None:
            user_agent_id = f"neuro_user_{str(uuid.uuid4())[:8]}"
        self.user_agent_id = user_agent_id
        
        # Ensure user agent exists in the governance system
        if user_agent_id not in self.governance_system.social_graph:
            logger.info(f"Adding neural user agent {user_agent_id} to governance system")
            
            # Add to social graph
            self.governance_system.social_graph[user_agent_id] = set()
            
            # Initialize agent resources
            for resource_type in ResourceType:
                # Give initial resources
                initial_amount = 20.0  # More than standard agents
                self.governance_system.economic_system.produce_resource(
                    user_agent_id, resource_type, initial_amount
                )
            
            # Initialize centrality
            self.governance_system.centrality_scores[user_agent_id] = 0.5
            
            # Save the updated governance system
            self.governance_system.save()
            
        # Mapping from neural dimensions to governance concepts
        self.dimension_to_resource = {
            NeuralDimension.INTENTIONALITY: ResourceType.ENERGY,
            NeuralDimension.CONCEPTUAL: ResourceType.KNOWLEDGE,
            NeuralDimension.CONTEXTUAL: ResourceType.SOCIAL_CAPITAL,
            NeuralDimension.ENTROPY: ResourceType.CREATIVITY,
            NeuralDimension.EMOTIONAL: ResourceType.TRUST,
            NeuralDimension.MEMORY: ResourceType.MEMORY,
            NeuralDimension.PREDICTION: ResourceType.PROBLEM_SOLVING
        }
        
        # Integration state
        self.integration_active = False
        self.integration_thread = None
        self.last_interaction_time = 0
        self.last_resource_generation_time = 0
        self.last_governance_influence_time = 0
        self.current_state_vector = np.zeros(11)
        self.thought_history = []
        
        # Agent preferences derived from neural patterns
        self.agent_preferences = {
            'governance_preference': 0.5,  # 0 = anarchic, 1 = structured
            'social_preference': 0.5,      # 0 = individualistic, 1 = collectivistic
            'economic_preference': 0.5,    # 0 = equal distribution, 1 = merit-based
            'cultural_preference': 0.5     # 0 = traditional, 1 = progressive
        }
        
        # Create base path for storage
        self.base_path = f"/var/lib/neurosynthix/{universe_id}"
        os.makedirs(self.base_path, exist_ok=True)
    
    def start(self) -> bool:
        """Start the neural-governance integration"""
        if self.integration_active:
            logger.warning("Neural integration already running")
            return False
            
        logger.info(f"Starting neural integration for universe {self.universe_id}")
        self.integration_active = True
        
        # Start integration thread
        self.integration_thread = threading.Thread(target=self._integration_loop)
        self.integration_thread.daemon = True
        self.integration_thread.start()
        
        return True
    
    def stop(self) -> bool:
        """Stop the neural-governance integration"""
        if not self.integration_active:
            logger.warning("Neural integration not running")
            return False
            
        logger.info(f"Stopping neural integration for universe {self.universe_id}")
        self.integration_active = False
        
        # Wait for integration thread to end
        if self.integration_thread and self.integration_thread.is_alive():
            self.integration_thread.join(timeout=5.0)
            
        return True
    
    def _integration_loop(self) -> None:
        """Main processing loop for neural-governance integration"""
        logger.info("Neural integration loop started")
        
        while self.integration_active:
            try:
                # 1. Get simulated EEG data (would be real EEG in production)
                eeg_data = self.eeg_processor.simulate_eeg_data()
                
                # 2. Process EEG data into 11D state vector
                state_vector = self.eeg_processor.receive_eeg_data(eeg_data)
                self.current_state_vector = state_vector
                
                # 3. Update thought history
                self.thought_history.append({
                    'timestamp': time.time(),
                    'state_vector': state_vector.tolist(),
                    'brain_activity': self.eeg_processor.get_brain_activity_type().name
                })
                
                # Keep history limited
                if len(self.thought_history) > 1000:
                    self.thought_history = self.thought_history[-1000:]
                
                # 4. Process neural-governance interactions
                current_time = time.time()
                
                # Generate social interactions every 2 seconds
                if current_time - self.last_interaction_time > 2:
                    self._process_social_interactions()
                    self.last_interaction_time = current_time
                    
                # Generate resources every 5 seconds
                if current_time - self.last_resource_generation_time > 5:
                    self._process_resource_generation()
                    self.last_resource_generation_time = current_time
                
                # Influence governance every 15 seconds
                if current_time - self.last_governance_influence_time > 15:
                    self._process_governance_influence()
                    self.last_governance_influence_time = current_time
                    
                # Update preferences based on neural patterns
                self._update_agent_preferences()
                
                # Sleep to prevent excessive CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in neural integration loop: {e}")
                time.sleep(1)
                
        logger.info("Neural integration loop ended")
    
    def _process_social_interactions(self) -> None:
        """Process social interactions based on neural patterns"""
        # Get the current brain activity type
        activity_type = self.eeg_processor.get_brain_activity_type()
        
        # Only generate social interactions for certain activity types
        social_activity_types = [
            BrainActivityType.SOCIAL,
            BrainActivityType.EMOTIONAL,
            BrainActivityType.LINGUISTIC,
            BrainActivityType.CURIOSITY
        ]
        
        if activity_type in social_activity_types and self.governance_system.social_graph:
            # Calculate interaction probability based on social orientation
            social_orientation = self.eeg_processor.get_social_orientation()
            interaction_probability = 0.3 + 0.4 * (social_orientation + 1) / 2  # 0.3 to 0.7
            
            if random.random() < interaction_probability:
                # Find potential interaction partners (other agents)
                other_agents = [agent_id for agent_id in self.governance_system.social_graph.keys()
                               if agent_id != self.user_agent_id]
                
                if not other_agents:
                    return
                    
                # Select interaction partner
                if self.user_agent_id in self.governance_system.social_graph and self.governance_system.social_graph[self.user_agent_id]:
                    # 70% chance to interact with existing connection
                    if random.random() < 0.7 and self.governance_system.social_graph[self.user_agent_id]:
                        partner_id = random.choice(list(self.governance_system.social_graph[self.user_agent_id]))
                    else:
                        # 30% chance to interact with new agent
                        new_agents = [a for a in other_agents if a not in self.governance_system.social_graph[self.user_agent_id]]
                        if new_agents:
                            partner_id = random.choice(new_agents)
                        else:
                            partner_id = random.choice(other_agents)
                else:
                    # No existing connections, pick random agent
                    partner_id = random.choice(other_agents)
                
                # Determine interaction type based on brain activity
                interaction_types = {
                    BrainActivityType.SOCIAL: ["CONVERSATION", "COLLABORATION"],
                    BrainActivityType.EMOTIONAL: ["EMPATHY", "SUPPORT"],
                    BrainActivityType.LINGUISTIC: ["TEACHING", "LEARNING"],
                    BrainActivityType.CURIOSITY: ["EXPLORATION", "INQUIRY"]
                }
                
                interaction_type = random.choice(interaction_types.get(activity_type, ["CONVERSATION"]))
                
                # Determine interaction outcome based on emotional valence
                emotional_valence = self.eeg_processor.get_emotional_valence()
                interaction_outcome = 0.2 + 0.6 * (emotional_valence + 1) / 2  # 0.2 to 0.8
                
                # Apply noise to the outcome
                interaction_outcome += (random.random() - 0.5) * 0.3
                interaction_outcome = max(-1.0, min(1.0, interaction_outcome))
                
                # Update the relationship
                relationship = self.governance_system.update_relationship(
                    self.user_agent_id, partner_id, interaction_type, interaction_outcome
                )
                
                logger.debug(f"Neural-induced interaction: {self.user_agent_id} <-> {partner_id}, " +
                           f"Type: {interaction_type}, Outcome: {interaction_outcome:.2f}")
                
                # Process economic interaction if appropriate
                if interaction_type == "COLLABORATION" and interaction_outcome > 0.5:
                    # Random resource type for collaboration
                    resource_type = random.choice(list(ResourceType))
                    
                    # Both agents produce resources
                    amount = 1.0 + random.random() * 2.0
                    self.governance_system.economic_system.produce_resource(
                        self.user_agent_id, resource_type, amount
                    )
                    self.governance_system.economic_system.produce_resource(
                        partner_id, resource_type, amount * 0.8
                    )
                    
                    logger.debug(f"Collaborative resource production: {resource_type.name}, " +
                               f"Amount: {amount:.2f}")
    
    def _process_resource_generation(self) -> None:
        """Generate resources based on neural activity"""
        # Get the relevant dimensions for resource production
        for dimension, resource_type in self.dimension_to_resource.items():
            # Get the dimension value (-1.0 to 1.0)
            dimension_value = self.current_state_vector[dimension.value]
            
            # Only positive values generate resources
            if dimension_value > 0:
                # Calculate production amount based on dimension value
                production_amount = dimension_value * random.uniform(0.5, 2.0)
                
                # Produce the resource
                if production_amount > 0:
                    self.governance_system.economic_system.produce_resource(
                        self.user_agent_id, resource_type, production_amount
                    )
                    
                    logger.debug(f"Neural resource generation: {resource_type.name}, " +
                               f"Amount: {production_amount:.2f}")
    
    def _process_governance_influence(self) -> None:
        """Influence governance based on neural patterns"""
        # Get decision confidence and brain activity type
        decision_confidence = self.eeg_processor.get_decision_confidence()
        activity_type = self.eeg_processor.get_brain_activity_type()
        
        # Only influence governance for certain activity types with sufficient confidence
        governance_activity_types = [
            BrainActivityType.DECISION,
            BrainActivityType.PLANNING,
            BrainActivityType.ANALYTICAL
        ]
        
        if activity_type in governance_activity_types and decision_confidence > 0.6:
            # Determine influence type based on current preferences
            if random.random() < self.agent_preferences['governance_preference']:
                # Create or modify policy
                self._influence_policy()
            else:
                # Create or join group
                self._influence_group()
    
    def _influence_policy(self) -> None:
        """Create or vote on a policy based on neural state"""
        # Decide whether to create a new policy or vote on existing ones
        if random.random() < 0.3 or not self.governance_system.policies:
            # Create new policy
            policy_title = f"Neural Policy {len(self.governance_system.policies) + 1}"
            
            # Determine policy description based on brain activity
            activity_type = self.eeg_processor.get_brain_activity_type()
            economic_preference = self.agent_preferences['economic_preference']
            social_preference = self.agent_preferences['social_preference']
            
            policy_descriptions = {
                BrainActivityType.DECISION: f"Resource allocation policy (Equality bias: {1-economic_preference:.2f})",
                BrainActivityType.PLANNING: f"Coordination protocol (Collectivism factor: {social_preference:.2f})",
                BrainActivityType.ANALYTICAL: f"Optimization directive (Efficiency factor: {economic_preference:.2f})"
            }
            
            policy_description = policy_descriptions.get(
                activity_type, 
                f"General policy (Governance structure: {self.agent_preferences['governance_preference']:.2f})"
            )
            
            # Create the policy
            policy = self.governance_system.create_policy(
                policy_title, policy_description, self.user_agent_id
            )
            
            logger.info(f"Created neural policy: {policy_title}")
            
        elif self.governance_system.policies:
            # Vote on existing policy
            policy_id = random.choice(list(self.governance_system.policies.keys()))
            policy = self.governance_system.policies[policy_id]
            
            # Determine vote based on preference alignment
            emotional_valence = self.eeg_processor.get_emotional_valence()
            governance_preference = self.agent_preferences['governance_preference']
            
            # Positive valence and high governance preference tends toward supporting
            support_probability = (emotional_valence + 1) / 2 * governance_preference
            support = random.random() < support_probability
            
            # Register vote
            policy.vote(self.user_agent_id, support)
            
            logger.debug(f"Voted on policy {policy.title}: {'Support' if support else 'Oppose'}")
            
            # Activate policy if it has sufficient support
            if policy.status == "proposed" and policy.calculate_support() > 0.6:
                policy.activate()
                logger.info(f"Activated policy {policy.title} with {policy.calculate_support():.2f} support")
    
    def _influence_group(self) -> None:
        """Create or join social groups based on neural state"""
        # Decide whether to create a new group or join existing ones
        if random.random() < 0.2 or not self.governance_system.social_groups:
            # Create new group
            group_name = f"Neural Collective {len(self.governance_system.social_groups) + 1}"
            
            # Determine group description based on preferences
            social_pref = self.agent_preferences['social_preference']
            cultural_pref = self.agent_preferences['cultural_preference']
            
            group_description = f"A collective formed from neural patterns (Social: {social_pref:.2f}, Cultural: {cultural_pref:.2f})"
            
            # Create the group
            group = self.governance_system.create_group(
                group_name, group_description, self.user_agent_id
            )
            
            logger.info(f"Created neural group: {group_name}")
            
        elif self.governance_system.social_groups:
            # Check if already member of any groups
            member_groups = [g_id for g_id, group in self.governance_system.social_groups.items() 
                            if self.user_agent_id in group.members]
            
            if not member_groups:
                # Join a random group
                group_id = random.choice(list(self.governance_system.social_groups.keys()))
                group = self.governance_system.social_groups[group_id]
                
                group.add_member(self.user_agent_id)
                logger.info(f"Joined group: {group.name}")
            else:
                # Already in groups, possibly create a cultural meme
                self._create_cultural_meme()
    
    def _create_cultural_meme(self) -> None:
        """Create a cultural meme based on neural state"""
        if random.random() < 0.4:  # 40% chance to create a meme
            # Get current brain activity type and state
            activity_type = self.eeg_processor.get_brain_activity_type()
            
            # Creativity and emotional states are most conducive to meme creation
            if activity_type in [BrainActivityType.CREATIVITY, BrainActivityType.EMOTIONAL, 
                                BrainActivityType.SOCIAL]:
                
                # Generate meme name based on activity type
                meme_names = {
                    BrainActivityType.CREATIVITY: [
                        "Quantum Thought Pattern", "Neural Innovation Wave", 
                        "Creative Dimension Shift", "Ideation Resonance"
                    ],
                    BrainActivityType.EMOTIONAL: [
                        "Emotional Harmonic", "Valence Oscillation", 
                        "Sentiment Cascade", "Empathy Field"
                    ],
                    BrainActivityType.SOCIAL: [
                        "Social Network Pulse", "Collective Convergence", 
                        "Community Resonance", "Group Harmony Pattern"
                    ]
                }
                
                default_names = ["Neural Meme", "Thought Pattern", "Cognitive Signature"]
                meme_name = random.choice(meme_names.get(activity_type, default_names))
                
                # Generate meme description using state vector
                emotional_valence = self.eeg_processor.get_emotional_valence()
                social_orientation = self.eeg_processor.get_social_orientation()
                
                valence_descriptor = "positive" if emotional_valence > 0 else "negative"
                social_descriptor = "collectivistic" if social_orientation > 0 else "individualistic"
                
                meme_description = f"A neural-originated cultural pattern with {valence_descriptor} emotional valence " + \
                                 f"and {social_descriptor} social orientation. Created through 11-dimensional " + \
                                 f"brain mapping at frequencies 98.7/99.1/98.9 Hz."
                
                # Generate meme attributes based on state vector
                meme_attributes = {
                    "emotional_valence": emotional_valence,
                    "social_orientation": social_orientation,
                    "complexity": abs(self.current_state_vector[NeuralDimension.ENTROPY.value]),
                    "novelty": abs(self.current_state_vector[NeuralDimension.CONCEPTUAL.value]),
                    "applicability": abs(self.current_state_vector[NeuralDimension.CONTEXTUAL.value]),
                    "memorability": abs(self.current_state_vector[NeuralDimension.MEMORY.value]),
                    "evolution_rate": self.eeg_processor.evolution_rate
                }
                
                # Create the meme
                meme = self.governance_system.create_meme(
                    meme_name, meme_description, self.user_agent_id, meme_attributes
                )
                
                logger.info(f"Created neural cultural meme: {meme_name}")
                
                # Try to spread the meme to some connections
                if self.user_agent_id in self.governance_system.social_graph:
                    connections = list(self.governance_system.social_graph[self.user_agent_id])
                    
                    # Select up to 3 connections to spread to
                    if connections:
                        spread_targets = random.sample(connections, min(3, len(connections)))
                        
                        for target_id in spread_targets:
                            # Spread probability based on meme attributes
                            spread_prob = 0.3 + 0.4 * meme_attributes["memorability"]
                            
                            # Attempt to spread
                            success = self.governance_system.spread_meme(
                                meme.meme_id, self.user_agent_id, target_id, spread_prob
                            )
                            
                            if success:
                                logger.debug(f"Spread meme {meme_name} to agent {target_id}")
    
    def _update_agent_preferences(self) -> None:
        """Update agent preferences based on neural patterns"""
        # Extract relevant metrics from state vector
        governance_signal = self.current_state_vector[NeuralDimension.PREDICTION.value]
        social_signal = self.current_state_vector[NeuralDimension.EMOTIONAL.value]
        economic_signal = self.current_state_vector[NeuralDimension.INTENTIONALITY.value]
        cultural_signal = self.current_state_vector[NeuralDimension.CONCEPTUAL.value]
        
        # Transform signals to 0-1 range
        governance_value = (governance_signal + 1) / 2
        social_value = (social_signal + 1) / 2
        economic_value = (economic_signal + 1) / 2
        cultural_value = (cultural_signal + 1) / 2
        
        # Update preferences with exponential moving average
        alpha = 0.05  # Small alpha for stable preferences
        self.agent_preferences['governance_preference'] = (
            (1 - alpha) * self.agent_preferences['governance_preference'] + 
            alpha * governance_value
        )
        
        self.agent_preferences['social_preference'] = (
            (1 - alpha) * self.agent_preferences['social_preference'] + 
            alpha * social_value
        )
        
        self.agent_preferences['economic_preference'] = (
            (1 - alpha) * self.agent_preferences['economic_preference'] + 
            alpha * economic_value
        )
        
        self.agent_preferences['cultural_preference'] = (
            (1 - alpha) * self.agent_preferences['cultural_preference'] + 
            alpha * cultural_value
        )
    
    def get_neural_governance_status(self) -> Dict[str, Any]:
        """Get comprehensive status about neural-governance integration"""
        status = {
            'user_agent_id': self.user_agent_id,
            'universe_id': self.universe_id,
            'integration_active': self.integration_active,
            'current_state_vector': self.current_state_vector.tolist(),
            'current_brain_activity': self.eeg_processor.get_brain_activity_type().name,
            'emotional_valence': self.eeg_processor.get_emotional_valence(),
            'decision_confidence': self.eeg_processor.get_decision_confidence(),
            'social_orientation': self.eeg_processor.get_social_orientation(),
            'agent_preferences': self.agent_preferences,
            'resource_holdings': {},
            'social_connections': [],
            'created_policies': [],
            'group_memberships': [],
            'created_memes': []
        }
        
        # Add resource holdings
        for resource_type in ResourceType:
            status['resource_holdings'][resource_type.name] = (
                self.governance_system.economic_system.agent_resources[self.user_agent_id][resource_type]
            )
        
        # Add social connections
        for key, rel in self.governance_system.social_relationships.items():
            if self.user_agent_id in key:
                other_id = key[0] if key[1] == self.user_agent_id else key[1]
                relationship_info = {
                    'agent_id': other_id,
                    'relationship_type': rel.relationship_type.name,
                    'strength': rel.strength,
                    'trust': rel.trust,
                    'interaction_count': rel.interaction_count
                }
                status['social_connections'].append(relationship_info)
        
        # Add created policies
        for policy_id, policy in self.governance_system.policies.items():
            if policy.creator_id == self.user_agent_id:
                policy_info = {
                    'policy_id': policy_id,
                    'title': policy.title,
                    'status': policy.status,
                    'support_level': policy.calculate_support()
                }
                status['created_policies'].append(policy_info)
        
        # Add group memberships
        for group_id, group in self.governance_system.social_groups.items():
            if self.user_agent_id in group.members:
                group_info = {
                    'group_id': group_id,
                    'name': group.name,
                    'role': group.members[self.user_agent_id],
                    'is_leader': group.leader_id == self.user_agent_id,
                    'member_count': len(group.members)
                }
                status['group_memberships'].append(group_info)
        
        # Add created memes
        for meme_id, meme in self.governance_system.cultural_memes.items():
            if meme.creator_id == self.user_agent_id:
                meme_info = {
                    'meme_id': meme_id,
                    'name': meme.name,
                    'adopter_count': len(meme.adopters),
                    'spread_rate': meme.spread_rate
                }
                status['created_memes'].append(meme_info)
        
        return status
    
    def get_society_impact(self) -> Dict[str, Any]:
        """Analyze the impact of neural integration on the AI society"""
        # Get society health metrics
        society_health = self.governance_system.get_society_health()
        
        # Calculate the neural user's social influence
        user_social_status = self.governance_system.get_agent_social_status(self.user_agent_id)
        
        # Calculate neural-originated content
        neural_policies = sum(1 for p in self.governance_system.policies.values() 
                             if p.creator_id == self.user_agent_id)
        neural_groups = sum(1 for g in self.governance_system.social_groups.values() 
                           if g.leader_id == self.user_agent_id)
        neural_memes = sum(1 for m in self.governance_system.cultural_memes.values() 
                          if m.creator_id == self.user_agent_id)
        
        # Calculate adoption metrics
        neural_meme_adoption = 0
        total_agents = len(self.governance_system.social_graph)
        
        if total_agents > 1:  # Exclude the neural user
            for meme in self.governance_system.cultural_memes.values():
                if meme.creator_id == self.user_agent_id:
                    # Calculate percentage of society that adopted this meme
                    adoption_rate = (len(meme.adopters) - 1) / (total_agents - 1)  # Exclude creator
                    neural_meme_adoption += adoption_rate
            
            # Average across all neural memes
            if neural_memes > 0:
                neural_meme_adoption /= neural_memes
        
        # Build impact report
        impact = {
            'neural_agent_influence': user_social_status['influence'],
            'neural_agent_status': user_social_status['status'],
            'neural_connections': len(self.governance_system.social_graph.get(self.user_agent_id, [])),
            'neural_resources_generated': sum(self.governance_system.economic_system.agent_resources[self.user_agent_id].values()),
            'neural_policies_created': neural_policies,
            'neural_groups_created': neural_groups,
            'neural_memes_created': neural_memes,
            'neural_meme_adoption_rate': neural_meme_adoption,
            'society_metrics': {
                'social_cohesion': society_health['social_cohesion'],
                'governance_effectiveness': society_health['governance_effectiveness'],
                'cultural_diversity': society_health['cultural_diversity'],
                'economic_health': society_health['economic_health'],
                'overall_health': society_health['overall_health']
            }
        }
        
        # Calculate neural impact on society
        if total_agents > 1:
            impact['neural_influence_percentage'] = user_social_status['influence'] / sum(
                self.governance_system.get_agent_social_status(agent_id)['influence']
                for agent_id in self.governance_system.social_graph
                if agent_id != self.user_agent_id
            ) * 100
        else:
            impact['neural_influence_percentage'] = 100
        
        return impact
    
    def save_state(self) -> bool:
        """Save the current state of the neural-governance bridge"""
        try:
            # Save governance system
            self.governance_system.save()
            
            # Save neural integration state
            integration_state = {
                'user_agent_id': self.user_agent_id,
                'agent_preferences': self.agent_preferences,
                'thought_history': self.thought_history[-100:],  # Last 100 thoughts only
                'eeg_calibration_matrix': self.eeg_processor.calibration_matrix.tolist()
            }
            
            with open(f"{self.base_path}/neural_integration.json", 'w') as f:
                json.dump(integration_state, f, indent=2)
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to save neural integration state: {e}")
            return False
    
    def load_state(self) -> bool:
        """Load the saved state of the neural-governance bridge"""
        try:
            # Load governance system
            if not self.governance_system.load():
                logger.warning("Failed to load governance system")
                return False
                
            # Load neural integration state
            state_path = f"{self.base_path}/neural_integration.json"
            if not os.path.exists(state_path):
                logger.warning("No saved neural integration state found")
                return False
                
            with open(state_path, 'r') as f:
                integration_state = json.load(f)
                
            self.user_agent_id = integration_state.get('user_agent_id', self.user_agent_id)
            self.agent_preferences = integration_state.get('agent_preferences', self.agent_preferences)
            self.thought_history = integration_state.get('thought_history', [])
            
            # Load calibration matrix if it exists
            if 'eeg_calibration_matrix' in integration_state:
                self.eeg_processor.calibration_matrix = np.array(integration_state['eeg_calibration_matrix'])
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to load neural integration state: {e}")
            return False

    def generate_simulation_data(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """Generate a simulation of neural-governance interaction over time"""
        logger.info(f"Starting neural-governance simulation for {duration_seconds} seconds")
        
        # Store simulation data
        simulation_data = {
            'timestamps': [],
            'brain_activity_types': [],
            'state_vectors': [],
            'social_interactions': [],
            'resource_generations': [],
            'governance_influences': [],
            'society_metrics': []
        }
        
        # Record starting society metrics
        society_health_start = self.governance_system.get_society_health()
        
        # Start simulation time
        start_time = time.time()
        last_interaction_time = 0
        last_resource_time = 0
        last_governance_time = 0
        last_metrics_time = 0
        
        # Run simulation for specified duration
        while time.time() - start_time < duration_seconds:
            # Generate simulated EEG data with bias toward specific activity types
            activity_bias = random.choice(list(BrainActivityType))
            eeg_data = self.eeg_processor.simulate_eeg_data(activity_bias)
            
            # Process EEG data
            state_vector = self.eeg_processor.receive_eeg_data(eeg_data)
            brain_activity = self.eeg_processor.get_brain_activity_type()
            
            # Record basic data
            current_time = time.time() - start_time
            simulation_data['timestamps'].append(current_time)
            simulation_data['brain_activity_types'].append(brain_activity.name)
            simulation_data['state_vectors'].append(state_vector.tolist())
            
            # Generate social interactions
            if current_time - last_interaction_time > 2:
                interaction = self._simulate_interaction(brain_activity)
                if interaction:
                    simulation_data['social_interactions'].append(interaction)
                last_interaction_time = current_time
            
            # Generate resources
            if current_time - last_resource_time > 5:
                resources = self._simulate_resource_generation(state_vector)
                if resources:
                    simulation_data['resource_generations'].append(resources)
                last_resource_time = current_time
            
            # Influence governance
            if current_time - last_governance_time > 15:
                influence = self._simulate_governance_influence(brain_activity, state_vector)
                if influence:
                    simulation_data['governance_influences'].append(influence)
                last_governance_time = current_time
            
            # Record society metrics
            if current_time - last_metrics_time > 10:
                metrics = {
                    'timestamp': current_time,
                    'metrics': self.governance_system.get_society_health()
                }
                simulation_data['society_metrics'].append(metrics)
                last_metrics_time = current_time
            
            # Sleep a bit to prevent excessive CPU usage
            time.sleep(0.1)
        
        # Record ending society metrics
        society_health_end = self.governance_system.get_society_health()
        
        # Calculate changes in society metrics
        simulation_data['society_changes'] = {
            'social_cohesion_change': society_health_end['social_cohesion'] - society_health_start['social_cohesion'],
            'governance_effectiveness_change': society_health_end['governance_effectiveness'] - society_health_start['governance_effectiveness'],
            'cultural_diversity_change': society_health_end['cultural_diversity'] - society_health_start['cultural_diversity'],
            'economic_health_change': society_health_end['economic_health'] - society_health_start['economic_health'],
            'overall_health_change': society_health_end['overall_health'] - society_health_start['overall_health']
        }
        
        logger.info(f"Completed neural-governance simulation with {len(simulation_data['timestamps'])} data points")
        
        return simulation_data
    
    def _simulate_interaction(self, brain_activity: BrainActivityType) -> Dict[str, Any]:
        """Simulate a social interaction based on brain activity"""
        # Only generate social interactions for certain activity types
        social_activity_types = [
            BrainActivityType.SOCIAL,
            BrainActivityType.EMOTIONAL,
            BrainActivityType.LINGUISTIC,
            BrainActivityType.CURIOSITY
        ]
        
        if brain_activity not in social_activity_types:
            return None
            
        # Get other agents (assume at least one other agent)
        other_agents = [agent_id for agent_id in self.governance_system.social_graph.keys()
                       if agent_id != self.user_agent_id]
        
        if not other_agents:
            # Create a simulated agent if none exist
            agent_id = f"sim_agent_{str(uuid.uuid4())[:8]}"
            self.governance_system.social_graph[agent_id] = set()
            other_agents = [agent_id]
            
        # Select a partner
        partner_id = random.choice(other_agents)
        
        # Determine interaction type based on brain activity
        interaction_types = {
            BrainActivityType.SOCIAL: ["CONVERSATION", "COLLABORATION"],
            BrainActivityType.EMOTIONAL: ["EMPATHY", "SUPPORT"],
            BrainActivityType.LINGUISTIC: ["TEACHING", "LEARNING"],
            BrainActivityType.CURIOSITY: ["EXPLORATION", "INQUIRY"]
        }
        
        interaction_type = random.choice(interaction_types.get(brain_activity, ["CONVERSATION"]))
        
        # Determine interaction outcome (-1.0 to 1.0)
        interaction_outcome = random.random() * 2 - 1.0
        
        # Update the relationship
        relationship = self.governance_system.update_relationship(
            self.user_agent_id, partner_id, interaction_type, interaction_outcome
        )
        
        # Return interaction data
        return {
            'timestamp': time.time(),
            'partner_id': partner_id,
            'interaction_type': interaction_type,
            'outcome': interaction_outcome,
            'brain_activity': brain_activity.name
        }
    
    def _simulate_resource_generation(self, state_vector: np.ndarray) -> Dict[str, Any]:
        """Simulate resource generation based on state vector"""
        # Get the relevant dimensions for resource production
        resources_generated = {}
        
        for dimension, resource_type in self.dimension_to_resource.items():
            # Get the dimension value (-1.0 to 1.0)
            dimension_value = state_vector[dimension.value]
            
            # Only positive values generate resources
            if dimension_value > 0:
                # Calculate production amount based on dimension value
                production_amount = dimension_value * random.uniform(0.5, 2.0)
                
                # Produce the resource
                if production_amount > 0:
                    self.governance_system.economic_system.produce_resource(
                        self.user_agent_id, resource_type, production_amount
                    )
                    
                    resources_generated[resource_type.name] = production_amount
        
        if not resources_generated:
            return None
            
        # Return resource generation data
        return {
            'timestamp': time.time(),
            'resources': resources_generated,
            'total_value': sum(resources_generated.values())
        }
    
    def _simulate_governance_influence(self, brain_activity: BrainActivityType, 
                                      state_vector: np.ndarray) -> Dict[str, Any]:
        """Simulate governance influence based on brain activity and state vector"""
        # Only influence governance for certain activity types
        governance_activity_types = [
            BrainActivityType.DECISION,
            BrainActivityType.PLANNING,
            BrainActivityType.ANALYTICAL
        ]
        
        if brain_activity not in governance_activity_types:
            return None
            
        # Determine influence type
        influence_types = ["POLICY", "GROUP", "MEME"]
        influence_type = random.choice(influence_types)
        
        influence_data = {
            'timestamp': time.time(),
            'type': influence_type,
            'brain_activity': brain_activity.name
        }
        
        if influence_type == "POLICY":
            # Create a policy
            policy_title = f"Neural Policy {len(self.governance_system.policies) + 1}"
            policy_description = f"Policy derived from {brain_activity.name} neural pattern"
            
            policy = self.governance_system.create_policy(
                policy_title, policy_description, self.user_agent_id
            )
            
            influence_data['policy_id'] = policy.policy_id
            influence_data['policy_title'] = policy_title
            
        elif influence_type == "GROUP":
            # Create or join a group
            if not self.governance_system.social_groups or random.random() < 0.3:
                # Create new group
                group_name = f"Neural Collective {len(self.governance_system.social_groups) + 1}"
                group_description = f"A collective formed from {brain_activity.name} neural pattern"
                
                group = self.governance_system.create_group(
                    group_name, group_description, self.user_agent_id
                )
                
                influence_data['group_id'] = group.group_id
                influence_data['group_name'] = group_name
                influence_data['action'] = "CREATE"
                
            else:
                # Join existing group
                group_id = random.choice(list(self.governance_system.social_groups.keys()))
                group = self.governance_system.social_groups[group_id]
                
                if self.user_agent_id not in group.members:
                    group.add_member(self.user_agent_id)
                    influence_data['group_id'] = group.group_id
                    influence_data['group_name'] = group.name
                    influence_data['action'] = "JOIN"
                else:
                    influence_data['group_id'] = group.group_id
                    influence_data['group_name'] = group.name
                    influence_data['action'] = "ALREADY_MEMBER"
                    
        elif influence_type == "MEME":
            # Create a cultural meme
            meme_name = f"Neural Meme {len(self.governance_system.cultural_memes) + 1}"
            meme_description = f"Cultural pattern derived from {brain_activity.name} neural activity"
            
            meme_attributes = {
                "emotional_valence": state_vector[NeuralDimension.EMOTIONAL.value],
                "complexity": abs(state_vector[NeuralDimension.ENTROPY.value]),
                "novelty": abs(state_vector[NeuralDimension.CONCEPTUAL.value])
            }
            
            meme = self.governance_system.create_meme(
                meme_name, meme_description, self.user_agent_id, meme_attributes
            )
            
            influence_data['meme_id'] = meme.meme_id
            influence_data['meme_name'] = meme_name
            
        return influence_data


# CLI for the NeuroSYNTHIX Integration System
def main():
    """Main entry point for the NeuroSYNTHIX Integration System"""
    import argparse
    
    parser = argparse.ArgumentParser(description='NeuroSYNTHIX Integration System')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Initialize system
    init_parser = subparsers.add_parser('init', help='Initialize NeuroSYNTHIX system')
    init_parser.add_argument('--universe', required=True, help='Universe ID')
    init_parser.add_argument('--user-agent', help='User agent ID (optional)')
    
    # Start integration
    start_parser = subparsers.add_parser('start', help='Start NeuroSYNTHIX integration')
    start_parser.add_argument('--universe', required=True, help='Universe ID')
    
    # Stop integration
    stop_parser = subparsers.add_parser('stop', help='Stop NeuroSYNTHIX integration')
    stop_parser.add_argument('--universe', required=True, help='Universe ID')
    
    # Get status
    status_parser = subparsers.add_parser('status', help='Get NeuroSYNTHIX status')
    status_parser.add_argument('--universe', required=True, help='Universe ID')
    
    # Run simulation
    sim_parser = subparsers.add_parser('simulate', help='Run neural-governance simulation')
    sim_parser.add_argument('--universe', required=True, help='Universe ID')
    sim_parser.add_argument('--duration', type=int, default=60, help='Simulation duration in seconds')
    sim_parser.add_argument('--output', help='Output file for simulation data (JSON)')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == 'init':
        # Initialize system
        bridge = NeuroSynthixBridge(args.universe, args.user_agent)
        bridge.save_state()
        print(f"Initialized NeuroSYNTHIX system for universe {args.universe}")
        print(f"User agent ID: {bridge.user_agent_id}")
        
    elif args.command == 'start':
        # Start integration
        bridge = NeuroSynthixBridge(args.universe)
        if not bridge.load_state():
            print(f"Error: NeuroSYNTHIX system for universe {args.universe} not found")
            return
            
        if bridge.start():
            print(f"Started NeuroSYNTHIX integration for universe {args.universe}")
        else:
            print(f"Error: Failed to start NeuroSYNTHIX integration")
            
    elif args.command == 'stop':
        # Stop integration
        bridge = NeuroSynthixBridge(args.universe)
        if not bridge.load_state():
            print(f"Error: NeuroSYNTHIX system for universe {args.universe} not found")
            return
            
        if bridge.stop():
            print(f"Stopped NeuroSYNTHIX integration for universe {args.universe}")
            # Save state on stop
            bridge.save_state()
        else:
            print(f"Error: Failed to stop NeuroSYNTHIX integration")
            
    elif args.command == 'status':
        # Get status
        bridge = NeuroSynthixBridge(args.universe)
        if not bridge.load_state():
            print(f"Error: NeuroSYNTHIX system for universe {args.universe} not found")
            return
            
        status = bridge.get_neural_governance_status()
        impact = bridge.get_society_impact()
        
        print("=== NeuroSYNTHIX Status ===")
        print(f"Universe: {status['universe_id']}")
        print(f"User Agent: {status['user_agent_id']}")
        print(f"Integration Active: {status['integration_active']}")
        print(f"Current Brain Activity: {status['current_brain_activity']}")
        print(f"Emotional Valence: {status['emotional_valence']:.2f}")
        print(f"Decision Confidence: {status['decision_confidence']:.2f}")
        print(f"Social Orientation: {status['social_orientation']:.2f}")
        print("\n=== Agent Preferences ===")
        for pref, value in status['agent_preferences'].items():
            print(f"{pref}: {value:.2f}")
        print("\n=== Social Impact ===")
        print(f"Social Influence: {impact['neural_agent_influence']:.2f}")
        print(f"Society-Wide Influence: {impact['neural_influence_percentage']:.2f}%")
        print(f"Neural Policies: {impact['neural_policies_created']}")
        print(f"Neural Groups: {impact['neural_groups_created']}")
        print(f"Neural Memes: {impact['neural_memes_created']}")
        print(f"Meme Adoption Rate: {impact['neural_meme_adoption_rate'] * 100:.2f}%")
        print("\n=== Society Health ===")
        for metric, value in impact['society_metrics'].items():
            print(f"{metric}: {value:.2f}")
            
    elif args.command == 'simulate':
        # Run simulation
        bridge = NeuroSynthixBridge(args.universe)
        if not bridge.load_state():
            print(f"Error: NeuroSYNTHIX system for universe {args.universe} not found")
            return
            
        print(f"Running neural-governance simulation for {args.duration} seconds...")
        simulation_data = bridge.generate_simulation_data(args.duration)
        
        # Save state after simulation
        bridge.save_state()
        
        print("\n=== Simulation Results ===")
        print(f"Data points: {len(simulation_data['timestamps'])}")
        print(f"Social interactions: {len(simulation_data['social_interactions'])}")
        print(f"Resource generations: {len(simulation_data['resource_generations'])}")
        print(f"Governance influences: {len(simulation_data['governance_influences'])}")
        
        print("\n=== Society Changes ===")
        for metric, change in simulation_data['society_changes'].items():
            direction = "increased" if change > 0 else "decreased" if change < 0 else "unchanged"
            print(f"{metric}: {direction} by {abs(change):.4f}")
            
        # Save simulation data if output file specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(simulation_data, f, indent=2)
            print(f"\nSimulation data saved to {args.output}")
    
    else:
        print("Error: Unknown command")
        parser.print_help()

if __name__ == "__main__":
    main()
