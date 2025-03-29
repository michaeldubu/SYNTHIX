#!/usr/bin/env python3
"""
SYNTHIX Governance System

This module extends the SYNTHIX OS to provide civilization and governance
capabilities for AI agent societies, including:

1. Social relationships and group dynamics
2. Resource allocation and economic systems
3. Decision making and governance structures
4. Cultural evolution and knowledge transfer

The governance system builds on the existing Agent Runtime Environment (ARE)
and Universe Simulation Engine (USE) to create emergent social behaviors.
"""

import os
import sys
import json
import time
import random
import logging
import threading
import numpy as np
from typing import Dict, List, Any, Tuple, Set
from collections import defaultdict
import uuid
import math
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('/var/log/synthix/governance.log'), logging.StreamHandler()]
)
logger = logging.getLogger('GOVERNANCE')

class RelationshipType(Enum):
    """Types of relationships between agents"""
    NEUTRAL = 0
    FRIEND = 1
    ALLY = 2
    FAMILY = 3
    PARTNER = 4
    LEADER = 5
    FOLLOWER = 6
    RIVAL = 7
    ENEMY = 8
    TEACHER = 9
    STUDENT = 10

class GovernanceType(Enum):
    """Types of governance structures"""
    ANARCHY = 0
    DIRECT_DEMOCRACY = 1
    REPRESENTATIVE_DEMOCRACY = 2
    REPUBLIC = 3
    MONARCHY = 4
    OLIGARCHY = 5
    DICTATORSHIP = 6
    TECHNOCRACY = 7
    MERITOCRACY = 8
    CONSENSUS = 9
    DISTRIBUTED = 10

class ResourceType(Enum):
    """Types of resources in the simulation"""
    ENERGY = 0
    COMPUTATION = 1
    MEMORY = 2
    BANDWIDTH = 3
    KNOWLEDGE = 4
    SOCIAL_CAPITAL = 5
    TRUST = 6
    CREATIVITY = 7
    PROBLEM_SOLVING = 8
    PHYSICAL_SPACE = 9

class SocialIdentity:
    """Represents a social identity or group membership"""
    
    def __init__(self, identity_id: str, name: str, description: str, values: Dict[str, float]):
        self.identity_id = identity_id
        self.name = name
        self.description = description
        self.values = values  # Dictionary of value dimensions and strengths
        self.members = set()  # Set of agent IDs who hold this identity
        self.created_at = time.time()
        self.prominence = 0.5  # Initial prominence score
        
    def add_member(self, agent_id: str) -> bool:
        """Add an agent as a member of this identity group"""
        if agent_id in self.members:
            return False
        self.members.add(agent_id)
        return True
    
    def remove_member(self, agent_id: str) -> bool:
        """Remove an agent from this identity group"""
        if agent_id not in self.members:
            return False
        self.members.remove(agent_id)
        return True
    
    def calculate_compatibility(self, other_identity) -> float:
        """Calculate the compatibility between two social identities"""
        if not self.values or not other_identity.values:
            return 0.5
            
        # Calculate the cosine similarity between value vectors
        shared_values = set(self.values.keys()) & set(other_identity.values.keys())
        
        if not shared_values:
            return 0.5
            
        dot_product = sum(self.values[value] * other_identity.values[value] for value in shared_values)
        
        magnitude1 = math.sqrt(sum(self.values[value]**2 for value in self.values))
        magnitude2 = math.sqrt(sum(other_identity.values[value]**2 for value in other_identity.values))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.5
            
        return (dot_product / (magnitude1 * magnitude2) + 1) / 2  # Normalize to [0, 1]
    
    def calculate_value_alignment(self, agent_values: Dict[str, float]) -> float:
        """Calculate how aligned an agent's values are with this identity"""
        if not self.values or not agent_values:
            return 0.5
            
        # Calculate the cosine similarity between value vectors
        shared_values = set(self.values.keys()) & set(agent_values.keys())
        
        if not shared_values:
            return 0.5
            
        dot_product = sum(self.values[value] * agent_values[value] for value in shared_values)
        
        magnitude1 = math.sqrt(sum(self.values[value]**2 for value in self.values))
        magnitude2 = math.sqrt(sum(agent_values[value]**2 for value in agent_values))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.5
            
        return (dot_product / (magnitude1 * magnitude2) + 1) / 2  # Normalize to [0, 1]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the social identity to a dictionary for serialization"""
        return {
            'identity_id': self.identity_id,
            'name': self.name,
            'description': self.description,
            'values': self.values,
            'members': list(self.members),
            'created_at': self.created_at,
            'prominence': self.prominence
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'SocialIdentity':
        """Create a SocialIdentity from a dictionary"""
        identity = SocialIdentity(
            data['identity_id'],
            data['name'],
            data['description'],
            data['values']
        )
        identity.members = set(data['members'])
        identity.created_at = data.get('created_at', time.time())
        identity.prominence = data.get('prominence', 0.5)
        return identity

class SocialRelationship:
    """Represents a relationship between two agents"""
    
    def __init__(self, agent1_id: str, agent2_id: str, relationship_type: RelationshipType = RelationshipType.NEUTRAL):
        self.agent1_id = agent1_id
        self.agent2_id = agent2_id
        self.relationship_type = relationship_type
        self.strength = 0.5  # Initial relationship strength
        self.trust = 0.5  # Initial trust level
        self.formed_at = time.time()
        self.last_interaction = time.time()
        self.interaction_count = 0
        self.history = []  # List of significant events in the relationship
        
    def update_strength(self, delta: float) -> float:
        """Update the relationship strength"""
        self.strength = max(0.0, min(1.0, self.strength + delta))
        return self.strength
    
    def update_trust(self, delta: float) -> float:
        """Update the trust level"""
        self.trust = max(0.0, min(1.0, self.trust + delta))
        return self.trust
    
    def record_interaction(self, interaction_type: str, outcome: float) -> None:
        """Record an interaction between the agents"""
        self.interaction_count += 1
        self.last_interaction = time.time()
        
        self.history.append({
            'type': interaction_type,
            'outcome': outcome,
            'timestamp': time.time()
        })
        
        # Limit history size
        if len(self.history) > 100:
            self.history = self.history[-100:]
    
    def calculate_compatibility(self, agent1_values: Dict[str, float], agent2_values: Dict[str, float]) -> float:
        """Calculate the compatibility between two agents based on their values"""
        if not agent1_values or not agent2_values:
            return 0.5
            
        # Calculate the cosine similarity between value vectors
        shared_values = set(agent1_values.keys()) & set(agent2_values.keys())
        
        if not shared_values:
            return 0.5
            
        dot_product = sum(agent1_values[value] * agent2_values[value] for value in shared_values)
        
        magnitude1 = math.sqrt(sum(agent1_values[value]**2 for value in agent1_values))
        magnitude2 = math.sqrt(sum(agent2_values[value]**2 for value in agent2_values))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.5
            
        return (dot_product / (magnitude1 * magnitude2) + 1) / 2  # Normalize to [0, 1]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the relationship to a dictionary for serialization"""
        return {
            'agent1_id': self.agent1_id,
            'agent2_id': self.agent2_id,
            'relationship_type': self.relationship_type.value,
            'strength': self.strength,
            'trust': self.trust,
            'formed_at': self.formed_at,
            'last_interaction': self.last_interaction,
            'interaction_count': self.interaction_count,
            'history': self.history
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'SocialRelationship':
        """Create a SocialRelationship from a dictionary"""
        relationship = SocialRelationship(
            data['agent1_id'],
            data['agent2_id'],
            RelationshipType(data['relationship_type'])
        )
        relationship.strength = data.get('strength', 0.5)
        relationship.trust = data.get('trust', 0.5)
        relationship.formed_at = data.get('formed_at', time.time())
        relationship.last_interaction = data.get('last_interaction', time.time())
        relationship.interaction_count = data.get('interaction_count', 0)
        relationship.history = data.get('history', [])
        return relationship

class SocialGroup:
    """Represents a group of agents with shared purpose"""
    
    def __init__(self, group_id: str, name: str, description: str, universe_id: str):
        self.group_id = group_id
        self.name = name
        self.description = description
        self.universe_id = universe_id
        self.members = {}  # Dictionary of agent_id -> role
        self.formed_at = time.time()
        self.governance_type = GovernanceType.CONSENSUS
        self.leader_id = None
        self.shared_resources = {}  # Dictionary of resource_type -> amount
        self.roles = {}  # Dictionary of role_name -> list of agent_ids
        self.rules = []  # List of rule dictionaries
        self.cohesion = 0.5  # Group cohesion score
        self.reputation = 0.5  # Group reputation score
        self.activity_level = 0.5  # Group activity level
        
    def add_member(self, agent_id: str, role: str = "member") -> bool:
        """Add an agent to the group with a specific role"""
        if agent_id in self.members:
            return False
            
        self.members[agent_id] = role
        
        if role not in self.roles:
            self.roles[role] = []
            
        self.roles[role].append(agent_id)
        return True
    
    def remove_member(self, agent_id: str) -> bool:
        """Remove an agent from the group"""
        if agent_id not in self.members:
            return False
            
        role = self.members[agent_id]
        del self.members[agent_id]
        
        if role in self.roles and agent_id in self.roles[role]:
            self.roles[role].remove(agent_id)
            
        if agent_id == self.leader_id:
            self.leader_id = None
            
        return True
    
    def set_leader(self, agent_id: str) -> bool:
        """Set an agent as the leader of the group"""
        if agent_id not in self.members:
            return False
            
        self.leader_id = agent_id
        
        # Update roles
        old_role = self.members[agent_id]
        self.members[agent_id] = "leader"
        
        if old_role in self.roles and agent_id in self.roles[old_role]:
            self.roles[old_role].remove(agent_id)
            
        if "leader" not in self.roles:
            self.roles["leader"] = []
            
        self.roles["leader"].append(agent_id)
        return True
    
    def change_governance(self, new_type: GovernanceType) -> bool:
        """Change the governance type of the group"""
        self.governance_type = new_type
        return True
    
    def add_resource(self, resource_type: ResourceType, amount: float) -> float:
        """Add a resource to the group's shared resources"""
        if resource_type not in self.shared_resources:
            self.shared_resources[resource_type] = 0
            
        self.shared_resources[resource_type] += amount
        return self.shared_resources[resource_type]
    
    def use_resource(self, resource_type: ResourceType, amount: float) -> bool:
        """Use a resource from the group's shared resources"""
        if resource_type not in self.shared_resources:
            return False
            
        if self.shared_resources[resource_type] < amount:
            return False
            
        self.shared_resources[resource_type] -= amount
        return True
    
    def add_rule(self, rule_text: str, enforcement: float, creator_id: str) -> int:
        """Add a rule to the group"""
        rule = {
            'id': len(self.rules),
            'text': rule_text,
            'enforcement': enforcement,
            'creator_id': creator_id,
            'created_at': time.time(),
            'supporters': [creator_id],
            'violators': []
        }
        
        self.rules.append(rule)
        return rule['id']
    
    def vote_on_rule(self, rule_id: int, agent_id: str, support: bool) -> bool:
        """Register a vote on a rule"""
        if rule_id >= len(self.rules):
            return False
            
        if agent_id not in self.members:
            return False
            
        rule = self.rules[rule_id]
        
        if support:
            if agent_id not in rule['supporters']:
                rule['supporters'].append(agent_id)
                
            if agent_id in rule['violators']:
                rule['violators'].remove(agent_id)
        else:
            if agent_id in rule['supporters']:
                rule['supporters'].remove(agent_id)
                
        return True
    
    def calculate_cohesion(self) -> float:
        """Calculate the group's cohesion score"""
        if not self.members:
            return 0
            
        # In a real implementation, this would analyze relationships between members
        # and their participation in group activities
        
        # Simple placeholder calculation - ratio of active members
        active_members = sum(1 for member_id in self.members if self.is_active_member(member_id))
        self.cohesion = active_members / len(self.members)
        
        return self.cohesion
    
    def is_active_member(self, agent_id: str) -> bool:
        """Determine if a member is active in the group"""
        # In a real implementation, this would track member participation
        return agent_id in self.members
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the group to a dictionary for serialization"""
        return {
            'group_id': self.group_id,
            'name': self.name,
            'description': self.description,
            'universe_id': self.universe_id,
            'members': self.members,
            'formed_at': self.formed_at,
            'governance_type': self.governance_type.value,
            'leader_id': self.leader_id,
            'shared_resources': {str(k.value): v for k, v in self.shared_resources.items()},
            'roles': self.roles,
            'rules': self.rules,
            'cohesion': self.cohesion,
            'reputation': self.reputation,
            'activity_level': self.activity_level
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'SocialGroup':
        """Create a SocialGroup from a dictionary"""
        group = SocialGroup(
            data['group_id'],
            data['name'],
            data['description'],
            data['universe_id']
        )
        group.members = data.get('members', {})
        group.formed_at = data.get('formed_at', time.time())
        group.governance_type = GovernanceType(data.get('governance_type', 0))
        group.leader_id = data.get('leader_id', None)
        # Convert resource types from strings back to enum
        group.shared_resources = {ResourceType(int(k)): v for k, v in data.get('shared_resources', {}).items()}
        group.roles = data.get('roles', {})
        group.rules = data.get('rules', [])
        group.cohesion = data.get('cohesion', 0.5)
        group.reputation = data.get('reputation', 0.5)
        group.activity_level = data.get('activity_level', 0.5)
        return group

class Policy:
    """Represents a policy or law in the governance system"""
    
    def __init__(self, policy_id: str, title: str, description: str, creator_id: str, group_id: str = None):
        self.policy_id = policy_id
        self.title = title
        self.description = description
        self.creator_id = creator_id
        self.group_id = group_id  # If None, this is a society-wide policy
        self.created_at = time.time()
        self.last_modified = time.time()
        self.status = "proposed"  # proposed, active, inactive, rejected
        self.votes = {}  # Dictionary of agent_id -> vote (True/False)
        self.enforcement_level = 0.5  # How strictly the policy is enforced
        self.compliance_rate = 0.0  # How well agents comply with the policy
        self.impact_metrics = {}  # Dictionary of metric_name -> value
        
    def vote(self, agent_id: str, support: bool) -> bool:
        """Register a vote for this policy"""
        self.votes[agent_id] = support
        return True
    
    def calculate_support(self) -> float:
        """Calculate the support level for this policy"""
        if not self.votes:
            return 0
            
        support_count = sum(1 for vote in self.votes.values() if vote)
        return support_count / len(self.votes)
    
    def activate(self) -> bool:
        """Activate the policy"""
        self.status = "active"
        self.last_modified = time.time()
        return True
    
    def deactivate(self) -> bool:
        """Deactivate the policy"""
        self.status = "inactive"
        self.last_modified = time.time()
        return True
    
    def reject(self) -> bool:
        """Reject the policy"""
        self.status = "rejected"
        self.last_modified = time.time()
        return True
    
    def update_compliance(self, compliance_rate: float) -> float:
        """Update the compliance rate for this policy"""
        self.compliance_rate = max(0.0, min(1.0, compliance_rate))
        return self.compliance_rate
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the policy to a dictionary for serialization"""
        return {
            'policy_id': self.policy_id,
            'title': self.title,
            'description': self.description,
            'creator_id': self.creator_id,
            'group_id': self.group_id,
            'created_at': self.created_at,
            'last_modified': self.last_modified,
            'status': self.status,
            'votes': self.votes,
            'enforcement_level': self.enforcement_level,
            'compliance_rate': self.compliance_rate,
            'impact_metrics': self.impact_metrics
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Policy':
        """Create a Policy from a dictionary"""
        policy = Policy(
            data['policy_id'],
            data['title'],
            data['description'],
            data['creator_id'],
            data.get('group_id', None)
        )
        policy.created_at = data.get('created_at', time.time())
        policy.last_modified = data.get('last_modified', time.time())
        policy.status = data.get('status', "proposed")
        policy.votes = data.get('votes', {})
        policy.enforcement_level = data.get('enforcement_level', 0.5)
        policy.compliance_rate = data.get('compliance_rate', 0.0)
        policy.impact_metrics = data.get('impact_metrics', {})
        return policy

class CulturalMeme:
    """Represents a cultural element or meme that can spread between agents"""
    
    def __init__(self, meme_id: str, name: str, description: str, creator_id: str):
        self.meme_id = meme_id
        self.name = name
        self.description = description
        self.creator_id = creator_id
        self.created_at = time.time()
        self.adopters = set([creator_id])  # Set of agent IDs who have adopted this meme
        self.spread_rate = 0.5  # How quickly the meme spreads
        self.persistence = 0.5  # How long the meme tends to persist
        self.mutation_rate = 0.1  # How quickly the meme evolves
        self.variants = []  # List of variant meme IDs
        self.parent_id = None  # Parent meme ID if this is a variant
        self.attributes = {}  # Dictionary of attributes
        
    def add_adopter(self, agent_id: str) -> bool:
        """Add an agent as an adopter of this meme"""
        if agent_id in self.adopters:
            return False
        self.adopters.add(agent_id)
        return True
    
    def remove_adopter(self, agent_id: str) -> bool:
        """Remove an agent from the adopters of this meme"""
        if agent_id not in self.adopters:
            return False
        self.adopters.remove(agent_id)
        return True
    
    def create_variant(self, variant_id: str, name: str, description: str, creator_id: str, attribute_changes: Dict[str, Any]) -> 'CulturalMeme':
        """Create a variant of this meme"""
        variant = CulturalMeme(variant_id, name, description, creator_id)
        variant.parent_id = self.meme_id
        
        # Copy attributes from parent and apply changes
        variant.attributes = self.attributes.copy()
        variant.attributes.update(attribute_changes)
        
        # Link in parent's variants list
        self.variants.append(variant_id)
        
        return variant
    
    def get_adoption_rate(self, total_population: int) -> float:
        """Calculate the adoption rate of this meme"""
        if total_population == 0:
            return 0
        return len(self.adopters) / total_population
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the meme to a dictionary for serialization"""
        return {
            'meme_id': self.meme_id,
            'name': self.name,
            'description': self.description,
            'creator_id': self.creator_id,
            'created_at': self.created_at,
            'adopters': list(self.adopters),
            'spread_rate': self.spread_rate,
            'persistence': self.persistence,
            'mutation_rate': self.mutation_rate,
            'variants': self.variants,
            'parent_id': self.parent_id,
            'attributes': self.attributes
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'CulturalMeme':
        """Create a CulturalMeme from a dictionary"""
        meme = CulturalMeme(
            data['meme_id'],
            data['name'],
            data['description'],
            data['creator_id']
        )
        meme.created_at = data.get('created_at', time.time())
        meme.adopters = set(data.get('adopters', []))
        meme.spread_rate = data.get('spread_rate', 0.5)
        meme.persistence = data.get('persistence', 0.5)
        meme.mutation_rate = data.get('mutation_rate', 0.1)
        meme.variants = data.get('variants', [])
        meme.parent_id = data.get('parent_id', None)
        meme.attributes = data.get('attributes', {})
        return meme

class SocialNorm:
    """Represents a social norm or expectation within a society or group"""
    
    def __init__(self, norm_id: str, name: str, description: str, group_id: str = None):
        self.norm_id = norm_id
        self.name = name
        self.description = description
        self.group_id = group_id  # If None, this is a society-wide norm
        self.created_at = time.time()
        self.strength = 0.5  # How strongly the norm is held
        self.adherence = 0.5  # How well agents adhere to the norm
        self.formality = 0.5  # How formal or codified the norm is
        self.sanctions = []  # List of sanctions for violating the norm
        self.related_norms = []  # List of related norm IDs
        self.contradicting_norms = []  # List of contradicting norm IDs
        
    def add_sanction(self, description: str, severity: float) -> int:
        """Add a sanction for violating this norm"""
        sanction = {
            'id': len(self.sanctions),
            'description': description,
            'severity': severity,
            'created_at': time.time()
        }
        
        self.sanctions.append(sanction)
        return sanction['id']
    
    def add_related_norm(self, norm_id: str) -> bool:
        """Add a related norm"""
        if norm_id in self.related_norms:
            return False
        self.related_norms.append(norm_id)
        return True
    
    def add_contradicting_norm(self, norm_id: str) -> bool:
        """Add a contradicting norm"""
        if norm_id in self.contradicting_norms:
            return False
        self.contradicting_norms.append(norm_id)
        return True
    
    def update_strength(self, delta: float) -> float:
        """Update the strength of the norm"""
        self.strength = max(0.0, min(1.0, self.strength + delta))
        return self.strength
    
    def update_adherence(self, adherence: float) -> float:
        """Update the adherence level for this norm"""
        self.adherence = max(0.0, min(1.0, adherence))
        return self.adherence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the norm to a dictionary for serialization"""
        return {
            'norm_id': self.norm_id,
            'name': self.name,
            'description': self.description,
            'group_id': self.group_id,
            'created_at': self.created_at,
            'strength': self.strength,
            'adherence': self.adherence,
            'formality': self.formality,
            'sanctions': self.sanctions,
            'related_norms': self.related_norms,
            'contradicting_norms': self.contradicting_norms
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'SocialNorm':
        """Create a SocialNorm from a dictionary"""
        norm = SocialNorm(
            data['norm_id'],
            data['name'],
            data['description'],
            data.get('group_id', None)
        )
        norm.created_at = data.get('created_at', time.time())
        norm.strength = data.get('strength', 0.5)
        norm.adherence = data.get('adherence', 0.5)
        norm.formality = data.get('formality', 0.5)
        norm.sanctions = data.get('sanctions', [])
        norm.related_norms = data.get('related_norms', [])
        norm.contradicting_norms = data.get('contradicting_norms', [])
        return norm

class EconomicSystem:
    """Manages the exchange and production of resources within a society"""
    
    def __init__(self, universe_id: str, name: str = "Default Economy"):
        self.universe_id = universe_id
        self.name = name
        self.created_at = time.time()
        self.resources = {}  # Dictionary of resource_type -> total available
        self.agent_resources = defaultdict(lambda: defaultdict(float))  # agent_id -> resource_type -> amount
        self.production_rates = defaultdict(lambda: defaultdict(float))  # agent_id -> resource_type -> rate
        self.consumption_rates = defaultdict(lambda: defaultdict(float))  # agent_id -> resource_type -> rate
        self.trade_history = []  # List of trade dictionaries
        self.price_history = defaultdict(list)  # resource_type -> list of (timestamp, price) tuples
        self.inflation_rate = 0.0  # Current inflation rate
        self.transaction_count = 0  # Number of transactions processed
        
    def allocate_resource(self, agent_id: str, resource_type: ResourceType, amount: float) -> bool:
        """Allocate a resource to an agent"""
        # Check if there are enough resources available
        if resource_type in self.resources and self.resources[resource_type] >= amount:
            self.resources[resource_type] -= amount
            self.agent_resources[agent_id][resource_type] += amount
            return True
        return False
    
    def transfer_resource(self, from_agent_id: str, to_agent_id: str, resource_type: ResourceType, amount: float, price: float = 0.0) -> bool:
        """Transfer a resource from one agent to another (trade)"""
        # Check if the source agent has enough resources
        if self.agent_resources[from_agent_id][resource_type] < amount:
            return False
            
        # Perform the transfer
        self.agent_resources[from_agent_id][resource_type] -= amount
        self.agent_resources[to_agent_id][resource_type] += amount
        
        # Record the trade
        trade = {
            'id': self.transaction_count,
            'from_agent_id': from_agent_id,
            'to_agent_id': to_agent_id,
            'resource_type': resource_type.value,
            'amount': amount,
            'price': price,
            'timestamp': time.time()
        }
        
        self.trade_history.append(trade)
        self.transaction_count += 1
        
        # Update price history
        self.price_history[resource_type].append((time.time(), price))
        
        return True
    
    def produce_resource(self, agent_id: str, resource_type: ResourceType, amount: float) -> float:
        """Record production of a resource by an agent"""
        # Add to total resources
        if resource_type not in self.resources:
            self.resources[resource_type] = 0
            
        self.resources[resource_type] += amount
        
        # Add to agent's resources
        self.agent_resources[agent_id][resource_type] += amount
        
        # Update production rate (using exponential moving average)
        current_rate = self.production_rates[agent_id][resource_type]
        alpha = 0.3  # Smoothing factor
        new_rate = alpha * amount + (1 - alpha) * current_rate
        self.production_rates[agent_id][resource_type] = new_rate
        
        return amount
    
    def consume_resource(self, agent_id: str, resource_type: ResourceType, amount: float) -> bool:
        """Record consumption of a resource by an agent"""
        # Check if the agent has enough of the resource
        if self.agent_resources[agent_id][resource_type] < amount:
            return False
            
        # Deduct from agent's resources
        self.agent_resources[agent_id][resource_type] -= amount
        
        # Update consumption rate (using exponential moving average)
        current_rate = self.consumption_rates[agent_id][resource_type]
        alpha = 0.3  # Smoothing factor
        new_rate = alpha * amount + (1 - alpha) * current_rate
        self.consumption_rates[agent_id][resource_type] = new_rate
        
        return True
    
    def get_price(self, resource_type: ResourceType) -> float:
        """Get the current price of a resource"""
        if resource_type not in self.price_history or not self.price_history[resource_type]:
            return 1.0  # Default price
            
        # Use the most recent price
        return self.price_history[resource_type][-1][1]
    
    def calculate_resource_supply(self, resource_type: ResourceType) -> float:
        """Calculate the total supply of a resource"""
        if resource_type not in self.resources:
            return 0
            
        return self.resources[resource_type]
    
    def calculate_resource_demand(self, resource_type: ResourceType) -> float:
        """Calculate the total demand for a resource based on consumption rates"""
        total_demand = sum(rates[resource_type] for rates in self.consumption_rates.values())
        return total_demand
    
    def calculate_inflation(self) -> float:
        """Calculate the inflation rate based on price history"""
        # This is a simplified model - in a real implementation,
        # this would use more sophisticated economic models
        
        if not any(self.price_history.values()):
            return 0.0
            
        # Calculate average price change across all resources
        total_change = 0
        resource_count = 0
        
        for resource_type, history in self.price_history.items():
            if len(history) < 2:
                continue
                
            old_price = history[-2][1]  # Second most recent price
            new_price = history[-1][1]  # Most recent price
            
            if old_price == 0:
                continue
                
            percent_change = (new_price - old_price) / old_price
            total_change += percent_change
            resource_count += 1
        
        if resource_count == 0:
            return 0.0
            
        self.inflation_rate = total_change / resource_count
        return self.inflation_rate
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the economic system to a dictionary for serialization"""
        return {
            'universe_id': self.universe_id,
            'name': self.name,
            'created_at': self.created_at,
            'resources': {str(k.value): v for k, v in self.resources.items()},
            'agent_resources': {agent_id: {str(k.value): v for k, v in resources.items()} 
                               for agent_id, resources in self.agent_resources.items()},
            'production_rates': {agent_id: {str(k.value): v for k, v in rates.items()} 
                                for agent_id, rates in self.production_rates.items()},
            'consumption_rates': {agent_id: {str(k.value): v for k, v in rates.items()} 
                                 for agent_id, rates in self.consumption_rates.items()},
            'trade_history': self.trade_history[-100:],  # Keep only the most recent 100 trades
            'price_history': {str(k.value): v[-100:] for k, v in self.price_history.items()},  # Keep only the most recent 100 prices
            'inflation_rate': self.inflation_rate,
            'transaction_count': self.transaction_count
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'EconomicSystem':
        """Create an EconomicSystem from a dictionary"""
        economy = EconomicSystem(
            data['universe_id'],
            data.get('name', "Default Economy")
        )
        economy.created_at = data.get('created_at', time.time())
        # Convert resource types from strings back to enum
        economy.resources = {ResourceType(int(k)): v for k, v in data.get('resources', {}).items()}
        
        # Convert agent resources
        for agent_id, resources in data.get('agent_resources', {}).items():
            for k, v in resources.items():
                economy.agent_resources[agent_id][ResourceType(int(k))] = v
        
        # Convert production rates
        for agent_id, rates in data.get('production_rates', {}).items():
            for k, v in rates.items():
                economy.production_rates[agent_id][ResourceType(int(k))] = v
        
        # Convert consumption rates
        for agent_id, rates in data.get('consumption_rates', {}).items():
            for k, v in rates.items():
                economy.consumption_rates[agent_id][ResourceType(int(k))] = v
        
        economy.trade_history = data.get('trade_history', [])
        
        # Convert price history
        for k, v in data.get('price_history', {}).items():
            economy.price_history[ResourceType(int(k))] = v
        
        economy.inflation_rate = data.get('inflation_rate', 0.0)
        economy.transaction_count = data.get('transaction_count', 0)
        return economy

class GovernanceSystem:
    """Main class for managing civilization and governance systems"""
    
    def __init__(self, universe_id: str, name: str = "Default Governance"):
        self.universe_id = universe_id
        self.name = name
        self.created_at = time.time()
        
        # Social components
        self.social_relationships = {}  # Dictionary of (agent1_id, agent2_id) -> SocialRelationship
        self.social_groups = {}  # Dictionary of group_id -> SocialGroup
        self.social_identities = {}  # Dictionary of identity_id -> SocialIdentity
        self.social_norms = {}  # Dictionary of norm_id -> SocialNorm
        
        # Cultural components
        self.cultural_memes = {}  # Dictionary of meme_id -> CulturalMeme
        
        # Governance components
        self.policies = {}  # Dictionary of policy_id -> Policy
        self.governance_type = GovernanceType.CONSENSUS
        self.leaders = []  # List of leader agent IDs
        self.authorities = {}  # Dictionary of domain -> list of authority agent IDs
        
        # Economic components
        self.economic_system = EconomicSystem(universe_id)
        
        # Social network metrics
        self.social_graph = defaultdict(set)  # agent_id -> set of connected agent_ids
        self.centrality_scores = {}  # agent_id -> centrality score
        
        # Save/load paths
        self.base_path = f"/var/lib/synthix/governance/{universe_id}"
        
        # Create storage directories
        os.makedirs(self.base_path, exist_ok=True)
    
    def get_relationship(self, agent1_id: str, agent2_id: str) -> SocialRelationship:
        """Get the social relationship between two agents, creating it if it doesn't exist"""
        # Ensure consistent ordering for the key
        if agent1_id > agent2_id:
            agent1_id, agent2_id = agent2_id, agent1_id
            
        key = (agent1_id, agent2_id)
        
        if key not in self.social_relationships:
            self.social_relationships[key] = SocialRelationship(agent1_id, agent2_id)
            
        return self.social_relationships[key]
    
    def update_relationship(self, agent1_id: str, agent2_id: str, interaction_type: str, outcome: float) -> SocialRelationship:
        """Update the relationship between two agents based on an interaction"""
        relationship = self.get_relationship(agent1_id, agent2_id)
        
        # Record the interaction
        relationship.record_interaction(interaction_type, outcome)
        
        # Update strength and trust based on the outcome
        if outcome > 0:
            relationship.update_strength(outcome * 0.1)
            relationship.update_trust(outcome * 0.1)
        else:
            relationship.update_strength(outcome * 0.2)
            relationship.update_trust(outcome * 0.2)
        
        # Update social graph
        self.social_graph[agent1_id].add(agent2_id)
        self.social_graph[agent2_id].add(agent1_id)
        
        return relationship
    
    def create_group(self, name: str, description: str, creator_id: str) -> SocialGroup:
        """Create a new social group"""
        group_id = str(uuid.uuid4())
        group = SocialGroup(group_id, name, description, self.universe_id)
        
        # Add creator as a member and leader
        group.add_member(creator_id, "leader")
        group.leader_id = creator_id
        
        self.social_groups[group_id] = group
        return group
    
    def create_identity(self, name: str, description: str, values: Dict[str, float], creator_id: str) -> SocialIdentity:
        """Create a new social identity"""
        identity_id = str(uuid.uuid4())
        identity = SocialIdentity(identity_id, name, description, values)
        
        # Add creator as a member
        identity.add_member(creator_id)
        
        self.social_identities[identity_id] = identity
        return identity
    
    def create_norm(self, name: str, description: str, group_id: str = None) -> SocialNorm:
        """Create a new social norm"""
        norm_id = str(uuid.uuid4())
        norm = SocialNorm(norm_id, name, description, group_id)
        
        self.social_norms[norm_id] = norm
        return norm
    
    def create_policy(self, title: str, description: str, creator_id: str, group_id: str = None) -> Policy:
        """Create a new policy"""
        policy_id = str(uuid.uuid4())
        policy = Policy(policy_id, title, description, creator_id, group_id)
        
        self.policies[policy_id] = policy
        return policy
    
    def create_meme(self, name: str, description: str, creator_id: str, attributes: Dict[str, Any] = None) -> CulturalMeme:
        """Create a new cultural meme"""
        meme_id = str(uuid.uuid4())
        meme = CulturalMeme(meme_id, name, description, creator_id)
        
        if attributes:
            meme.attributes = attributes
            
        self.cultural_memes[meme_id] = meme
        return meme
    
    def spread_meme(self, meme_id: str, from_agent_id: str, to_agent_id: str, adoption_probability: float = 0.5) -> bool:
        """Attempt to spread a meme from one agent to another"""
        if meme_id not in self.cultural_memes:
            return False
            
        meme = self.cultural_memes[meme_id]
        
        if from_agent_id not in meme.adopters:
            return False
            
        if to_agent_id in meme.adopters:
            return True  # Already adopted
            
        # Get the relationship to modify adoption probability
        relationship = self.get_relationship(from_agent_id, to_agent_id)
        
        # Adjust probability based on relationship strength and trust
        adjusted_probability = adoption_probability * relationship.strength * relationship.trust
        
        # Factor in the meme's spread rate
        adjusted_probability *= meme.spread_rate
        
        # Random chance to adopt
        if random.random() < adjusted_probability:
            meme.add_adopter(to_agent_id)
            
            # Record this as a positive interaction
            self.update_relationship(from_agent_id, to_agent_id, "MEME_SPREAD", 0.1)
            
            return True
            
        return False
    
    def elect_leader(self, candidates: List[str], voters: List[str], method: str = "plurality") -> str:
        """Elect a leader using the specified voting method"""
        if not candidates or not voters:
            return None
            
        if method == "plurality":
            # Each voter votes for one candidate
            votes = {}
            for candidate in candidates:
                votes[candidate] = 0
                
            for voter in voters:
                # In a real implementation, this would use the voter's preferences
                # Here we'll just use a simple random vote weighted by relationships
                weights = []
                for candidate in candidates:
                    if candidate == voter:
                        # Small bonus for voting for oneself
                        weights.append(0.2)
                    elif (voter, candidate) in self.social_relationships or (candidate, voter) in self.social_relationships:
                        # Use relationship strength as weight
                        rel = self.get_relationship(voter, candidate)
                        weights.append(max(0.1, rel.strength))
                    else:
                        # No relationship
                        weights.append(0.1)
                        
                # Normalize weights
                total = sum(weights)
                if total > 0:
                    weights = [w/total for w in weights]
                else:
                    weights = [1/len(candidates)] * len(candidates)
                    
                # Random weighted choice
                chosen = random.choices(candidates, weights=weights, k=1)[0]
                votes[chosen] = votes.get(chosen, 0) + 1
                
            # Find the winner
            winner = max(votes.items(), key=lambda x: x[1])[0]
            return winner
            
        elif method == "approval":
            # Each voter can approve multiple candidates
            approvals = {}
            for candidate in candidates:
                approvals[candidate] = 0
                
            for voter in voters:
                # In a real implementation, this would use the voter's preferences
                # Here we'll just approve candidates with good relationships
                for candidate in candidates:
                    if candidate == voter:
                        # Always approve oneself
                        approvals[candidate] = approvals.get(candidate, 0) + 1
                    elif (voter, candidate) in self.social_relationships or (candidate, voter) in self.social_relationships:
                        rel = self.get_relationship(voter, candidate)
                        if rel.strength > 0.5:  # Approve if relationship is positive
                            approvals[candidate] = approvals.get(candidate, 0) + 1
                            
            # Find the winner
            winner = max(approvals.items(), key=lambda x: x[1])[0]
            return winner
            
        return None
    
    def resolve_resource_conflict(self, agent1_id: str, agent2_id: str, resource_type: ResourceType, amount: float) -> Tuple[str, float]:
        """Resolve a conflict over resources between two agents"""
        # Get the social relationship
        relationship = self.get_relationship(agent1_id, agent2_id)
        
        # Get the agents' positions in the social network
        agent1_centrality = self.centrality_scores.get(agent1_id, 0.5)
        agent2_centrality = self.centrality_scores.get(agent2_id, 0.5)
        
        # Calculate the power ratio
        agent1_power = agent1_centrality * (1 + len(self.social_graph[agent1_id]) * 0.1)
        agent2_power = agent2_centrality * (1 + len(self.social_graph[agent2_id]) * 0.1)
        
        # Adjust based on relationship
        if relationship.relationship_type == RelationshipType.LEADER and relationship.agent1_id == agent1_id:
            agent1_power *= 1.5
        elif relationship.relationship_type == RelationshipType.LEADER and relationship.agent1_id == agent2_id:
            agent2_power *= 1.5
            
        # Calculate outcome
        total_power = agent1_power + agent2_power
        if total_power == 0:
            agent1_share = 0.5
        else:
            agent1_share = agent1_power / total_power
            
        agent2_share = 1 - agent1_share
        
        # Determine who gets more of the resource
        if agent1_share > agent2_share:
            winner = agent1_id
            winner_amount = amount * agent1_share
        else:
            winner = agent2_id
            winner_amount = amount * agent2_share
            
        # Record the conflict as an interaction with negative outcome
        self.update_relationship(agent1_id, agent2_id, "RESOURCE_CONFLICT", -0.3)
        
        return winner, winner_amount
    
    def update_social_metrics(self) -> None:
        """Update various social network metrics"""
        # Calculate degree centrality (very basic centrality measure)
        for agent_id in self.social_graph:
            self.centrality_scores[agent_id] = len(self.social_graph[agent_id]) / max(1, len(self.social_graph))
        
        # In a real implementation, this would calculate more sophisticated
        # network metrics like betweenness centrality, closeness centrality, etc.
    
    def simulate_cultural_evolution(self, iterations: int = 1) -> None:
        """Simulate the evolution of cultural memes"""
        if not self.cultural_memes:
            return
            
        for _ in range(iterations):
            # Randomly select a meme to potentially mutate
            meme_id = random.choice(list(self.cultural_memes.keys()))
            meme = self.cultural_memes[meme_id]
            
            # Check if it should mutate based on mutation rate
            if random.random() < meme.mutation_rate:
                # Create a variant with slightly different attributes
                variant_name = f"{meme.name} Variant"
                variant_description = f"A variant of {meme.name}"
                
                # Select a random adopter as the creator
                if meme.adopters:
                    creator_id = random.choice(list(meme.adopters))
                    
                    # Create random attribute changes
                    attribute_changes = {}
                    for key, value in meme.attributes.items():
                        if isinstance(value, (int, float)):
                            # Perturb numeric values
                            attribute_changes[key] = value * (0.8 + random.random() * 0.4)
                        elif isinstance(value, str):
                            # Slightly modify strings
                            attribute_changes[key] = value + " (modified)"
                            
                    # Create the variant
                    variant_id = str(uuid.uuid4())
                    variant = meme.create_variant(variant_id, variant_name, variant_description, creator_id, attribute_changes)
                    
                    # Add it to the meme collection
                    self.cultural_memes[variant_id] = variant
                    
            # Randomly select memes to spread
            if self.social_graph:
                # Randomly select a sender
                from_agent_id = random.choice(list(self.social_graph.keys()))
                
                # Find their connections
                connections = self.social_graph[from_agent_id]
                if connections:
                    # Randomly select a receiver
                    to_agent_id = random.choice(list(connections))
                    
                    # Find memes the sender has adopted
                    adopted_memes = [m_id for m_id, m in self.cultural_memes.items() if from_agent_id in m.adopters]
                    if adopted_memes:
                        # Try to spread a random meme
                        meme_to_spread = random.choice(adopted_memes)
                        self.spread_meme(meme_to_spread, from_agent_id, to_agent_id)
    
    def save(self) -> bool:
        """Save the governance system state to files"""
        try:
            # Save social relationships
            relationships_data = {str(k): v.to_dict() for k, v in self.social_relationships.items()}
            with open(f"{self.base_path}/relationships.json", 'w') as f:
                json.dump(relationships_data, f, indent=2)
                
            # Save social groups
            groups_data = {k: v.to_dict() for k, v in self.social_groups.items()}
            with open(f"{self.base_path}/groups.json", 'w') as f:
                json.dump(groups_data, f, indent=2)
                
            # Save social identities
            identities_data = {k: v.to_dict() for k, v in self.social_identities.items()}
            with open(f"{self.base_path}/identities.json", 'w') as f:
                json.dump(identities_data, f, indent=2)
                
            # Save social norms
            norms_data = {k: v.to_dict() for k, v in self.social_norms.items()}
            with open(f"{self.base_path}/norms.json", 'w') as f:
                json.dump(norms_data, f, indent=2)
                
            # Save cultural memes
            memes_data = {k: v.to_dict() for k, v in self.cultural_memes.items()}
            with open(f"{self.base_path}/memes.json", 'w') as f:
                json.dump(memes_data, f, indent=2)
                
            # Save policies
            policies_data = {k: v.to_dict() for k, v in self.policies.items()}
            with open(f"{self.base_path}/policies.json", 'w') as f:
                json.dump(policies_data, f, indent=2)
                
            # Save economic system
            economy_data = self.economic_system.to_dict()
            with open(f"{self.base_path}/economy.json", 'w') as f:
                json.dump(economy_data, f, indent=2)
                
            # Save basic system info
            system_data = {
                'universe_id': self.universe_id,
                'name': self.name,
                'created_at': self.created_at,
                'governance_type': self.governance_type.value,
                'leaders': self.leaders,
                'authorities': self.authorities,
                'social_graph': {k: list(v) for k, v in self.social_graph.items()},
                'centrality_scores': self.centrality_scores
            }
            with open(f"{self.base_path}/system.json", 'w') as f:
                json.dump(system_data, f, indent=2)
                
            return True
                
        except Exception as e:
            logger.error(f"Failed to save governance system: {e}")
            return False
    
    def load(self) -> bool:
        """Load the governance system state from files"""
        try:
            # Check if the base path exists
            if not os.path.exists(self.base_path):
                return False
                
            # Load basic system info
            system_path = f"{self.base_path}/system.json"
            if os.path.exists(system_path):
                with open(system_path, 'r') as f:
                    system_data = json.load(f)
                    
                self.name = system_data.get('name', self.name)
                self.created_at = system_data.get('created_at', self.created_at)
                self.governance_type = GovernanceType(system_data.get('governance_type', 0))
                self.leaders = system_data.get('leaders', [])
                self.authorities = system_data.get('authorities', {})
                self.social_graph = defaultdict(set, {k: set(v) for k, v in system_data.get('social_graph', {}).items()})
                self.centrality_scores = system_data.get('centrality_scores', {})
                
            # Load social relationships
            relationships_path = f"{self.base_path}/relationships.json"
            if os.path.exists(relationships_path):
                with open(relationships_path, 'r') as f:
                    relationships_data = json.load(f)
                    
                for k_str, v_dict in relationships_data.items():
                    # Convert string key back to tuple
                    k_parts = k_str.strip('()').split(', ')
                    k = (k_parts[0], k_parts[1])
                    self.social_relationships[k] = SocialRelationship.from_dict(v_dict)
                    
            # Load social groups
            groups_path = f"{self.base_path}/groups.json"
            if os.path.exists(groups_path):
                with open(groups_path, 'r') as f:
                    groups_data = json.load(f)
                    
                for k, v_dict in groups_data.items():
                    self.social_groups[k] = SocialGroup.from_dict(v_dict)
                    
            # Load social identities
            identities_path = f"{self.base_path}/identities.json"
            if os.path.exists(identities_path):
                with open(identities_path, 'r') as f:
                    identities_data = json.load(f)
                    
                for k, v_dict in identities_data.items():
                    self.social_identities[k] = SocialIdentity.from_dict(v_dict)
                    
            # Load social norms
            norms_path = f"{self.base_path}/norms.json"
            if os.path.exists(norms_path):
                with open(norms_path, 'r') as f:
                    norms_data = json.load(f)
                    
                for k, v_dict in norms_data.items():
                    self.social_norms[k] = SocialNorm.from_dict(v_dict)
                    
            # Load cultural memes
            memes_path = f"{self.base_path}/memes.json"
            if os.path.exists(memes_path):
                with open(memes_path, 'r') as f:
                    memes_data = json.load(f)
                    
                for k, v_dict in memes_data.items():
                    self.cultural_memes[k] = CulturalMeme.from_dict(v_dict)
                    
            # Load policies
            policies_path = f"{self.base_path}/policies.json"
            if os.path.exists(policies_path):
                with open(policies_path, 'r') as f:
                    policies_data = json.load(f)
                    
                for k, v_dict in policies_data.items():
                    self.policies[k] = Policy.from_dict(v_dict)
                    
            # Load economic system
            economy_path = f"{self.base_path}/economy.json"
            if os.path.exists(economy_path):
                with open(economy_path, 'r') as f:
                    economy_data = json.load(f)
                    
                self.economic_system = EconomicSystem.from_dict(economy_data)
                
            return True
                
        except Exception as e:
            logger.error(f"Failed to load governance system: {e}")
            return False
    
    def get_social_network_stats(self) -> Dict[str, Any]:
        """Calculate statistics about the social network"""
        stats = {
            'agent_count': len(self.social_graph),
            'relationship_count': len(self.social_relationships),
            'group_count': len(self.social_groups),
            'identity_count': len(self.social_identities),
            'norm_count': len(self.social_norms),
            'meme_count': len(self.cultural_memes),
            'policy_count': len(self.policies),
            'average_connections': 0,
            'max_connections': 0,
            'isolated_agents': 0,
            'average_relationship_strength': 0,
            'group_membership_distribution': {},
            'identity_adoption_distribution': {}
        }
        
        # Connection stats
        if self.social_graph:
            connection_counts = [len(connections) for connections in self.social_graph.values()]
            stats['average_connections'] = sum(connection_counts) / len(connection_counts)
            stats['max_connections'] = max(connection_counts) if connection_counts else 0
            stats['isolated_agents'] = sum(1 for count in connection_counts if count == 0)
        
        # Relationship strength stats
        if self.social_relationships:
            strengths = [rel.strength for rel in self.social_relationships.values()]
            stats['average_relationship_strength'] = sum(strengths) / len(strengths)
            
        # Group membership distribution
        for group in self.social_groups.values():
            member_count = len(group.members)
            stats['group_membership_distribution'][member_count] = stats['group_membership_distribution'].get(member_count, 0) + 1
            
        # Identity adoption distribution
        for identity in self.social_identities.values():
            adopter_count = len(identity.members)
            stats['identity_adoption_distribution'][adopter_count] = stats['identity_adoption_distribution'].get(adopter_count, 0) + 1
            
        return stats
    
    def simulate_day(self) -> Dict[str, Any]:
        """Simulate a day of social interactions, governance, and economic activity"""
        # Track events and metrics for the day
        daily_metrics = {
            'interactions': 0,
            'new_relationships': 0,
            'economic_transactions': 0,
            'policy_changes': 0,
            'meme_spreads': 0,
            'conflict_resolutions': 0,
            'elections': 0,
            'significant_events': []
        }
        
        # 1. Simulate social interactions
        if len(self.social_graph) >= 2:
            # Determine the number of interactions based on network size
            interaction_count = int(len(self.social_graph) * 0.3) + 1
            
            for _ in range(interaction_count):
                # Randomly select two agents
                agent1_id = random.choice(list(self.social_graph.keys()))
                
                # Decide if this is interaction with an existing connection or a new one
                if random.random() < 0.7 and self.social_graph[agent1_id]:  # 70% chance to interact with existing connection
                    agent2_id = random.choice(list(self.social_graph[agent1_id]))
                else:
                    # Find an agent not already connected
                    potential_agents = [a for a in self.social_graph.keys() if a != agent1_id and a not in self.social_graph[agent1_id]]
                    if potential_agents:
                        agent2_id = random.choice(potential_agents)
                        daily_metrics['new_relationships'] += 1
                    else:
                        agent2_id = random.choice(list(self.social_graph.keys()))
                        
                # Skip if we selected the same agent
                if agent1_id == agent2_id:
                    continue
                    
                # Determine interaction outcome (-1.0 to 1.0)
                # In a real implementation, this would depend on agent personalities and circumstances
                interaction_outcome = random.random() * 2 - 1.0
                
                # Choose interaction type
                interaction_types = [
                    "CONVERSATION", "COLLABORATION", "TRADE", "DISPUTE", 
                    "TEACHING", "LEARNING", "ASSISTANCE", "COMPETITION"
                ]
                interaction_type = random.choice(interaction_types)
                
                # Update the relationship
                self.update_relationship(agent1_id, agent2_id, interaction_type, interaction_outcome)
                daily_metrics['interactions'] += 1
                
                # Generate a significant event for notable interactions
                if abs(interaction_outcome) > 0.8:
                    event_type = "positive" if interaction_outcome > 0 else "negative"
                    event = {
                        'type': f"{event_type.upper()}_{interaction_type}",
                        'agents': [agent1_id, agent2_id],
                        'outcome': interaction_outcome,
                        'timestamp': time.time()
                    }
                    daily_metrics['significant_events'].append(event)
        
        # 2. Simulate economic transactions
        if len(self.social_graph) >= 2:
            # Determine number of transactions
            transaction_count = int(len(self.social_graph) * 0.2) + 1
            
            for _ in range(transaction_count):
                # Randomly select two agents
                agent1_id = random.choice(list(self.social_graph.keys()))
                agent2_id = random.choice([a for a in self.social_graph.keys() if a != agent1_id])
                
                # Randomly select a resource type
                resource_type = random.choice(list(ResourceType))
                
                # Determine amount and price
                amount = random.uniform(0.1, 10.0)
                price = random.uniform(0.5, 5.0)
                
                # Only proceed if agent1 has the resource
                if self.economic_system.agent_resources[agent1_id][resource_type] >= amount:
                    # Perform the transaction
                    success = self.economic_system.transfer_resource(agent1_id, agent2_id, resource_type, amount, price)
                    
                    if success:
                        daily_metrics['economic_transactions'] += 1
                        
                        # Update the relationship based on fair pricing
                        fair_price = self.economic_system.get_price(resource_type)
                        if fair_price > 0:
                            price_ratio = price / fair_price
                            if 0.8 <= price_ratio <= 1.2:  # Fair trade
                                self.update_relationship(agent1_id, agent2_id, "FAIR_TRADE", 0.1)
                            elif price_ratio < 0.8:  # Buyer got a good deal
                                self.update_relationship(agent1_id, agent2_id, "GOOD_DEAL_FOR_BUYER", 0.05)
                            else:  # Seller got a good deal
                                self.update_relationship(agent1_id, agent2_id, "GOOD_DEAL_FOR_SELLER", -0.05)
        
        # 3. Simulate resource conflicts
        if len(self.social_graph) >= 2:
            # Determine number of conflicts
            conflict_count = int(len(self.social_graph) * 0.05) + 1
            
            for _ in range(conflict_count):
                # Randomly select two agents
                agent1_id = random.choice(list(self.social_graph.keys()))
                agent2_id = random.choice([a for a in self.social_graph.keys() if a != agent1_id])
                
                # Randomly select a resource type
                resource_type = random.choice(list(ResourceType))
                
                # Determine amount
                amount = random.uniform(1.0, 20.0)
                
                # Resolve the conflict
                winner, winner_amount = self.resolve_resource_conflict(agent1_id, agent2_id, resource_type, amount)
                
                # Award resources to the winner
                self.economic_system.produce_resource(winner, resource_type, winner_amount)
                
                daily_metrics['conflict_resolutions'] += 1
                
                # Generate a significant event for the conflict
                event = {
                    'type': "RESOURCE_CONFLICT_RESOLUTION",
                    'agents': [agent1_id, agent2_id],
                    'winner': winner,
                    'resource_type': resource_type.value,
                    'amount': winner_amount,
                    'timestamp': time.time()
                }
                daily_metrics['significant_events'].append(event)
        
        # 4. Simulate group elections
        for group_id, group in self.social_groups.items():
            # Small chance for an election
            if random.random() < 0.05:
                # Only proceed if there are members
                if group.members:
                    # Get candidates (all members are eligible)
                    candidates = list(group.members.keys())
                    
                    # Hold the election
                    new_leader = self.elect_leader(candidates, candidates)
                    
                    if new_leader and new_leader != group.leader_id:
                        # Update group leadership
                        old_leader = group.leader_id
                        group.set_leader(new_leader)
                        
                        daily_metrics['elections'] += 1
                        
                        # Generate a significant event for the election
                        event = {
                            'type': "GROUP_LEADERSHIP_CHANGE",
                            'group_id': group_id,
                            'old_leader': old_leader,
                            'new_leader': new_leader,
                            'timestamp': time.time()
                        }
                        daily_metrics['significant_events'].append(event)
                        
                        # Update relationships within the group
                        for member_id in group.members:
                            if member_id != new_leader:
                                # Member->Leader relationship changes
                                rel = self.get_relationship(member_id, new_leader)
                                rel.relationship_type = RelationshipType.LEADER
                                
                                # Update former leader relationship
                                if old_leader and member_id != old_leader:
                                    old_rel = self.get_relationship(member_id, old_leader)
                                    if old_rel.relationship_type == RelationshipType.LEADER:
                                        old_rel.relationship_type = RelationshipType.NEUTRAL
        
        # 5. Simulate policy changes
        # Small chance to create a new policy
        if random.random() < 0.1 and self.social_graph:
            agent_id = random.choice(list(self.social_graph.keys()))
            
            policy_title = f"Policy {len(self.policies) + 1}"
            policy_description = f"A policy created by agent {agent_id}"
            
            # Decide if it's a group policy or society-wide
            if self.social_groups and random.random() < 0.5:
                # Find groups the agent is a member of
                agent_groups = [g_id for g_id, group in self.social_groups.items() if agent_id in group.members]
                if agent_groups:
                    group_id = random.choice(agent_groups)
                    policy = self.create_policy(policy_title, policy_description, agent_id, group_id)
                else:
                    policy = self.create_policy(policy_title, policy_description, agent_id)
            else:
                policy = self.create_policy(policy_title, policy_description, agent_id)
                
            daily_metrics['policy_changes'] += 1
            
            # Generate a significant event for the policy creation
            event = {
                'type': "POLICY_CREATION",
                'policy_id': policy.policy_id,
                'creator_id': agent_id,
                'group_id': policy.group_id,
                'timestamp': time.time()
            }
            daily_metrics['significant_events'].append(event)
        
        # 6. Simulate cultural meme spreading
        self.simulate_cultural_evolution(iterations=5)
        daily_metrics['meme_spreads'] += random.randint(0, 10)  # Simple approximation
        
        # 7. Update social network metrics
        self.update_social_metrics()
        
        # 8. Save the current state
        self.save()
        
        return daily_metrics
    
    def get_agent_social_status(self, agent_id: str) -> Dict[str, Any]:
        """Calculate an agent's social status and influence"""
        if agent_id not in self.social_graph:
            return {
                'agent_id': agent_id,
                'status': 0.0,
                'influence': 0.0,
                'connection_count': 0,
                'group_memberships': 0,
                'leadership_positions': 0,
                'identity_memberships': 0,
                'resource_wealth': 0.0,
                'relationship_quality': 0.0
            }
            
        # Basic network metrics
        connection_count = len(self.social_graph[agent_id])
        centrality = self.centrality_scores.get(agent_id, 0.0)
        
        # Group memberships and leadership positions
        group_memberships = []
        leadership_positions = []
        for group_id, group in self.social_groups.items():
            if agent_id in group.members:
                group_memberships.append(group_id)
                if group.leader_id == agent_id:
                    leadership_positions.append(group_id)
        
        # Identity memberships
        identity_memberships = []
        for identity_id, identity in self.social_identities.items():
            if agent_id in identity.members:
                identity_memberships.append(identity_id)
        
        # Resource wealth - sum of all resources
        resource_wealth = sum(
            self.economic_system.agent_resources[agent_id][resource_type]
            for resource_type in ResourceType
        )
        
        # Relationship quality - average strength of connections
        relationships = []
        for key, rel in self.social_relationships.items():
            if agent_id in key:
                relationships.append(rel)
                
        avg_relationship_quality = (
            sum(rel.strength for rel in relationships) / len(relationships)
            if relationships else 0.0
        )
        
        # Calculate status score
        # This is a simplified model - in a real implementation,
        # this would be more sophisticated
        status_factors = {
            'centrality': centrality * 0.3,
            'connections': min(connection_count / 10.0, 1.0) * 0.2,
            'leadership': len(leadership_positions) * 0.15,
            'group_membership': min(len(group_memberships) / 3.0, 1.0) * 0.1,
            'identity_membership': min(len(identity_memberships) / 3.0, 1.0) * 0.05,
            'wealth': min(resource_wealth / 100.0, 1.0) * 0.15,
            'relationship_quality': avg_relationship_quality * 0.05
        }
        
        status_score = sum(status_factors.values())
        
        # Calculate influence score
        influence_factors = {
            'centrality': centrality * 0.25,
            'connections': min(connection_count / 10.0, 1.0) * 0.2,
            'leadership': len(leadership_positions) * 0.25,
            'group_membership': min(len(group_memberships) / 3.0, 1.0) * 0.1,
            'identity_membership': min(len(identity_memberships) / 3.0, 1.0) * 0.05,
            'wealth': min(resource_wealth / 100.0, 1.0) * 0.1,
            'relationship_quality': avg_relationship_quality * 0.05
        }
        
        influence_score = sum(influence_factors.values())
        
        return {
            'agent_id': agent_id,
            'status': status_score,
            'influence': influence_score,
            'connection_count': connection_count,
            'group_memberships': len(group_memberships),
            'leadership_positions': len(leadership_positions),
            'identity_memberships': len(identity_memberships),
            'resource_wealth': resource_wealth,
            'relationship_quality': avg_relationship_quality,
            'status_factors': status_factors,
            'influence_factors': influence_factors
        }
    
    def get_society_health(self) -> Dict[str, Any]:
        """Calculate metrics for overall society health and functioning"""
        # Gather basic statistics
        stats = {
            'agent_count': len(self.social_graph),
            'relationship_count': len(self.social_relationships),
            'group_count': len(self.social_groups),
            'policy_count': len(self.policies),
            'active_policy_count': sum(1 for p in self.policies.values() if p.status == "active"),
            'meme_count': len(self.cultural_memes),
            'norm_count': len(self.social_norms),
            'identity_count': len(self.social_identities),
            'economic_transaction_count': self.economic_system.transaction_count,
            'total_resources': {str(rt.value): 0 for rt in ResourceType},
            'resource_distribution_gini': {},
            'social_cohesion': 0.0,
            'governance_effectiveness': 0.0,
            'cultural_diversity': 0.0,
            'economic_health': 0.0,
            'overall_health': 0.0
        }
        
        # Calculate total resources and distribution
        for resource_type in ResourceType:
            # Calculate total
            total = sum(
                self.economic_system.agent_resources[agent_id][resource_type]
                for agent_id in self.social_graph
            )
            stats['total_resources'][str(resource_type.value)] = total
            
            # Calculate Gini coefficient for resource distribution
            if self.social_graph and total > 0:
                values = [
                    self.economic_system.agent_resources[agent_id][resource_type]
                    for agent_id in self.social_graph
                ]
                stats['resource_distribution_gini'][str(resource_type.value)] = self._calculate_gini(values)
        
        # Calculate social cohesion
        if self.social_relationships:
            # Average relationship strength
            avg_strength = sum(rel.strength for rel in self.social_relationships.values()) / len(self.social_relationships)
            
            # Network density (ratio of actual connections to possible connections)
            possible_connections = len(self.social_graph) * (len(self.social_graph) - 1) / 2
            actual_connections = len(self.social_relationships)
            density = actual_connections / possible_connections if possible_connections > 0 else 0
            
            # Group membership overlap
            avg_group_membership = sum(len(group.members) for group in self.social_groups.values()) / len(self.social_graph) if self.social_groups and self.social_graph else 0
            
            # Social cohesion score
            stats['social_cohesion'] = (avg_strength * 0.4 + density * 0.4 + min(avg_group_membership / 2, 1.0) * 0.2)
        
        # Calculate governance effectiveness
        if self.policies:
            # Average policy compliance
            avg_compliance = sum(policy.compliance_rate for policy in self.policies.values()) / len(self.policies)
            
            # Active policy ratio
            active_ratio = stats['active_policy_count'] / stats['policy_count'] if stats['policy_count'] > 0 else 0
            
            # Leadership stability
            leadership_changes = sum(1 for event in self.social_relationships.values() if event.relationship_type == RelationshipType.LEADER)
            leadership_stability = 1.0 / (1.0 + leadership_changes / len(self.social_graph)) if self.social_graph else 0
            
            # Governance effectiveness score
            stats['governance_effectiveness'] = (avg_compliance * 0.4 + active_ratio * 0.3 + leadership_stability * 0.3)
        
        # Calculate cultural diversity
        if self.cultural_memes:
            # Count unique attributes across all memes
            all_attributes = set()
            for meme in self.cultural_memes.values():
                all_attributes.update(meme.attributes.keys())
                
            attribute_count = len(all_attributes)
            
            # Count unique combinations of attributes
            attribute_combinations = set()
            for meme in self.cultural_memes.values():
                attribute_combinations.add(frozenset(meme.attributes.items()))
                
            combination_count = len(attribute_combinations)
            
            # Adoption distribution
            adoption_counts = [len(meme.adopters) for meme in self.cultural_memes.values()]
            adoption_diversity = 1.0 - self._calculate_gini(adoption_counts) if adoption_counts else 0
            
            # Cultural diversity score
            stats['cultural_diversity'] = (
                min(attribute_count / 10.0, 1.0) * 0.3 +
                min(combination_count / (len(self.cultural_memes) * 0.5), 1.0) * 0.4 +
                adoption_diversity * 0.3
            )
        
        # Calculate economic health
        # Use various economic indicators
        inflation = abs(self.economic_system.inflation_rate)
        economic_stability = 1.0 / (1.0 + inflation * 10)  # Lower inflation is better
        
        transaction_volume = stats['economic_transaction_count'] / max(1, len(self.social_graph))
        trade_activity = min(transaction_volume / 10.0, 1.0)
        
        # Average resource Gini (lower is better for equality)
        avg_gini = sum(stats['resource_distribution_gini'].values()) / len(stats['resource_distribution_gini']) if stats['resource_distribution_gini'] else 0
        resource_equality = 1.0 - avg_gini
        
        # Economic health score
        stats['economic_health'] = (
            economic_stability * 0.4 +
            trade_activity * 0.3 +
            resource_equality * 0.3
        )
        
        # Calculate overall health
        stats['overall_health'] = (
            stats['social_cohesion'] * 0.3 +
            stats['governance_effectiveness'] * 0.25 +
            stats['cultural_diversity'] * 0.2 +
            stats['economic_health'] * 0.25
        )
        
        return stats
    
    def _calculate_gini(self, values: List[float]) -> float:
        """Calculate the Gini coefficient, a measure of inequality"""
        if not values or sum(values) == 0:
            return 0
            
        # Sort values
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        # Calculate Gini coefficient
        index = np.arange(1, n + 1)
        return (np.sum((2 * index - n - 1) * sorted_values)) / (n * np.sum(sorted_values))

# CLI for the Governance System
def main():
    """Main entry point for the Governance System"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SYNTHIX Governance System')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Initialize governance system
    init_parser = subparsers.add_parser('init', help='Initialize governance system')
    init_parser.add_argument('--universe', required=True, help='Universe ID')
    init_parser.add_argument('--name', default='Default Governance', help='Governance system name')
    
    # Simulate day command
    simulate_parser = subparsers.add_parser('simulate', help='Simulate a day')
    simulate_parser.add_argument('--universe', required=True, help='Universe ID')
    simulate_parser.add_argument('--days', type=int, default=1, help='Number of days to simulate')
    
    # Create group command
    group_parser = subparsers.add_parser('create_group', help='Create a social group')
    group_parser.add_argument('--universe', required=True, help='Universe ID')
    group_parser.add_argument('--name', required=True, help='Group name')
    group_parser.add_argument('--desc', required=True, help='Group description')
    group_parser.add_argument('--creator', required=True, help='Creator agent ID')
    
    # Create identity command
    identity_parser = subparsers.add_parser('create_identity', help='Create a social identity')
    identity_parser.add_argument('--universe', required=True, help='Universe ID')
    identity_parser.add_argument('--name', required=True, help='Identity name')
    identity_parser.add_argument('--desc', required=True, help='Identity description')
    identity_parser.add_argument('--creator', required=True, help='Creator agent ID')
    
    # Get society stats command
    stats_parser = subparsers.add_parser('stats', help='Get society statistics')
    stats_parser.add_argument('--universe', required=True, help='Universe ID')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == 'init':
        # Initialize governance system
        gov = GovernanceSystem(args.universe, args.name)
        gov.save()
        print(f"Initialized governance system for universe {args.universe}")
        
    elif args.command == 'simulate':
        # Load governance system
        gov = GovernanceSystem(args.universe)
        if not gov.load():
            print(f"Error: Governance system for universe {args.universe} not found")
            return
            
        # Simulate specified number of days
        for day in range(args.days):
            print(f"Simulating day {day+1}...")
            metrics = gov.simulate_day()
            print(f"  Interactions: {metrics['interactions']}")
            print(f"  Economic transactions: {metrics['economic_transactions']}")
            print(f"  Significant events: {len(metrics['significant_events'])}")
            
        print(f"Simulation complete. Governance system saved.")
        
    elif args.command == 'create_group':
        # Load governance system
        gov = GovernanceSystem(args.universe)
        if not gov.load():
            print(f"Error: Governance system for universe {args.universe} not found")
            return
            
        # Create group
        group = gov.create_group(args.name, args.desc, args.creator)
        gov.save()
        print(f"Created group '{args.name}' with ID {group.group_id}")
        
    elif args.command == 'create_identity':
        # Load governance system
        gov = GovernanceSystem(args.universe)
        if not gov.load():
            print(f"Error: Governance system for universe {args.universe} not found")
            return
            
        # Create identity with some default values
        values = {
            'tradition': random.random(),
            'progress': random.random(),
            'authority': random.random(),
            'equality': random.random(),
            'loyalty': random.random(),
            'independence': random.random()
        }
        
        identity = gov.create_identity(args.name, args.desc, values, args.creator)
        gov.save()
        print(f"Created identity '{args.name}' with ID {identity.identity_id}")
        
    elif args.command == 'stats':
        # Load governance system
        gov = GovernanceSystem(args.universe)
        if not gov.load():
            print(f"Error: Governance system for universe {args.universe} not found")
            return
            
        # Get statistics
        stats = gov.get_society_health()
        
        print(f"Society Statistics for {gov.name} (Universe {args.universe}):")
        print(f"  Population: {stats['agent_count']} agents")
        print(f"  Social connections: {stats['relationship_count']} relationships")
        print(f"  Groups: {stats['group_count']} social groups")
        print(f"  Policies: {stats['active_policy_count']} active out of {stats['policy_count']} total")
        print(f"  Cultural memes: {stats['meme_count']} memes")
        print()
        print(f"  Social cohesion: {stats['social_cohesion']:.2f}")
        print(f"  Governance effectiveness: {stats['governance_effectiveness']:.2f}")
        print(f"  Cultural diversity: {stats['cultural_diversity']:.2f}")
        print(f"  Economic health: {stats['economic_health']:.2f}")
        print()
        print(f"  Overall society health: {stats['overall_health']:.2f}")
        
    else:
        print("Error: Unknown command")
        parser.print_help()

if __name__ == "__main__":
    main()
