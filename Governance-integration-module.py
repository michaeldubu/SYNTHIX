#!/usr/bin/env python3
"""
SYNTHIX OS Governance Integration Module

This module integrates the governance system with the Agent Runtime Environment (ARE)
and Universe Simulation Engine (USE) to create a comprehensive AI civilization.

It enables:
1. Agents to form social relationships and participate in governance
2. Resource generation and allocation through the economic system
3. Cultural evolution through meme creation and transmission
4. Societal development and emergence of complex social structures
"""

import os
import sys
import json
import time
import logging
import subprocess
import threading
import random
from typing import Dict, List, Any, Tuple, Set

# Import governance system
sys.path.append('/usr/lib/synthix')
from governance_system import GovernanceSystem, SocialRelationship, SocialGroup, ResourceType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('/var/log/synthix/governance_integration.log'), logging.StreamHandler()]
)
logger = logging.getLogger('GOVERNANCE_INTEGRATION')

class SynthixGovernanceIntegration:
    """Main class for integrating the governance system with SYNTHIX OS components"""
    
    def __init__(self, universe_id: str):
        self.universe_id = universe_id
        self.governance_system = GovernanceSystem(universe_id)
        
        # Load governance system or create if it doesn't exist
        if not self.governance_system.load():
            logger.info(f"Creating new governance system for universe {universe_id}")
            self.governance_system.save()
        
        # Integration state
        self.agent_properties = {}  # Dictionary of agent_id -> properties dictionary
        self.integration_active = False
        self.integration_thread = None
        self.last_simulation_time = 0
        
        # Paths for integration
        self.are_socket_path = f"/var/lib/synthix/sockets/{universe_id}_are"
        self.use_socket_path = f"/var/lib/synthix/sockets/{universe_id}_use"
        
        # Create socket directory if it doesn't exist
        os.makedirs(os.path.dirname(self.are_socket_path), exist_ok=True)
    
    def start(self):
        """Start the governance integration"""
        if self.integration_active:
            logger.warning(f"Governance integration for universe {self.universe_id} already running")
            return False
            
        logger.info(f"Starting governance integration for universe {self.universe_id}")
        self.integration_active = True
        
        # Start integration thread
        self.integration_thread = threading.Thread(target=self._integration_loop)
        self.integration_thread.daemon = True
        self.integration_thread.start()
        
        return True
    
    def stop(self):
        """Stop the governance integration"""
        if not self.integration_active:
            logger.warning(f"Governance integration for universe {self.universe_id} not running")
            return False
            
        logger.info(f"Stopping governance integration for universe {self.universe_id}")
        self.integration_active = False
        
        # Wait for integration thread to end
        if self.integration_thread and self.integration_thread.is_alive():
            self.integration_thread.join(timeout=5.0)
            
        return True
    
    def _integration_loop(self):
        """Main integration loop"""
        logger.info(f"Governance integration loop started for universe {self.universe_id}")
        
        while self.integration_active:
            try:
                # 1. Fetch current agents from ARE
                self._update_agent_roster()
                
                # 2. Process agent activities and generate social interactions
                self._process_agent_activities()
                
                # 3. Simulate governance and society
                current_time = time.time()
                # Simulate a day every 15 minutes of real time
                if current_time - self.last_simulation_time > 900:  # 15 minutes
                    self._simulate_society()
                    self.last_simulation_time = current_time
                
                # Sleep to prevent excessive CPU usage
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in governance integration loop: {e}")
                time.sleep(10)  # Sleep longer after an error
                
        logger.info(f"Governance integration loop ended for universe {self.universe_id}")
    
    def _update_agent_roster(self):
        """Update the list of agents from the Agent Runtime Environment"""
        try:
            # In a real implementation, this would use IPC with the ARE service
            # For this demonstration, we'll use a simpler approach
            
            # Get agent list by calling the ARE CLI
            cmd = ["/usr/lib/synthix/are.py", "agent", "list", "--universe", self.universe_id, "--json"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Failed to get agent list: {result.stderr}")
                return
                
            # Parse the JSON output
            agents_data = json.loads(result.stdout)
            
            # Update our internal agent list
            for agent_data in agents_data:
                agent_id = agent_data['agent_id']
                
                # Store/update agent properties
                self.agent_properties[agent_id] = agent_data
                
                # Ensure the agent is represented in the governance system
                if agent_id not in self.governance_system.social_graph:
                    logger.info(f"Adding new agent {agent_id} to governance system")
                    
                    # Add to social graph
                    self.governance_system.social_graph[agent_id] = set()
                    
                    # Initialize agent resources
                    for resource_type in ResourceType:
                        # Give some initial resources
                        initial_amount = random.uniform(5.0, 15.0)
                        self.governance_system.economic_system.produce_resource(agent_id, resource_type, initial_amount)
                    
                    # Calculate initial centrality
                    self.governance_system.centrality_scores[agent_id] = 0.0
                    
                    # Save the updated governance system
                    self.governance_system.save()
            
            # Check for agents that have been removed
            removed_agents = set(self.agent_properties.keys()) - set(agent_data['agent_id'] for agent_data in agents_data)
            for agent_id in removed_agents:
                logger.info(f"Removing agent {agent_id} from governance system")
                
                # Remove from agent properties
                del self.agent_properties[agent_id]
                
                # Note: We'll keep the agent in the governance system for history purposes
                # but mark them as inactive by removing their connections
                
                # Remove from others' social connections
                for other_agent_id in self.governance_system.social_graph:
                    if agent_id in self.governance_system.social_graph[other_agent_id]:
                        self.governance_system.social_graph[other_agent_id].remove(agent_id)
                
                # Clear their own connections
                if agent_id in self.governance_system.social_graph:
                    self.governance_system.social_graph[agent_id] = set()
                
                # Save the updated governance system
                self.governance_system.save()
                
        except Exception as e:
            logger.error(f"Error updating agent roster: {e}")
    
    def _process_agent_activities(self):
        """Process recent agent activities and generate social interactions"""
        try:
            # In a real implementation, this would monitor agent actions from the ARE
            # and USE services, and translate them into governance interactions
            
            # For this demonstration, we'll generate some random interactions
            if len(self.agent_properties) < 2:
                return  # Need at least 2 agents for interactions
                
            # Small chance to generate an interaction each cycle
            if random.random() < 0.3:
                # Select two random agents
                agent_ids = list(self.agent_properties.keys())
                if len(agent_ids) < 2:
                    return
                    
                agent1_id = random.choice(agent_ids)
                agent2_id = random.choice([a for a in agent_ids if a != agent1_id])
                
                # Generate random interaction
                interaction_types = [
                    "CONVERSATION", "COLLABORATION", "TRADE", "DISPUTE", 
                    "TEACHING", "LEARNING", "ASSISTANCE", "COMPETITION"
                ]
                interaction_type = random.choice(interaction_types)
                
                # Generate random outcome (-1.0 to 1.0)
                interaction_outcome = random.random() * 2 - 1.0
                
                # Update the relationship
                self.governance_system.update_relationship(agent1_id, agent2_id, interaction_type, interaction_outcome)
                
                logger.debug(f"Generated interaction: {agent1_id} <-> {agent2_id}, Type: {interaction_type}, Outcome: {interaction_outcome:.2f}")
                
                # Process economic interaction if it's a trade
                if interaction_type == "TRADE" and interaction_outcome > 0:
                    # Random resource type
                    resource_type = random.choice(list(ResourceType))
                    
                    # Random amount and price
                    amount = random.uniform(0.1, 5.0)
                    price = random.uniform(0.5, 3.0)
                    
                    # Randomly determine buyer and seller
                    if random.random() < 0.5:
                        seller_id, buyer_id = agent1_id, agent2_id
                    else:
                        seller_id, buyer_id = agent2_id, agent1_id
                    
                    # Ensure seller has enough resources
                    if self.governance_system.economic_system.agent_resources[seller_id][resource_type] >= amount:
                        # Execute the trade
                        self.governance_system.economic_system.transfer_resource(seller_id, buyer_id, resource_type, amount, price)
                        logger.debug(f"Trade executed: {seller_id} sold {amount:.2f} of {resource_type.name} to {buyer_id} for {price:.2f}")
                
                # Save changes
                self.governance_system.save()
                
        except Exception as e:
            logger.error(f"Error processing agent activities: {e}")
    
    def _simulate_society(self):
        """Run a simulation step for the society"""
        try:
            logger.info(f"Simulating society for universe {self.universe_id}")
            
            # Simulate a day in the society
            metrics = self.governance_system.simulate_day()
            
            logger.info(f"Society simulation complete: {metrics['interactions']} interactions, " + 
                       f"{metrics['economic_transactions']} economic transactions, " +
                       f"{len(metrics['significant_events'])} significant events")
            
            # Check for governance changes that need to be communicated to agents
            self._process_governance_changes(metrics)
            
            # Update society health metrics
            health_metrics = self.governance_system.get_society_health()
            logger.info(f"Society health: Overall: {health_metrics['overall_health']:.2f}, " +
                       f"Social: {health_metrics['social_cohesion']:.2f}, " +
                       f"Governance: {health_metrics['governance_effectiveness']:.2f}, " +
                       f"Economy: {health_metrics['economic_health']:.2f}")
            
        except Exception as e:
            logger.error(f"Error simulating society: {e}")
    
    def _process_governance_changes(self, metrics: Dict[str, Any]):
        """Process governance changes and notify agents"""
        # In a real implementation, this would communicate governance changes
        # back to the agents through the ARE service
        
        # Process elections
        if metrics['elections'] > 0:
            # Find election events
            election_events = [event for event in metrics['significant_events'] 
                              if event['type'] == "GROUP_LEADERSHIP_CHANGE"]
            
            for event in election_events:
                group_id = event['group_id']
                new_leader = event['new_leader']
                old_leader = event.get('old_leader')
                
                if group_id in self.governance_system.social_groups:
                    group = self.governance_system.social_groups[group_id]
                    logger.info(f"Leadership change in group '{group.name}': " +
                              f"{old_leader if old_leader else 'None'} -> {new_leader}")
                    
                    # In a real implementation, this would notify agents about the change
        
        # Process policy changes
        if metrics['policy_changes'] > 0:
            # Find policy events
            policy_events = [event for event in metrics['significant_events'] 
                            if event['type'] == "POLICY_CREATION"]
            
            for event in policy_events:
                policy_id = event['policy_id']
                creator_id = event['creator_id']
                group_id = event.get('group_id')
                
                if policy_id in self.governance_system.policies:
                    policy = self.governance_system.policies[policy_id]
                    scope = f"group '{self.governance_system.social_groups[group_id].name}'" if group_id else "society-wide"
                    logger.info(f"New policy '{policy.title}' created by {creator_id} ({scope})")
                    
                    # In a real implementation, this would notify agents about the new policy
    
    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get comprehensive status information for an agent"""
        status = {
            'agent_id': agent_id,
            'in_governance_system': agent_id in self.governance_system.social_graph,
            'properties': self.agent_properties.get(agent_id, {}),
            'social_status': None,
            'relationships': [],
            'group_memberships': [],
            'resource_holdings': {}
        }
        
        if agent_id in self.governance_system.social_graph:
            # Get social status metrics
            status['social_status'] = self.governance_system.get_agent_social_status(agent_id)
            
            # Get relationship info
            for key, rel in self.governance_system.social_relationships.items():
                if agent_id in key:
                    other_id = key[0] if key[1] == agent_id else key[1]
                    relationship_info = {
                        'other_agent': other_id,
                        'type': rel.relationship_type.name,
                        'strength': rel.strength,
                        'trust': rel.trust,
                        'interactions': rel.interaction_count
                    }
                    status['relationships'].append(relationship_info)
            
            # Get group memberships
            for group_id, group in self.governance_system.social_groups.items():
                if agent_id in group.members:
                    group_info = {
                        'group_id': group_id,
                        'name': group.name,
                        'role': group.members[agent_id],
                        'is_leader': group.leader_id == agent_id
                    }
                    status['group_memberships'].append(group_info)
            
            # Get resource holdings
            for resource_type in ResourceType:
                status['resource_holdings'][resource_type.name] = self.governance_system.economic_system.agent_resources[agent_id][resource_type]
        
        return status
    
    def get_society_status(self) -> Dict[str, Any]:
        """Get comprehensive status information for the society"""
        # Get basic society health metrics
        status = self.governance_system.get_society_health()
        
        # Add additional information
        status['agent_statuses'] = {}
        for agent_id in self.agent_properties:
            agent_social_status = self.governance_system.get_agent_social_status(agent_id)
            status['agent_statuses'][agent_id] = {
                'status': agent_social_status['status'],
                'influence': agent_social_status['influence'],
                'connection_count': agent_social_status['connection_count']
            }
        
        # Calculate most influential agents
        influential_agents = sorted(
            [(agent_id, data['influence']) for agent_id, data in status['agent_statuses'].items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]  # Top 5
        
        status['most_influential_agents'] = influential_agents
        
        # Calculate most active groups
        active_groups = sorted(
            [(group_id, len(group.members)) for group_id, group in self.governance_system.social_groups.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]  # Top 5
        
        status['most_active_groups'] = [
            {
                'group_id': group_id,
                'name': self.governance_system.social_groups[group_id].name,
                'member_count': member_count,
                'leader': self.governance_system.social_groups[group_id].leader_id
            }
            for group_id, member_count in active_groups
        ]
        
        # Calculate most popular cultural memes
        popular_memes = sorted(
            [(meme_id, len(meme.adopters)) for meme_id, meme in self.governance_system.cultural_memes.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]  # Top 5
        
        status['most_popular_memes'] = [
            {
                'meme_id': meme_id,
                'name': self.governance_system.cultural_memes[meme_id].name,
                'adopter_count': adopter_count,
                'creator': self.governance_system.cultural_memes[meme_id].creator_id
            }
            for meme_id, adopter_count in popular_memes
        ]
        
        return status

# CLI for the Governance Integration
def main():
    """Main entry point for the Governance Integration"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SYNTHIX Governance Integration')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Start integration
    start_parser = subparsers.add_parser('start', help='Start governance integration')
    start_parser.add_argument('--universe', required=True, help='Universe ID')
    
    # Stop integration
    stop_parser = subparsers.add_parser('stop', help='Stop governance integration')
    stop_parser.add_argument('--universe', required=True, help='Universe ID')
    
    # Agent status
    agent_parser = subparsers.add_parser('agent_status', help='Get agent status')
    agent_parser.add_argument('--universe', required=True, help='Universe ID')
    agent_parser.add_argument('--agent', required=True, help='Agent ID')
    
    # Society status
    society_parser = subparsers.add_parser('society_status', help='Get society status')
    society_parser.add_argument('--universe', required=True, help='Universe ID')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == 'start':
        # Start integration
        integration = SynthixGovernanceIntegration(args.universe)
        if integration.start():
            print(f"Started governance integration for universe {args.universe}")
        else:
            print(f"Failed to start governance integration for universe {args.universe}")
            
    elif args.command == 'stop':
        # Stop integration
        integration = SynthixGovernanceIntegration(args.universe)
        if integration.stop():
            print(f"Stopped governance integration for universe {args.universe}")
        else:
            print(f"Failed to stop governance integration for universe {args.universe}")
            
    elif args.command == 'agent_status':
        # Get agent status
        integration = SynthixGovernanceIntegration(args.universe)
        status = integration.get_agent_status(args.agent)
        print(json.dumps(status, indent=2))
        
    elif args.command == 'society_status':
        # Get society status
        integration = SynthixGovernanceIntegration(args.universe)
        status = integration.get_society_status()
        print(json.dumps(status, indent=2))
        
    else:
        print("Error: Unknown command")
        parser.print_help()

if __name__ == "__main__":
    main()
