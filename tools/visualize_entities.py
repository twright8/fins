#!/usr/bin/env python3
"""
Entity visualization tool that creates a network graph from extracted entities.
Can be used for debugging or demonstrations.
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import networkx as nx
from pyvis.network import Network

def load_data(entities_file, relationships_file):
    """
    Load entities and relationships from JSON files.
    
    Args:
        entities_file: Path to entities JSON file
        relationships_file: Path to relationships JSON file
        
    Returns:
        tuple: (entities, relationships)
    """
    entities = []
    relationships = []
    
    if os.path.exists(entities_file):
        with open(entities_file, 'r') as f:
            entities = json.load(f)
    
    if os.path.exists(relationships_file):
        with open(relationships_file, 'r') as f:
            relationships = json.load(f)
    
    return entities, relationships

def create_entity_graph(entities, relationships, limit=50, min_mentions=1):
    """
    Create a network graph from entities and relationships.
    
    Args:
        entities: List of entity dictionaries
        relationships: List of relationship dictionaries
        limit: Maximum number of entities to include
        min_mentions: Minimum number of mentions for an entity to be included
        
    Returns:
        tuple: (graph, network)
    """
    # Filter entities by mention count
    filtered_entities = [e for e in entities if e.get('mention_count', 0) >= min_mentions]
    
    # Sort by mention count (descending) and limit
    top_entities = sorted(
        filtered_entities,
        key=lambda e: e.get('mention_count', 1),
        reverse=True
    )[:limit]
    
    # Create entity ID to text mapping
    entity_texts = {e.get('entity_id'): e.get('text') for e in top_entities}
    entity_text_to_id = {e.get('text'): e.get('entity_id') for e in top_entities}
    
    # Create network graph
    G = nx.DiGraph()
    
    # Node color map by type
    colors = {
        "person": "#3B82F6",  # Blue
        "organization": "#10B981",  # Green
        "location": "#F59E0B",  # Yellow
        "money": "#EC4899",  # Pink
        "product": "#8B5CF6",  # Purple
        "law": "#6B7280",  # Gray
        "norp": "#EF4444",  # Red
        "event": "#6366F1",  # Indigo
        "unknown": "#9CA3AF"  # Light gray
    }
    
    # Add nodes (entities)
    for entity in top_entities:
        entity_id = entity.get('entity_id')
        text = entity.get('text')
        entity_type = entity.get('type', 'unknown')
        mention_count = entity.get('mention_count', 1)
        
        # Add node to graph
        G.add_node(
            entity_id,
            label=text,
            type=entity_type,
            weight=mention_count,
            color=colors.get(entity_type, colors['unknown'])
        )
    
    # Filter relationships involving top entities
    filtered_relationships = []
    for rel in relationships:
        subject = rel.get('subject')
        obj = rel.get('object')
        
        if subject in entity_text_to_id and obj in entity_text_to_id:
            filtered_relationships.append({
                'subject_id': entity_text_to_id[subject],
                'object_id': entity_text_to_id[obj],
                'subject': subject,
                'object': obj,
                'type': rel.get('type', 'unknown'),
                'confidence': rel.get('confidence', 0)
            })
    
    # Add edges (relationships)
    for rel in filtered_relationships:
        G.add_edge(
            rel['subject_id'],
            rel['object_id'],
            type=rel['type'],
            confidence=rel['confidence'],
            subject=rel['subject'],
            object=rel['object']
        )
    
    # Create PyVis network
    net = Network(height="800px", width="100%", directed=True, notebook=False)
    
    # Add nodes
    for node_id, attrs in G.nodes(data=True):
        size = 10 + (attrs.get('weight', 1) * 3)  # Size based on mention count
        color = attrs.get('color', colors['unknown'])
        label = attrs.get('label', node_id)
        entity_type = attrs.get('type', 'unknown')
        
        title = f"<b>{label}</b><br>Type: {entity_type}<br>Mentions: {attrs.get('weight', 1)}"
        
        net.add_node(
            node_id,
            label=label,
            title=title,
            size=size,
            color=color
        )
    
    # Add edges
    for source, target, attrs in G.edges(data=True):
        width = 1 + (attrs.get('confidence', 0) * 5)  # Width based on confidence
        title = f"<b>{attrs.get('type', 'unknown')}</b><br>Confidence: {attrs.get('confidence', 0):.2f}"
        
        net.add_edge(
            source,
            target,
            title=title,
            width=width,
            label=attrs.get('type', '')
        )
    
    # Set physics options
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": {
          "enabled": true,
          "iterations": 1000,
          "updateInterval": 25
        }
      },
      "edges": {
        "smooth": {
          "type": "continuous",
          "forceDirection": "none"
        },
        "arrows": {
          "to": {
            "enabled": true,
            "scaleFactor": 0.5
          }
        },
        "color": {
          "inherit": true
        },
        "font": {
          "size": 10
        }
      },
      "nodes": {
        "font": {
          "size": 12,
          "face": "Roboto"
        }
      }
    }
    """)
    
    return G, net

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Entity Visualization Tool")
    parser.add_argument("-e", "--entities", default="data/extracted/entities.json", help="Path to entities JSON file")
    parser.add_argument("-r", "--relationships", default="data/extracted/relationships.json", help="Path to relationships JSON file")
    parser.add_argument("-o", "--output", default="entity_graph.html", help="Output HTML file")
    parser.add_argument("-l", "--limit", type=int, default=50, help="Maximum number of entities to include")
    parser.add_argument("-m", "--min-mentions", type=int, default=1, help="Minimum number of mentions for an entity")
    
    args = parser.parse_args()
    
    # Load data
    entities, relationships = load_data(args.entities, args.relationships)
    
    print(f"Loaded {len(entities)} entities and {len(relationships)} relationships")
    
    # Create graph
    G, net = create_entity_graph(entities, relationships, args.limit, args.min_mentions)
    
    print(f"Created graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
    
    # Save graph
    net.save_graph(args.output)
    
    print(f"Saved graph to {args.output}")

if __name__ == "__main__":
    main()
