# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 15:25:58 2022

@author: ALEXRB
"""

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import random


def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.  
    Licensed under Creative Commons Attribution-Share Alike 
    
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    
    G: the graph (must be a tree)
    
    root: the root node of current branch 
    - if the tree is directed and this is not given, 
      the root will be found and used
    - if the tree is directed and this is given, then 
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given, 
      then a random choice will be used.
    
    width: horizontal space allocated for this branch - avoids overlap with other branches
    
    vert_gap: gap between levels of hierarchy
    
    vert_loc: vertical location of root
    
    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''
    
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos

            
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


class TreeNode:
    def __init__(self, parent=None):
        self.parent = parent
    
    def fitness(self):
        return 1.


class Branch:
    def __init__(self, nodes, parent=None):
        self.parent = parent
        self.nodes = nodes


class BnB:
    def __init__(self, roots):
        
        self.sol_graph = nx.DiGraph()
        self.sol_graph.add_node('root')
        edges = [('root', i) for i in roots]
        self.sol_graph.add_edges_from(edges)
    
    def create_branch(self, parent, nodes):
        edges = [(parent, i) for i in nodes]
        self.sol_graph.add_edges_from(edges)
        
        for node in nodes:
            print(node, nx.ancestors(self.sol_graph, node))
    
    def draw_tree(self):
        print(self.sol_graph.nodes())
        nx.draw(self.sol_graph,
                hierarchy_pos(self.sol_graph, 'root'),
                with_labels=True)
    
    
pn = [0, 1, 2, 3, 4]
bnb = BnB(pn)
pn2 = [p+len(pn) for p in pn]
bnb.create_branch(pn[2], pn2)
pn3 = [p+len(pn2) for p in pn2]
bnb.create_branch(pn2[0], pn3)
bnb.draw_tree()