#include "lib/SolutionCloneGraph.h"

// Default constructor
SolutionCloneGraph::Node::Node() {
    val = 0;
    neighbors = std::vector<Node*>();
}

// Constructor with a value
SolutionCloneGraph::Node::Node(int _val) {
    val = _val;
    neighbors = std::vector<Node*>();
}

// Constructor with a value and neighbors
SolutionCloneGraph::Node::Node(int _val, std::vector<Node*> _neighbors) {
    val = _val;
    neighbors = _neighbors;
}

// Clone graph function
SolutionCloneGraph::Node* SolutionCloneGraph::cloneGraph(Node* node) {
    if (!node)
        return nullptr;

    if (visited.find(node) != visited.end())
        return visited[node];

    Node* newNode = new Node(node->val);
    visited[node] = newNode;

    for (const auto& neighbor : node->neighbors) {
        newNode->neighbors.push_back(cloneGraph(neighbor));
    }

    return newNode;
}