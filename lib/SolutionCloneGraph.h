#ifndef SOLUTION_CLONE_GRAPH_H
#define SOLUTION_CLONE_GRAPH_H

#include <vector>
#include <unordered_map>

class SolutionCloneGraph {
public:
    class Node {
    public:
        int val;
        std::vector<Node*> neighbors;

        Node();
        Node(int _val);
        Node(int _val, std::vector<Node*> _neighbors);
    };

    std::unordered_map<Node*, Node*> visited;

    Node* cloneGraph(Node* node);
};

#endif // SOLUTION_CLONE_GRAPH_H