#ifndef GRAPHNODE_HPP
#define GRAPHNODE_HPP

#include <vector>
using namespace std;

class GraphNode
{
public:
      int val;                           // Value of the node
      vector<GraphNode*> neighbors;      // List of neighboring nodes

      // Default constructor
      GraphNode()
            : val(0), neighbors(vector<GraphNode*>()) {
      }

      // Constructor with value
      GraphNode(int _val)
            : val(_val), neighbors(vector<GraphNode*>()) {
      }

      // Constructor with value and neighbors
      GraphNode(int _val, vector<GraphNode*> _neighbors)
            : val(_val), neighbors(_neighbors) {
      }
};

#endif // GRAPHNODE_HPP