#ifndef TREENODE_H
#define TREENODE_H

// TreeNode class definition
class TreeNode {
public:
    int val;
    TreeNode* left;
    TreeNode* right;

    // Default constructor for TreeNode
    TreeNode() : val(0), left(nullptr), right(nullptr) {}

    // Constructor for TreeNode with a value
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}

    // Constructor for TreeNode with a value, left, and right children
    TreeNode(int x, TreeNode* left, TreeNode* right) 
        : val(x), left(left), right(right) {}
};

#endif // TREENODE_H