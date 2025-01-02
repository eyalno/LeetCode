#ifndef BST_H
#define BST_H



class BST {
public:
    TreeNode* root;

    BST() : root(nullptr) {}

    bool search(int val)
    {
        return _search(root, val) != nullptr;
    }

    TreeNode* _search(TreeNode* node, int val)
    {
        if (node == nullptr || node->val == val)
            return node;

        if (val < node->val)
            return _search(node->left, val);
        else
            return _search(node->right, val);
    }

    void insert(int val)
    {
        root = _insert(root, val);
    }

    TreeNode* _insert(TreeNode* node, int val)
    {
        if (node == nullptr)
            return new TreeNode(val);

        if (val < node->val)
            node->left = _insert(node->left, val);
        else
            node->right = _insert(node->right, val);

        return node;
    }

    void deleteNode(int key)
    {
        root = _delete(root, key);
    }

    TreeNode* _delete(TreeNode* root, int key)
    {
        if (root == nullptr)
            return root;

        if (key < root->val)
            root->left = _delete(root->left, key);
        else if (key > root->val)
            root->right = _delete(root->right, key);
        else
        {
            if (root->left == nullptr)
            {
                TreeNode* temp = root->right;
                delete root;
                return temp;
            }
            else if (root->right == nullptr)
            {
                TreeNode* temp = root->left;
                delete root;
                return temp;
            }

            TreeNode* temp = root->right;
            while (temp->left != nullptr)
                temp = temp->left;

            root->val = temp->val;
            root->right = _delete(root->right, temp->val);
        }
        return root;
    }
};

#endif // BST_H