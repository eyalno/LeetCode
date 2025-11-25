#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <map>
#include <queue>
#include <unordered_set>
#include <set>
#include <utility>
#include <bitset>
#include <stack>
#include <array>
#include <sstream>
#include <iomanip>

#include "lib/TreeNode.h"
#include "lib/ListNode.h"
#include "lib/Trie.h"
#include "lib/DisJointSet.h"
#include "lib/LinkedList.h"
#include "lib/binarySearchTree.h"
#include "lib/GraphNode.h"

using namespace std;

//100. Same Tree
bool isSameTree(TreeNode* p, TreeNode* q)
{
      switch (1) {

      case 1: { // DFS

            if (!p && !q)
                  return true;

            if (!p || !q)
                  return false;

            stack<TreeNode*> stackP;
            stack<TreeNode*> stackQ;

            stackP.push(p);
            stackQ.push(q);

            while (!stackP.empty() && !stackQ.empty())
            {

                  TreeNode* nodeP = stackP.top();
                  TreeNode* nodeQ = stackQ.top();

                  stackP.pop();
                  stackQ.pop();

                  if (nodeP->val != nodeQ->val)
                        return false;

                  if (nodeP->left && nodeQ->left)
                  {
                        stackP.push(nodeP->left);
                        stackQ.push(nodeQ->left);
                  }
                  else if (nodeP->left || nodeQ->left)
                        return false;

                  if (nodeP->right && nodeQ->right)
                  {
                        stackP.push(nodeP->right);
                        stackQ.push(nodeQ->right);
                  }
                  else if (nodeP->right || nodeQ->right)
                        return false;
            }

            return stackP.empty() && stackQ.empty();
      }

      case 2: { // BFS

            if (!p && !q) return true;

            if (!p || !q) return false;

            //BFS
            queue<TreeNode*> queueP;
            queue<TreeNode*> queueQ;

            queueP.push(p);
            queueQ.push(q);

            while (!queueP.empty() && !queueQ.empty()) {

                  p = queueP.front();
                  q = queueQ.front();
                  queueP.pop();
                  queueQ.pop();

                  if (p->val != q->val)
                        return false;

                  if (p->left && q->left) {
                        queueP.push(p->left);
                        queueQ.push(q->left);
                  }
                  else if (p->left || q->left)
                        return false;

                  if (p->right && q->right) {
                        queueP.push(p->right);
                        queueQ.push(q->right);
                  }
                  else if (p->right || q->right)
                        return false;

            }

            return queueP.empty() && queueQ.empty();

      }

      case 3: { // recursive 

            if (!p && !q)
                  return true;

            if (!p || !q)
                  return false;

            bool left = isSameTree(p->left, q->left);
            bool right = isSameTree(p->right, q->right);

            if (p->val != q->val)
                  return false;

            return left && right;
      }

      }
}

//543. Diameter of Binary Tree
class DiameterSolution
{

private:
      int maxDiameter = 0;

      int diameterOfBinaryTreeHelper(TreeNode* root)
      {

            if (root == nullptr)
                  return 0;

            int leftHeight = diameterOfBinaryTreeHelper(root->left);
            int rightHeight = diameterOfBinaryTreeHelper(root->right);

            maxDiameter = max(maxDiameter, leftHeight + rightHeight);

            return max(leftHeight, rightHeight) + 1;
      }

      int diameterOfBinaryTree(TreeNode* root)
      {

            /* 1 recursion not 2
            if (root == nullptr)
                  return 0;

            diameterOfBinaryTreeHelper(root);

      return maxDiameter;
      */

      /**/
            if (root == nullptr)
                  return 0;

            // height of each subtree = edges to the root
            int leftHeight = treeHeight(root->left);
            int rightHeight = treeHeight(root->right);

            int diameterThroughRoot = leftHeight + rightHeight;

            return max(max(diameterOfBinaryTree(root->left), diameterOfBinaryTree(root->right)), diameterThroughRoot);
      }

      int treeHeight(TreeNode* root)
      {
            if (root == nullptr)
                  return 0;

            return max(treeHeight(root->left), treeHeight(root->right)) + 1;
      }
};


// Encodes a tree to a single string.
string serialize(TreeNode* root)
{
      if (root == nullptr)
            return "null";
      string str = to_string(root->val) + ",";

      str += serialize(root->left);
      str += ",";
      str += serialize(root->right);

      return str;
}

int treeHeight(TreeNode* root)
{

      if (root == nullptr)
            return 0;

      return max(treeHeight(root->left), treeHeight(root->right)) + 1;
}



//110. Balanced Binary Tree
bool isBalancedHelper(TreeNode* root, int& height);
bool isBalanced(TreeNode* root)
{
      // Bottom up

      int height;

      if (root == nullptr)
            return true;

      return isBalancedHelper(root, height);

      /*top down
            if(root == nullptr)
                  return true;

            int leftHeight = treeHeight(root->left);
            int rightHeight = treeHeight(root->right);

            if (abs (leftHeight - rightHeight) > 1  )
                  return false;

            return isBalanced(root->left) && isBalanced(root->right);
            */
}


bool isBalancedHelper(TreeNode* root, int& height)
{
      if (root == nullptr)
      {
            height = 0;
            return true;
      }

      int leftHeight;
      int rightHeight;

      bool bLeftBalanced = isBalancedHelper(root->left, leftHeight);
      bool bRightBalanced = isBalancedHelper(root->right, rightHeight);

      if (bLeftBalanced && bRightBalanced && abs(leftHeight - rightHeight) < 2)
      {
            height = max(leftHeight, rightHeight) + 1;
            return true;
      }
      return false;
}


//236. Lowest Common Ancestor of a Binary Tree
class LCA
{
public:
      TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q)
      {

            /* //recursvie
             if (root == nullptr || root == p || root == q  )
                   return root;
             int rVal = root->val;
             int pVal = p->val;
             int qVal = q->val;



             if ( (pVal > rVal && qVal < rVal)  || (pVal < rVal && qVal > rVal) )
                   return root;

             if ( (pVal > rVal && qVal > rVal)   )
                   return  lowestCommonAncestor(root->right,p,q);
             else
                   return  lowestCommonAncestor(root->left,p,q);

             */

             // Iterative
            if (root == nullptr || root == p || root == q)
                  return root;
            int pVal = p->val;
            int qVal = q->val;

            while (root != nullptr)
            {

                  if (root == p || root == q)
                        return root;

                  int rVal = root->val;

                  if ((pVal > rVal && qVal < rVal) || (pVal < rVal && qVal > rVal))
                        return root;

                  if ((pVal > rVal && qVal > rVal))
                        root = root->right;
                  else
                        root = root->left;
            }
            return root;
      }
};

/*
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {

// Itertative  we will keep track of all parents

//Binary Tree not BST
stack<TreeNode*> stack;

unordered_map<TreeNode *, TreeNode*> parent;

stack.push(root);

parent[root] = nullptr;

// Traverse the tree until we find both nodes p and q
while ( !(parent.count(p) == 1 && parent.count(q) == 1)   ){

      TreeNode * node = stack.top();
      stack.pop();

      if (node->left){
            parent[node->left] = node;
            stack.push(node->left);
      }

      if (node->right){
            parent[node->right] = node;
            stack.push(node->right);
      }
}

 // Now we have parent pointers for both p and q,
 // traverse up from p and mark visited nodes
unordered_set<TreeNode*> ancestors;

//populate all the way up to root
while (p){
      ancestors.insert(p);
      p = parent[p];
}

while (ancestors.count(q) == 0)
{
      q= parent[q];

}
return q;

//Traverse up from q until we find the first common ancestor
*/

// Binary Tree not BST

// base case
/*recursive
if (root == nullptr || root == q || root == p)
      return root;

TreeNode * leftLCA =  lowestCommonAncestor(root->left,p,q);
TreeNode * rightLCA = lowestCommonAncestor(root->right,p,q);

if (leftLCA && rightLCA )
      return root;



return leftLCA == nullptr ? rightLCA:leftLCA;
}
*/



void connectHelper(Node* left, Node* right);
Node* connect(Node* root)
{
      // not a prefect tree

      if (!root)
            return root;

      Node* levelStart = root;

      while (levelStart != nullptr)
      {

            Node* curr = levelStart; // start of each level
            Node* prev = nullptr;
            Node* nextLevelStart = nullptr;

            while (curr != nullptr)
            {
                  if (curr->left != nullptr)
                  {
                        if (prev != nullptr)
                              prev->next = curr->left;
                        else
                              nextLevelStart = curr->left;

                        prev = curr->left;
                  }

                  if (curr->right != nullptr)
                  {
                        if (prev != nullptr)
                              prev->next = curr->right;
                        else
                              nextLevelStart = curr->right;

                        prev = curr->right;
                  }

                  curr = curr->next;
            }

            levelStart = nextLevelStart;
      }

      // no queue . create connection from level above
      /*
      if (!root)
            return nullptr;

      Node* leftMost = root;

      // last level
      while (leftMost->left){

            Node* curr = leftMost;

            while (curr){

                  //1 connection next level left -> right
                  curr->left->next = curr->right;

                  //2 connection next level right ->left
                  if (curr->next)
                      curr->right->next = curr->next->left;
                  //next node in level
                  curr = curr->next;
            }
            //next level
            leftMost = leftMost->left;
      }

      return root;
      */
      /*recrsive
      if (!root)
           return root;

      connectHelper(root->left,root->right);
      return root;
      */

      /* queue
      if (!root   )
            return root;

      queue<Node *> queue;

      queue.push(root);

      while (!queue.empty())
      {

            int size = queue.size();
            for (int i = 0 ; i < size; i++){

                  Node * curr = queue.front();
                  queue.pop();

                  if (i < size - 1) {
                      curr->next = queue.front();
                  }

                  if (curr->left){
                        queue.push(curr->left);
                  }

                  if (curr->right){
                        queue.push(curr->right);
                  }
            }
      }

      return root;

      */
      return root;
}

void connectHelper(Node* left, Node* right)
{

      if (!left || !right)
            return;

      left->next = right;

      connectHelper(left->left, left->right);
      connectHelper(left->right, right->left);
      connectHelper(right->left, right->right);
}


class BuildTree
{
public:
      TreeNode* buildTreePost(vector<int>& inorder, vector<int>& postorder)
      {
            // Store the indices of inorder elements for quick lookup
            unordered_map<int, int> inorder_map;
            for (int i = 0; i < inorder.size(); ++i)
            {
                  inorder_map[inorder[i]] = i;
            }
            return buildTreeHelper(inorder, postorder, 0, inorder.size() - 1, 0, postorder.size() - 1, inorder_map);
      }

      TreeNode* buildTreeHelper(vector<int>& inorder, vector<int>& postorder, int inStart, int inEnd, int postStart, int postEnd, unordered_map<int, int>& inorder_map)
      {
            if (inStart > inEnd || postStart > postEnd)
                  return nullptr;

            // Create the root node from the last element of postorder
            TreeNode* root = new TreeNode(postorder[postEnd]);

            // Find the index of the root value in the inorder traversal
            int rootIndexInInorder = inorder_map[root->val];
            int leftSubtreeSize = rootIndexInInorder - inStart;

            // Recursively build left and right subtrees
            root->left = buildTreeHelper(inorder, postorder, inStart, rootIndexInInorder - 1, postStart, postStart + leftSubtreeSize - 1, inorder_map);
            root->right = buildTreeHelper(inorder, postorder, rootIndexInInorder + 1, inEnd, postStart + leftSubtreeSize, postEnd - 1, inorder_map);

            return root;
      }

      TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder)
      {
            // Store the indices of inorder elements for quick lookup
            unordered_map<int, int> inorder_map;
            for (int i = 0; i < inorder.size(); ++i)
            {
                  inorder_map[inorder[i]] = i;
            }

            return buildTreeHelperPre(preorder, inorder, 0, preorder.size() - 1, 0, inorder.size() - 1, inorder_map);
      }

      TreeNode* buildTreeHelperPre(vector<int>& preorder, vector<int>& inorder, int preStart, int preEnd, int inStart, int inEnd, unordered_map<int, int>& inorder_map)
      {
            if (inStart > inEnd || preStart > preEnd)
                  return nullptr;

            // Create the root node from the last element of postorder
            TreeNode* root = new TreeNode(preorder[preStart]);

            // Find the index of the root value in the inorder traversal
            int rootIndexInInorder = inorder_map[root->val];
            int leftSubtreeSize = rootIndexInInorder - inStart;

            // Recursively build left and right subtrees
            root->left = buildTreeHelperPre(preorder, inorder, preStart + 1, preStart + leftSubtreeSize, inStart, rootIndexInInorder - 1, inorder_map);
            root->right = buildTreeHelperPre(preorder, inorder, preStart + leftSubtreeSize + 1, preEnd, rootIndexInInorder + 1, inEnd, inorder_map);

            return root;
      }
};


/*
Global count
int count = 0;
bool countUnivalSubtreesHelper(TreeNode* root) {

      if (!root)
            return true;

      bool isLeftSub = countUnivalSubtreesHelper(root->left);
      bool isRightSub = countUnivalSubtreesHelper(root->right);

      if (isLeftSub && isRightSub){
            if ( root->left && root->left->val != root->val)
                  return false;

            if (root->right && root->right->val != root->val)
                  return false;

              count++;
             return true;

      }


    return false;



}
 */

bool countUnivalSubtreesHelper(TreeNode* root, int& count)
{

      if (!root)
            return true;

      bool isLeftSub = countUnivalSubtreesHelper(root->left, count);
      bool isRightSub = countUnivalSubtreesHelper(root->right, count);

      if (isLeftSub && isRightSub)
      {
            if (root->left && root->left->val != root->val)
                  return false;

            if (root->right && root->right->val != root->val)
                  return false;

            count++;
            return true;
      }

      return false;
}

/* pair return value
pair<bool,int> countUnivalSubtreesHelper(TreeNode* root) {

      if (!root)
            return {true,0} ;


      auto left = countUnivalSubtreesHelper(root->left);
      auto right = countUnivalSubtreesHelper(root->right);
      int count = left.second +right.second;

      if (left.first && right.first){

            if ( root->left && root->left->val != root->val)
                  return {false,count};

            if (root->right && root->right->val != root->val)
                  return {false,count};


             return {true, count+1};


      }
      return {false,count};
}
*/


int countUnivalSubtrees(TreeNode* root)
{

      if (!root)
            return 0;

      int count = 0;

      countUnivalSubtreesHelper(root, count);
      return count;
}


bool hasPathSum(TreeNode* root, int targetSum)
{
      // iterative stack
      if (!root)
            return false; // If the tree is empty, there's no path

      stack<pair<TreeNode*, int>> nodeStack;
      nodeStack.push({ root, root->val }); // Initialize the stack with the root node and its value

      while (!nodeStack.empty())
      {
            auto topElem = nodeStack.top();

            nodeStack.pop();

            if (!topElem.first->left && !topElem.first->right && targetSum == topElem.second)
                  return true;

            if (topElem.first->left)
                  nodeStack.push({ topElem.first->left, topElem.second + topElem.first->left->val });

            if (topElem.first->right)
                  nodeStack.push({ topElem.first->right, topElem.second + topElem.first->right->val });
      }

      return false;

      /* recursive
      if (!root)
            return false;

      targetSum -= root->val;

      if (!root->left && !root->right)
            return targetSum == 0;

      return (hasPathSum(root->left,targetSum) || hasPathSum(root->right,targetSum));
      */
}

//checking is level symmetrry . propgate up 
bool isMirror(TreeNode* t1, TreeNode* t2)
{

      if (t1 == nullptr && t2 == nullptr)
            return true;

      if (t1 == nullptr || t2 == nullptr)
            return false;

      return t1->val == t2->val && isMirror(t1->left, t2->right) && isMirror(t1->right, t2->left);
}


bool isSymmetric(TreeNode* root)
{
      // return isMirror(root,root);

      // iteratrive queue every 2 consecutive nodes should be equal

      queue<TreeNode*> queue;

      queue.push(root->left);
      queue.push(root->right);

      while (!queue.empty())
      {

            TreeNode* t1 = queue.front();
            queue.pop();
            TreeNode* t2 = queue.front();
            queue.pop();

            if (!t1 && !t2)
                  continue;

            if (!t1 || !t2)
                  return false;
            if (t1->val != t2->val)
                  return false;

            queue.push(t1->left);
            queue.push(t2->right);
            queue.push(t1->right);
            queue.push(t2->left);
      }

      /*//iterative vector
      vector<TreeNode*> row;

      row.push_back(root);

      while (!row.empty()){

            int rowSize = row.size();
            for (int i = 0; i < rowSize ; i++){

                  TreeNode * node = row[i];
                  if (!node)
                        continue;

                  row.push_back(node->left);

                  row.push_back(node->right);
            }
            row.erase(row.begin(),row.begin() +rowSize);


            for (int i = 0, j = row.size() -1 ; i <j; i++, j--){
                  if(row[i] == nullptr && row[j] == nullptr)
                        continue;


                  if(row[i] == nullptr && row[j] != nullptr)
                        return false;

                  if(row[j] == nullptr && row[i] != nullptr)
                        return false;


                  if ( row[i]->val != row[j]->val)
                        return false;
            }


      }*/

      return true;
}


//226. Invert Binary Tree
TreeNode* invertTree(TreeNode* root)
{
      // BFS

      if (!root)
            return root;

      queue<TreeNode*> queue;

      queue.push(root);
      // we add to queue row by row
      while (!queue.empty())
      {

            int size = queue.size();

            for (int i = 0; i < size; i++)
            {

                  TreeNode* node = queue.front();
                  queue.pop();
                  TreeNode* temp = node->left;
                  node->left = node->right;
                  node->right = temp;

                  if (node->left)
                        queue.push(node->left);

                  if (node->right)
                        queue.push(node->right);
            }
      }

      return root;

      /* recrurssion
      if (!root)
            return nullptr;

            invertTree(root->left);
            invertTree(root->right);
            TreeNode* temp = root->left;
            root->left = root->right;
            root->right = temp;

      return root;
      */
}


//104. Maximum Depth of Binary Tree
int maxDepth(TreeNode* root)
{

      /*      //DFS
            if (!root)
                  return 0;

               return    max( maxDepth(root->left) +1 ,maxDepth(root->right) +1   );
      */

      // itertive
      stack<pair<int, TreeNode*>> stack;

      if (!root)
            return 0;

      stack.push({ 1, root });

      int maxDepth = 0;
      while (!stack.empty())
      {
            // depth, node
            auto stackTop = stack.top();

            stack.pop();
            maxDepth = max(stackTop.first, maxDepth);

            if (stackTop.second->left)
                  stack.push({ stackTop.first + 1, stackTop.second->left });

            if (stackTop.second->right)
                  stack.push({ stackTop.first + 1, stackTop.second->right });
      }
      return maxDepth;
}


void levelOrderHelper(TreeNode* root, int level, vector<vector<int>>& result)
{

      if (!root)
            return;

      if (result.size() <= level)
      {
            result.push_back(vector<int>{});
      }
      // BFS inside recursion
      result[level].push_back(root->val);

      levelOrderHelper(root->left, level + 1, result);
      levelOrderHelper(root->right, level + 1, result);
}

//102. Binary Tree Level Order Traversal
void levelOrderHelper(TreeNode* root, int level, vector<vector<int>>& result);
vector<vector<int>> levelOrder(TreeNode* root)
{

      vector<vector<int>> result;

      if (!root)
            return result;
      // levelOrderHelper(root, 0, result);

      // BFS
      //
      queue<TreeNode*> queue;

      queue.push(root);
      // we add to queue row by row
      while (!queue.empty())
      {

            int size = queue.size();
            vector<int> level;

            for (int i = 0; i < size; i++)
            {

                  TreeNode* node = queue.front();
                  queue.pop();
                  level.push_back(node->val);

                  if (node->left)
                        queue.push(node->left);

                  if (node->right)
                        queue.push(node->right);
            }

            result.push_back(level);
      }

      return result;
}


//145. Binary Tree Postorder Traversal
class Solution
{
public:
      void postOrderTraversalHelper(TreeNode* root, vector<int>& result)
      {

            if (!root)
                  return;

            postOrderTraversalHelper(root->left, result);
            postOrderTraversalHelper(root->right, result);
            result.push_back(root->val);
      }

      vector<int> postorderTraversal(TreeNode* root)
      {

            vector<int> result;
            postOrderTraversalHelper(root, result);

            return result;
      }

      void inOrderTraversalHelper(TreeNode* root, vector<int>& result)
      {

            if (!root)
                  return;

            inOrderTraversalHelper(root->left, result);
            result.push_back(root->val);
            inOrderTraversalHelper(root->right, result);
      }

      vector<int> inorderTraversal(TreeNode* root)
      {

            vector<int> result;
            // inOrderTraversalHelper(root,result);

            stack<TreeNode*> stack;
            TreeNode* current = root;

            if (!root)
                  return result;

            while (current || !stack.empty())
            {

                  while (current)
                  {
                        stack.push(current);
                        current = current->left;
                  }

                  current = stack.top();
                  stack.pop();

                  result.push_back(current->val);

                  current = current->right;
            }

            return result;
      }

      void preorderTraversalHelper(TreeNode* root, vector<int>& result)
      {

            if (!root)
                  return;

            result.push_back(root->val);
            preorderTraversalHelper(root->left, result);
            preorderTraversalHelper(root->right, result);
      }

      vector<int> preorderTraversal(TreeNode* root)
      {

            vector<int> result;
            stack<TreeNode*> stack;

            if (!root)
                  return result;

            stack.push(root);

            while (!stack.empty())
            {

                  TreeNode* node = stack.top();
                  stack.pop();

                  result.push_back(node->val);

                  if (node->right)
                        stack.push(node->right);
                  if (node->left)
                        stack.push(node->left);
            }

            /// preorderTraversalHelper(root,result);

            return result;
      }
};