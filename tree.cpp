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

using namespace std;

//100. Same Tree
bool isSameTree(TreeNode* p, TreeNode* q)
{

switch (1){ 

case 1:{ // DFS

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

case 2:{ // BFS

       if (!p && !q ) return true;

       if (!p || !q) return false;

      //BFS
      queue<TreeNode*> queueP;
      queue<TreeNode*> queueQ;

      queueP.push(p);
      queueQ.push(q);

      while (!queueP.empty()  && !queueQ.empty()  ) {

      p = queueP.front();
      q = queueQ.front();
      queueP.pop();
      queueQ.pop();

      if (p->val != q->val)
            return false;

      if (p->left && q->left){
            queueP.push(p->left);
            queueQ.push(q->left);
      }
      else if (p->left || q->left)
            return false;

      if (p->right && q->right){
            queueP.push(p->right);
            queueQ.push(q->right);
      }
      else if (p->right || q->right)
      return false;

      }

      return queueP.empty()  && queueQ.empty();

}

case 3:{ // recursive 
     
        if (!p && !q )
            return true;

      if (!p || !q )
            return false;

      bool left = isSameTree(p->left,q->left);
      bool right = isSameTree(p->right,q->right);

      if (p->val != q->val)
            return false;

      return left && right;
}

}
}
