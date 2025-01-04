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



ListNode* middleNodeHelper(ListNode* slow, ListNode* fast)
{

      if (!fast || !fast->next)
            return slow;

      return middleNodeHelper(slow->next, fast->next->next);
}

ListNode* middleNode(ListNode* head)
{

      // recursive
      if (!head || !head->next)
            return head;

      return middleNodeHelper(head, head->next);

      /*slow fast pointers
      ListNode* slow = head ,*fast = head;


      while (fast && fast->next){

            fast = fast->next->next;

            slow = slow->next;
      }

      return slow;
      */
}

  ListNode* rotateRight(ListNode* head, int k)
      {

            if (!head)
                  return nullptr;

            int size = 1;
            ListNode* curr = head;

            while (curr->next)
            {
                  size++;
                  curr = curr->next;
            }
            k = k % size;

            curr->next = head;
            curr = head;
            for (int i = 0; i < size - k - 1; i++, curr = curr->next)
                  ;

            head = curr->next;
            curr->next = nullptr;

            return head;
      }

       Node* getClonedNode(Node* oldNode, unordered_map<Node*, Node*>* hashMap)
      {

            if (oldNode == nullptr)
                  return nullptr;

            if (hashMap->find(oldNode) == hashMap->end())
                  hashMap->insert({ oldNode, new Node(oldNode->val) });

            return (*hashMap)[oldNode];
      }

       Node* copyRandomList(Node* head)
      {

            // Weave the two lists so not O(N) space needed

            if (!head)
                  return head;

            Node* curr = head;
            Node* newHead = nullptr;

            while (curr)
            {

                  Node* newNode = new Node(curr->val);

                  newNode->next = curr->next;

                  curr->next = newNode;

                  curr = newNode->next;
            }

            /// set random
            curr = head;
            while (curr)
            {
                  curr->next->random = curr->random ? curr->random->next : nullptr;

                  curr = curr->next->next;
            }

            // unweave
            curr = head;
            newHead = curr->next;
            Node* newCurr = newHead;
            while (curr)
            {

                  curr->next = newCurr->next;
                  curr = curr->next;

                  if (curr)
                  {
                        newCurr->next = curr->next;
                        newCurr = curr->next;
                  }
                  else
                        newCurr->next = nullptr;
            }

            return newHead;

            /*//recursive
            if (visited.find(head) !=  visited.end()  )
                  return visited[head];


            Node * newNode = new Node(head->val);
            visited.insert({head,newNode});

            newNode->next = copyRandomList(head->next);
            newNode->random = copyRandomList(head->random);

            return newNode;
            */

            /*  using hasemap iterative
                  if (!head)
                        return head;

            unordered_map<Node*,Node* > * hashMap = new unordered_map<Node*,Node*>;

            Node* oldNode = head;

            Node* newNode = getClonedNode(oldNode,hashMap);

            Node* newHead = newNode;

            while (oldNode){

            newNode = getClonedNode(oldNode,hashMap);

            newNode->next = getClonedNode(oldNode->next,hashMap);
            newNode->random = getClonedNode(oldNode->random,hashMap);

                  newNode = newNode->next;
                  oldNode= oldNode->next;
            }

            return newHead;
              */
      }

       Node* insertCircularList(Node* head, int insertVal)
      {

            Node* newNode = new Node(insertVal);

            if (!head)
            {
                  newNode->next = newNode;
                  return newNode;
            }

            Node* curr = head->next, * prev = head;

            // case 1  3 <i < 5
            while (!(prev->val <= insertVal && insertVal <= curr->val))
            {

                  if (prev->val > curr->val)
                  {
                        // case 2   9 < 10 < 1
                        if (prev->val <= insertVal && insertVal >= curr->val)
                              break;

                        // case 2.1 9 < 0 < 1
                        if (prev->val >= insertVal && insertVal <= curr->val)
                              break;
                  }
                  // case 3 same values closed a loop
                  if (head == curr)
                        break;

                  prev = curr;
                  curr = curr->next;
            }

            prev->next = newNode;
            newNode->next = curr;

            return head;
      }


      /* return the tail of the flatten list */
      Node* flattenDFS(Node* prev, Node* curr)
      {
            if (!curr)
                  return prev;

            prev->next = curr;
            curr->prev = prev;

            Node* temp = curr->next;
            Node* tail = flattenDFS(curr, curr->child);
            curr->child = nullptr;

            return flattenDFS(tail, temp);
      }

      Node* flatten(Node* head)
      {

            if (!head)
                  return nullptr;

            Node* sentinal = new Node(0);

            sentinal->next = head;

            Node* prev = sentinal;
            stack<Node*> stack;
            stack.push(head);

            while (!stack.empty())
            {

                  Node* curr = stack.top();
                  stack.pop();
                  prev->next = curr;
                  curr->prev = prev;

                  if (curr->next)
                  {
                        stack.push(curr->next);
                  }
                  if (curr->child)
                  {
                        stack.push(curr->child);
                        curr->child = nullptr;
                  }

                  prev = curr;
            }

            /* recursive
            flattenDFS(sentinal,head);
            */

            Node* tmp = sentinal->next;
            tmp->prev = nullptr;
            delete sentinal;
            return tmp;
      }

      ListNode* addTwoNumbers(ListNode* l1, ListNode* l2)
      {

            ListNode* head = new ListNode(-1);
            ListNode* curr = head;

            int carry = 0;
            while (l1 || l2 || carry != 0)
            {

                  int x = (l1 == nullptr) ? 0 : l1->val;
                  int y = (l2 == nullptr) ? 0 : l2->val;

                  int sum = (carry + x + y);
                  carry = sum / 10;

                  curr->next = new ListNode(sum % 10);

                  curr = curr->next;

                  l1 = l1 ? l1->next : nullptr;
                  l2 = l2 ? l2->next : nullptr;
            }

            return head->next;
      }

      ListNode* mergeTwoLists(ListNode* list1, ListNode* list2)
      {

            // recursive

            if (!list1)
                  return list2;
            if (!list2)
                  return list1;

            if (list1->val <= list2->val)
            {
                  list1->next = mergeTwoLists(list1->next, list2);
                  return list1;
            }
            else
            {
                  list2->next = mergeTwoLists(list1, list2->next);
                  return list2;
            }

            /*.Iterative
            //assume we are just inserting the l2 values to l1

            ListNode * sentinal = new ListNode(-1);

            ListNode * prev = sentinal;

            while(list1  && list2){

                  if(list1->val <= list2->val){
                        prev->next = list1;
                        list1 = list1->next;
                  }else{
                        prev->next = list2;
                        list2 = list2->next;
                  }
                  prev= prev->next;
            }

            if (!list1)
                  prev->next = list2;
            else
                 prev->next = list1;


            return sentinal->next;
            */
      }
     ListNode* reverseList(ListNode* head);
      bool isPalindrome(ListNode* head)
      {

            // 2 pointers

            ListNode* slow = head, * fast = head;

            if (!head)
                  return false;

            while (fast->next && fast->next->next)
            {

                  fast = fast->next->next;
                  slow = slow->next;
            }

            ListNode* headSec = reverseList(slow->next);

            slow->next = nullptr;

            ListNode* first = head;
            ListNode* sec = headSec;

            bool ret = true;

            while (first && sec)
            {
                  if (first->val != sec->val)
                  {
                        ret = false;
                        break;
                  }
                  first = first->next;
                  sec = sec->next;
            }

            headSec = reverseList(headSec);
            slow->next = headSec;

            return ret;

            // recursive;

            /*
                    bool recCheck(ListNode * curr){

                    if (!curr)
                        return true;

                     if (!recCheck(curr->next))
                        return false;

                    if (curr->val != front ->val)
                        return false;

                    front = front->next;

                    return true;


                }

            */

            /*
            // using vec
                  vector<int> vec;
                  ListNode * curr= head;


                  if (!head)
                        return false;


                  while (curr){
                        vec.push_back(curr->val);
                        curr =curr->next;
                  }

                  for (int i =0 , j = vec.size()-1; i <j ; i++,j-- ){
                        if (vec[i] != vec[j]){
                              return false;
                        }
                  }


                  return true;
            */
      }

      ListNode* oddEvenList(ListNode* head)
      {

            if (!head)
                  return nullptr;

            ListNode* odd = head;
            ListNode* even = head->next;
            ListNode* evenHead = even;

            // minimum step
            while (even && even->next)
            {

                  odd->next = even->next;
                  odd = odd->next;
                  even->next = odd->next;
                  even = even->next;
            }

            odd->next = evenHead;
            return head;
      }

      ListNode* removeElements(ListNode* head, int val)
      {

            ListNode* sentinel = new ListNode(0);
            sentinel->next = head;
            ListNode* prev, * curr, * toDel = nullptr;
            prev = sentinel;
            curr = sentinel->next;

            while (curr)
            {
                  if (curr->val == val)
                  {
                        prev->next = curr->next;
                        delete curr;
                        curr = prev->next;
                  }
                  else
                  {
                        curr = curr->next;
                        prev = prev->next;
                  }
            }

            ListNode* ret = sentinel->next;

            delete sentinel;

            return ret;
      }

      ListNode* reverseList(ListNode* head)
      {

            // iterative

            /*
                  ListNode * prev = nullptr ;
                  ListNode * curr = head; ;
                  ListNode * next  ;

                  while (curr){

                        next = curr->next;
                        curr->next = prev;
                        prev =curr;
                        curr = next;
                  }

                  return prev;
            */

            // recursive

            if (!head || !head->next)
                  return head;

            ListNode* p = reverseList(head->next);
            head->next->next = head;
            head->next = nullptr;

            return p;
      }

      ListNode* removeNthFromEnd(ListNode* head, int n)
      {

            // one pass algorithem

            ListNode* ptr1 = head;
            ListNode* ptr2 = head;

            for (int i = 0; i < n; i++)
                  ptr1 = ptr1->next;

            if (ptr1 == nullptr)
                  return head->next;

            while (ptr1->next != nullptr)
            {
                  ptr1 = ptr1->next;
                  ptr2 = ptr2->next;
            }

            ptr2->next = ptr2->next->next;

            return head;

            // two pass algorithem
            /*  int size= 0;

              for (ListNode * curr = head; curr != nullptr; curr= curr->next)
                    size++;

              if (size == n){
                    return head->next;
              }

              ListNode * beforeEle = head;

              for (int i =0; i < size - n -1 ; i++)
                    beforeEle = beforeEle->next;

              beforeEle->next = beforeEle->next->next;

              return head;


        return head;
              */

              // two pass algorithem V2
      }