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



class PatternsToSolveLeetCode8
{

public:
      // Sliding Window
      int maxSumSubarray(const vector<int>& nums, int k)
      {
            int n = nums.size();
            if (n < k)
                  return -1; // Invalid if array size is smaller than k

            int windowSum = 0;
            for (int i = 0; i < k; i++)
            {
                  windowSum += nums[i];
            }

            int maxSum = windowSum;

            for (int i = k; i < n; i++)
            {
                  windowSum += nums[i] - nums[i - k];
                  maxSum = max(maxSum, windowSum);
            }

            return maxSum;
      }

      // Subset pattern
      void findSubsets(const vector<int>& nums, vector<int>& current, int index, vector<vector<int>>& result)
      {

            if (index == nums.size())
            {
                  result.push_back(current);
            }
            current.push_back(nums[index]);

            findSubsets(nums, current, index + 1, result);

            current.pop_back();
            findSubsets(nums, current, index + 1, result);
      }

      // Modified Binary Search
      int search(vector<int>& nums, int target)
      {

            int size = nums.size();
            int left = 0;
            int right = size - 1;

            while (left <= right)
            {
                  int mid = left + (right - left) / 2; //  == (left + right) /2

                  if (nums[mid] == target)
                        return mid;

                  if (nums[left] <= nums[mid])                          // left sorted can do binary search
                        if (nums[left] <= target && target < nums[mid]) // on left
                              right = mid - 1;
                        else
                              left = mid + 1; // on right
                  else                        // right sorted
                        if (nums[mid] < target && target <= nums[right])
                              left = mid + 1;
                        else
                              right = mid - 1;
            }
            return -1;
      }
      // K largest from vector
      vector<int> kLargestElements(vector<int>& nums, int k)
      {

            vector<int> result(k);
            priority_queue<int, vector<int>, greater<int>> minHeap;

            for (const auto& num : nums)
            {
                  minHeap.push(num);

                  if (minHeap.size() > k)
                        minHeap.pop();
            }

            while (!minHeap.empty())
            {
                  result.push_back(minHeap.top());
                  minHeap.pop();
            }
            return result;
      }

      bool hasCycle(int course, vector<vector<int>>& graph, vector<int>& visited)
      {
            // a cycle mean we can't finish like lock.
            // we check for cycles for current DFS path .
            //2 state need if diamond shape we will detect circle incorrectly  
            visited[course] = 1; // visitng now
            
            // Visit all neighbors BFS
            for (int next : graph[course])
            {
                  if (visited[next] == 1) // cycle detected
                        return true;
                  
                  if (visited[next] == 0) 
                        if (hasCycle(next, graph, visited))
                           return true; 
            }

            visited[course] = 2; // mark as visited after traveresring.  so we won't traverse it again.
            // on all descendants no cycle detected at this point we ma
            

            return false;
      }

      bool canFinish(int numCourses, vector<vector<int>>& prerequisites)
      {
            // 0 = not visited, 1 = visiting (in current DFS path), 2 = visited
            vector<int> visited(numCourses, 0);  //if visited = cycle

      
            vector<vector<int>> graph(numCourses);

            //build graph adjacency list based on prerequisites             
            for (const auto& pre : prerequisites)
                  graph[pre[1]].push_back(pre[0]);
            
            //check if graph Directed Acyclic Graph Check for cycles using DFS
                  for (int i = 0; i < numCourses; i++) // every course starting point 
                        if (visited[i] == 0)
                              if (hasCycle(i, graph, visited)) //  independent subgraphs every course can be a starting point 
                                    return false;

            return true;
      }

      // traverse on indegree BFS
      vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {

            vector<int> inDegree(numCourses,0);
            
            vector<vector<int>> adjList(numCourses);
            
            vector<int> res;

            for (auto pre:prerequisites){
                  adjList[pre[1]].push_back(pre[0]);
                  inDegree[pre[0]]++;
            }

            queue<int> queue;
            for (int i = 0 ; i < numCourses; i ++) //starying points
                   if (inDegree[i] == 0)
                        queue.push(i);

            while(!queue.empty()){
                        
                  int course = queue.front();
                  queue.pop();
                  res.push_back(course);
                  
                  for (int next:adjList[course]){
                        inDegree[next]--;
                        if (inDegree[next] == 0) 
                              queue.push(next);
                  }
            }

            return (res.size() != numCourses) ? vector<int>{} : res  ;
      }



      vector<vector<int>> subsets(vector<int>& nums)
      {
            vector<vector<int>> result;
            vector<int> curr;
            findSubsets(nums, curr, 0, result);
            return result;
      }

      vector<vector<int>> levelOrder(TreeNode* root)
      {

            vector<vector<int>> result;
            if (!root)
                  return result;

            queue<TreeNode*> queue;

            queue.push(root);

            while (!queue.empty())
            {

                  int size = queue.size();
                  vector<int> level;

                  for (int i = 0; i < size; i++)
                  {
                        TreeNode* curr = queue.front();
                        level.push_back(curr->val);
                        queue.pop();
                        if (curr->left)
                              queue.push(curr->left);
                        if (curr->right)
                              queue.push(curr->right);
                  }

                  result.push_back(level);
            }
            return result;
      }
};