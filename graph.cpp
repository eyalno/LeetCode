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




//1971. Find if Path Exists in Graph
bool validPathDfs(int curr, int dest, unordered_map<int, vector<int>>& graph, unordered_set<int>& visited);
bool validPath(int n, vector<vector<int>>& edges, int source, int destination)
{
      switch (1) {

      case 1: { //DFS recursion
            unordered_map<int, vector<int>> graph;

            for (const auto& edge : edges)
            {
                  int u = edge[0];
                  int v = edge[1];
                  graph[u].push_back(v);
                  graph[v].push_back(u);
            }

            unordered_set<int> visited;

            return validPathDfs(source, destination, graph, visited);
      }
      case 2: { //stack

            unordered_map<int, vector<int>> graph;

            for (const auto& edge : edges)
            {
                  int u = edge[0];
                  int v = edge[1];
                  graph[u].push_back(v);
                  graph[v].push_back(u);
            }

            stack<int> st;
            unordered_set<int> visited;

            st.push(source);

            while (!st.empty())
            {

                  int curr = st.top();
                  st.pop();

                  if (curr == destination)
                        return true;

                  if (visited.find(curr) != visited.end())
                        continue;

                  visited.insert(curr);

                  for (int neighbor : graph[curr])
                        st.push(neighbor);
            }

            return false;

      }
      }
}

bool validPathDfs(int curr, int dest, unordered_map<int, vector<int>>& graph, unordered_set<int>& visited)
{

      if (curr == dest)
            return true;
      visited.insert(curr);

      for (int neighbor : graph[curr])
      {

            if (visited.find(neighbor) == visited.end())
                  if (validPathDfs(neighbor, dest, graph, visited))
                        return true;
      }

      return false;
}

//133. Clone Graph
unordered_map<GraphNode*, GraphNode*> visited;

GraphNode* cloneGraph(GraphNode* node)
{
      if (!node)
            return nullptr;

      if (visited.find(node) != visited.end())
            return visited[node];

      GraphNode* newNode = new GraphNode(node->val);
      visited[node] = newNode;

      for (const auto& neighbor : node->neighbors)
      {
            newNode->neighbors.push_back(cloneGraph(neighbor));
      }

      return newNode;
}


//797. All Paths From Source to Target // Backtracking
void allPathsDfs(int curr, int dest, vector<vector<int>>& graph, vector<vector<int>>& ret, vector<int>& path);

vector<vector<int>> allPathsSourceTarget(vector<vector<int>>& graph)
{
      vector<vector<int>> result;

      vector<int> path{ 0 };

      allPathsDfs(0, graph.size() - 1, graph, result, path);

      return result;
}

void allPathsDfs(int curr, int dest, vector<vector<int>>& graph, vector<vector<int>>& ret, vector<int>& path)
{
      if (curr == dest)
      {
            ret.push_back(path);
            return;
      }

      for (const auto& neighbor : graph[curr])
      {

            path.push_back(neighbor);

            allPathsDfs(neighbor, dest, graph, ret, path);
            path.pop_back();
      }

      return;
}


//332. Reconstruct Itinerary
class FindItinerarySolution
{
public:
      vector<string> findItinerary(vector<vector<string>>& tickets)
      {
            for (const auto& ticket : tickets)
            {
                  graph[ticket[0]].push_back(ticket[1]);
            }

            for (auto& pair : graph)
            {
                  sort(pair.second.begin(), pair.second.end());
                  visited[pair.first] = vector<bool>(pair.second.size(), false);
            }
            ticketCount = tickets.size();
            dfs("JFK");
            return result;
      }

private:
      unordered_map<string, vector<string>> graph;
      unordered_map<string, vector<bool>> visited;
      vector<string> result;
      bool bFound = false;
      int ticketCount = 0;

      bool dfs(const string& from)
      {
            result.push_back(from);

            if (result.size() == ticketCount + 1)
            {
                  bFound = true;
                  return bFound;
            }

            auto& dest = graph[from];
            auto& bitMap = visited[from];

            for (int i = 0; i < dest.size(); i++)
            {
                  /*   if (bitMap[i] == false ){
                           bitMap[i] = true;
                           string   next  = dest[i];
                           if  (dfs(next) )
                            return true;
                            bitMap[i] = false;
                     }
                    */

                    /* dest.erase(dest.begin()+i);
                     if  (dfs(next) )
                           return true;
                    dest.insert(dest.begin()+i,next);
                    */
            }

            result.pop_back();
            return bFound;
      }
};

//1059. All Paths from Source Lead to D
class LeadsToDestinationSolution
{
public:
      bool leadsToDestination(int n, vector<vector<int>>& edges, int source, int destination)
      {

            graph.resize(n);
            visited.resize(n, false);

            for (const auto& edge : edges)
            {
                  graph[edge[0]].push_back(edge[1]);
            }
            return dfs(source, destination);
      }

private:
      vector<bool> visited;

      bool dfs(int curr, int dest)
      {

            // If the node has no outgoing edges, it must be the destination
            if (graph[curr].empty())
                  return curr == dest;

            // If the node is currently being visited, a cycle is detected
            if (visited[curr] == true)
                  return false;
            visited[curr] = true;

            for (auto& neighb : graph[curr])
            {
                  if (!dfs(neighb, dest))
                        return false;
            }
            visited[curr] = false;
            return true;
      }

      vector<vector<int>> graph;
};


//200. Number of Islands
class NumIslandsSolution
{
public:
      int numIslands(vector<vector<char>>& grid)
      {
            m = grid.size();
            n = grid[0].size();
            islands = 0;

            for (int i = 0; i < m; i++)
                  for (int j = 0; j < n; j++)
                        if (grid[i][j] == '1')
                        {
                              islands++;
                              dfs(grid, i, j);
                        }

            return islands;
      }

private:
      void dfs(vector<vector<char>>& grid, int i, int j)
      {

            grid[i][j] = '0';

            if (i + 1 < m && grid[i + 1][j] == '1')
                  dfs(grid, i + 1, j);
            if (j + 1 < n && grid[i][j + 1] == '1')
                  dfs(grid, i, j + 1);
            if (i - 1 >= 0 && grid[i - 1][j] == '1')
                  dfs(grid, i - 1, j);
            if (j - 1 >= 0 && grid[i][j - 1] == '1')
                  dfs(grid, i, j - 1);
            return;
      }

      int islands;
      int m;
      int n;
};


//310. Minimum Height Trees
class FindMinHeightTreesSolution
{
public:
      vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges)
      {
            if (n == 1)
                  return { 0 };

            numOfNodes = n;
            vector<vector<int>> graph;
            vector<int> result;

            graph.resize(n);
            // build graph  for dfs
            for (const auto& edge : edges)
            {
                  graph[edge[0]].push_back(edge[1]);
                  graph[edge[1]].push_back(edge[0]);
            }
            // vector<bool> visited(n,false);
            int minHeight = n;
            for (int i = 0; i < n; i++)
            {

                  int height = treeHeight(i, -1, graph);

                  if (height < minHeight)
                  {
                        minHeight = height;
                        result.clear();
                        result.push_back(i);
                  }
                  else if (height == minHeight)
                  {
                        result.push_back(i);
                  }
            }

            return result;
      }

private:
      int numOfNodes;
      int treeHeight(int node, int parent, vector<vector<int>>& graph)
      {

            int depth = 0;
            for (const auto& leaf : graph[node])
            {
                  if (leaf != parent)
                  {
                        depth = max(depth, 1 + treeHeight(leaf, node, graph));
                  }
            }
            return depth;
      }
};


class SocialNetwork
{
private:
      // A map of user name to their friends (set ensures no duplicate friends)
      unordered_map<string, unordered_set<string>> network;

public:
      // Add a new user to the network
      void addUser(const string& user)
      {
            if (network.find(user) == network.end())
            {
                  network[user] = unordered_set<string>();
                  cout << user << " added to the network.\n";
            }
            else
            {
                  cout << user << " already exists in the network.\n";
            }
      }

      // Add a friendship between two users
      void addFriendship(const string& user1, const string& user2)
      {
            if (network.find(user1) == network.end() || network.find(user2) == network.end())
            {
                  cout << "Both users must exist in the network.\n";
                  return;
            }

            // Add the friendship (bidirectional)
            network[user1].insert(user2);
            network[user2].insert(user1);
            cout << "Friendship added between " << user1 << " and " << user2 << ".\n";
      }

      // Display a user's friends
      void displayUser(const string& user) const
      {
            if (network.find(user) == network.end())
            {
                  cout << user << " does not exist in the network.\n";
                  return;
            }

            cout << user << "'s friends: ";
            if (network.at(user).empty())
            {
                  cout << "No friends yet.\n";
            }
            else
            {
                  for (const auto& friendName : network.at(user))
                  {
                        cout << friendName << " ";
                  }
                  cout << endl;
            }
      }

      // Suggest friends (friends of friends) for a user
      void suggestFriends(const string& user) const
      {
            if (network.find(user) == network.end())
            {
                  cout << user << " does not exist in the network.\n";
                  return;
            }

            unordered_set<string> suggestions;
            const auto& friends = network.at(user);

            for (const auto& friendName : friends)
            {
                  // Check each friend of the current user's friends
                  for (const auto& fof : network.at(friendName))
                  {
                        // If the friend of a friend is not the user and not already a direct friend
                        if (fof != user && friends.find(fof) == friends.end())
                        {
                              suggestions.insert(fof);
                        }
                  }
            }

            // Display suggestions
            cout << "Friend suggestions for " << user << ": ";
            if (suggestions.empty())
            {
                  cout << "No suggestions available.\n";
            }
            else
            {
                  for (const auto& suggestion : suggestions)
                  {
                        cout << suggestion << " ";
                  }
                  cout << endl;
            }
      }
};


//Coding Interview Patterns Pg. 287 
bool prerequisites(int n, const vector<vector<int>>& prerequisites) {
  
vector<vector<int>> adjList(n);
vector<int> inDegree(n,0); 

for (const auto & pair :prerequisites){

      int prerequisute = pair[0];
      int course = pair[1];

      adjList[prerequisute].push_back(course);
      inDegree[course]++;
}

queue<int> q;

for (int i = 0 ; i < inDegree.size(); i++)
      if (inDegree[i] == 0)
            q.push(i);

int proccessed = 0;
while (!q.empty()){

      proccessed++;
      int pre = q.front();
      q.pop();

      for (const int course :adjList[pre]  ){
            inDegree[course]--;

            if (inDegree[course] == 0 )
                  q.push(course);
      }
}
  return (proccessed == n);
}
