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
#include "lib/GraphNode.h"

// BFS
GraphNode* graphDeepCopyBFS(GraphNode* node)
{
      if (!node)
            return nullptr;

      GraphNode* copy = new GraphNode(node->val);
      unordered_map<GraphNode*, GraphNode*> visited; // map of all connection between the 2 graphs

      queue<GraphNode*> queue;
      visited[node] = copy;
      queue.push(node);

      while (!queue.empty())
      {

            GraphNode* currNode = queue.front();
            queue.pop();

            for (const auto& neighbor : currNode->neighbors)
            {

                  if (visited.find(neighbor) == visited.end())
                  {

                        visited[neighbor] = new GraphNode(neighbor->val);
                        queue.push(neighbor);
                  }

                  visited[currNode]->neighbors.push_back(visited[neighbor]);
            }
      }
      return copy;
}

// DFS
GraphNode* DFSDeepCopy(GraphNode* node, unordered_map<GraphNode*, GraphNode*>& visited);
GraphNode* graphDeepCopyDFS(GraphNode* node)
{

      if (!node)
            return nullptr;

      unordered_map<GraphNode*, GraphNode*> visited;

      GraphNode* copy = DFSDeepCopy(node, visited);

      return copy;
}

GraphNode* DFSDeepCopy(GraphNode* node, unordered_map<GraphNode*, GraphNode*>& visited)
{

      if (visited.find(node) != visited.end())
            return visited[node];

      GraphNode* newNode = new GraphNode(node->val);
      visited[node] = newNode;

      for (const auto& neighbor : node->neighbors)
      {

            newNode->neighbors.push_back(DFSDeepCopy(neighbor, visited));
      }

      return newNode;
}

void countIslandsDFS(vector<vector<int>>& matrix, int i, int j)
{

      // check boundries and
      if (i < 0 || i >= matrix.size() || j < 0 || j >= matrix[0].size() || matrix[i][j] == 0)
            return;

      matrix[i][j] = 0;

      countIslandsDFS(matrix, i + 1, j);

      countIslandsDFS(matrix, i, j + 1);

      countIslandsDFS(matrix, i - 1, j);

      countIslandsDFS(matrix, i, j - 1);
}

// Count Islands
int countIslands(vector<vector<int>>& matrix)
{

      if (matrix.empty() || matrix[0].empty())
            return 0;

      int islands = 0;
      for (int i = 0; i < matrix.size(); i++)
            for (int j = 0; j < matrix[0].size(); j++)
                  if (matrix[i][j] == 1)
                  {
                        islands++;
                        countIslandsDFS(matrix, i, j);
                  }

      return islands;
}

int matrixInfection(vector<vector<int>>& matrix)
{

      if (matrix.empty() || matrix[0].empty())
            return 0;

      int rows = matrix.size();
      int cols = matrix[0].size();

      queue<pair<int, int>> queue;
      int ones = 0;
      int seconds = 0;

      for (int i = 0; i < matrix.size(); i++)
            for (int j = 0; j < matrix[0].size(); j++)
            {
                  if (matrix[i][j] == 2)
                        queue.push({ i, j });

                  if (matrix[i][j] == 1)
                        ones++;
            }

      // If no uninfected cells exist
      if (ones == 0)
      {
            return 0;
      }

      while (!queue.empty())
      {

            int levelSize = queue.size();
            bool infectionFound = false;
            for (int k = 0; k < levelSize; k++)
            { // give me level by level logic

                  auto pos = queue.front();
                  int i = pos.first;
                  int j = pos.second;

                  queue.pop();

                  if (i + 1 < rows && matrix[i + 1][j] == 1)
                  {
                        queue.push({ i + 1, j });
                        ones--;
                        matrix[i + 1][j] = 2;
                        infectionFound = true;
                  }

                  if (i - 1 >= 0 && matrix[i - 1][j] == 1)
                  {
                        queue.push({ i - 1, j });
                        ones--;
                        matrix[i - 1][j] = 2;
                        infectionFound = true;
                  }

                  if (j + 1 < cols && matrix[i][j + 1] == 1)
                  {
                        queue.push({ i, j + 1 });
                        ones--;
                        matrix[i][j + 1] = 2;
                        infectionFound = true;
                  }
                  if (j - 1 >= 0 && matrix[i][j - 1] == 1)
                  {
                        queue.push({ i, j - 1 });
                        ones--;
                        matrix[i][j - 1] = 2;
                        infectionFound = true;
                  }
            }
            if (infectionFound)
                  seconds++;
      }

      return ones == 0 ? seconds : -1;
}

bool bipartiteDFS(int node, int color, const vector<vector<int>>& graph, vector<int>& colors)
{

      // If already colored, check if it's the same color
      if (colors[node] != 0)
            return color == colors[node];

      // Color the current node
      colors[node] = color;

      for (int neighbor : graph[node])
            if (!bipartiteDFS(neighbor, -color, graph, colors))
                  return false;

      return true;
}

bool bipartite_graph_validation(const vector<vector<int>>& graph)
{

      bool isValid = true;

      int size = graph.size();
      vector<int> colors(size, 0);

      for (int i = 0; i < size && isValid; i++)
      {
            if (colors[i] == 0)
            {
                  isValid = bipartiteDFS(i, -1, graph, colors);
            }
      }

      return isValid;
}

int lipDFS(int i, int j, int prevNum, const vector<vector<int>>& matrix, vector<vector<int>>& memo)
{

      int rows = matrix.size();
      int cols = matrix[0].size();

      if (i < 0 || i >= rows || j < 0 || j >= cols || prevNum >= matrix[i][j])
            return 0;

      if (memo[i][j] != -1)
            return memo[i][j];

      vector<pair<int, int>> directions({ {-1, 0}, {1, 0}, {0, -1}, {0, 1} });
      int res = 1; // if it's the last cell the value is 1

      for (auto [dx, dy] : directions)
            res = std::max(res, 1 + lipDFS(i + dx, j + dy, matrix[i][j], matrix, memo));

      memo[i][j] = res;
      return res;
}

int longest_increasing_path(const vector<vector<int>>& matrix)
{

      if (matrix.empty() || matrix[0].empty())
            return 0;

      int rows = matrix.size();
      int cols = matrix[0].size();

      vector<vector<int>> memo(rows, vector<int>(cols, -1));

      int maxPath = 0;
      for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                  maxPath = std::max(maxPath, lipDFS(i, j, -1, matrix, memo));

      return maxPath;
}

// Shortest Transformation Sequence
int shortestTransformationSequence(string start, string end, vector<string>& dictionary)
{
      // Convert dictionary to a set for O(1) lookup
      unordered_set<string> wordSet(dictionary.begin(), dictionary.end());

      if (wordSet.find(end) == wordSet.end())
            return -1;

      queue<pair<string, int>> queue;

      queue.push({ start, 1 });

      while (!queue.empty())
      {

            auto [currentWord, steps] = queue.front();
            queue.pop();

            if (currentWord == end)
                  return steps;

            // Try changing each character in the current word and search it
            for (int i = 0; i < currentWord.length(); ++i)
            {
                  string tempWord = currentWord;
                  for (char ch = 'a'; ch <= 'z'; ch++)
                  {
                        tempWord[i] = ch;

                        if (wordSet.find(tempWord) != wordSet.end())
                        {
                              queue.push({ tempWord, steps + 1 });
                              wordSet.erase(tempWord); // Remove to prevent revisiting
                        }
                  }
            }
      }
      return -1;
}

class MergingCommunities
{
private:
      std::vector<int> parent, rank, size;

public:
      // Constructor
      MergingCommunities(int n)
      {
            // Initialize parent and size vectors

            size.resize(n, 1);
            rank.resize(n, 0);
            parent.resize(n);
            for (int i = 0; i < n; i++)
                  parent[i] = i; // Each node is its own parent
      }

      int find(int x)
      {
            if (parent[x] == x)
                  return x;
            parent[x] = find(parent[x]);
            return parent[x];
      }

      // Connect two nodes
      void connect(int x, int y)
      {
            // Implementation for union operation
            int rootX = find(x);
            int rootY = find(y);

            if (rootX == rootY)
                  return;

            if (rank[rootX] == rank[rootY])
            {
                  rank[rootX]++;
                  parent[rootY] = rootX;
                  size[rootX] += size[rootY];
            }
            else if (rank[rootX] > rank[rootY])
            {
                  parent[rootY] = rootX;
                  size[rootX] += size[rootY];
            }
            else
            {
                  parent[rootX] = rootY;
                  size[rootY] += size[rootX];
            }
            return;
      }

      // Get the size of the community of a given node
      int getCommunitySize(int x)
      {
            // Implementation to find the size of the community

            return size[find(x)];
            ; // Placeholder
      }
};

