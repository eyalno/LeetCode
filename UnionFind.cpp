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

bool validTree(int n, vector<vector<int>>& edges)
{

      if ((n - 1) != edges.size())
            return false;

      DisJointSet set(n);

      for (int i = 0; i < edges.size(); i++)
            if (!set.unionbyRank(edges[i][0], edges[i][1]))
                  return false;

      return true;
}


//1101. The Earliest Moment When Everyone Become Friends
int earliestAcq(vector<vector<int>>& logs, int n){

      int timeStamp = -1;
      int nodes = n;

      sort(logs.begin(), logs.end(), [](const vector<int>& a, const vector<int>& b)
            { return a[0] < b[0]; });

      DisJointSet set(n);

      for (int i = 0; i < logs.size(); i++)
      {
            int a = logs[i][1];
            int b = logs[i][2];

            if (set.unionbyRank(a, b))
            {
                  timeStamp = logs[i][0];
                  nodes--;
            }
            if (nodes == 1)
                  return timeStamp;
      }
      return -1;
}

//547. Number of Provinces
int findCircleNum(std::vector<std::vector<int>>& isConnected)
{
      int size = isConnected.size();
      DisJointSet set(size);

      for (int i = 0; i < size; i++)
            for (int j = i + 1; j < size; j++) // symmetrical matrix
                  if (isConnected[i][j])
                        set.unionbyRank(i, j);

      int province = 0;

      for (int i = 0; i < size; i++)
            if (set.findPathCompression(i) == i)
                  province++;

      return province;
}

//323. Number of Connected Componen
int countComponents(int n, vector<vector<int>>& edges)
{
      int size = edges.size();
      DisJointSet set(n);

      for (int i = 0; i < edges.size(); i++)
            set.unionbyRank(edges[i][0], edges[i][1]);

      int province = 0;

      for (int i = 0; i < n; i++)
            if (set.findPathCompression(i) == i)
                  province++;

      return province;
}


//1202. Smallest String With Swaps
string smallestStringWithSwaps(string s, vector<vector<int>>& pairs)
{
      DisJointSet set(s.size());
      if (pairs.size() == 0)
            return s;

      string ret(s.length(), ' ');

      for (const auto& pair : pairs)
            set.unionbyRank(pair[0], pair[1]);

      unordered_map<int, vector<int>> map;

      for (int i = 0; i < s.size(); i++)
            map[set.find(i)].push_back(i);

      for (const auto& it : map)
      {
            string temp;

            for (int element : it.second)
                  temp.push_back(s[element]);

            sort(temp.begin(), temp.end());

            for (int i = 0; i < it.second.size(); i++)
                  ret[it.second[i]] = temp[i];
      }

      return ret;
}
