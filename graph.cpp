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


//Coding Interview Patterns - Backtracking Pg. 298 
// Time complexity O(n!) factorial. we explore n-1 n-2 for each level  
// space O(n) the extra vector and the recursion call maximum depth n   

void  find_all_permutationsDFS(vector<int>& nums, vector<bool> & visited, vector<int>  & current, vector<vector<int>> & results  );

vector<vector<int>> find_all_permutations(vector<int>& nums) {
    
      int size = nums.size();
      vector<bool>  visited(size,false);
      vector<vector<int>>  results;
      vector<int> current;
      current.reserve(size);
      find_all_permutationsDFS(nums,visited,current,results);

      return results;
}

//for every node try all options.
void  find_all_permutationsDFS(vector<int>& nums, vector<bool> & visited, vector<int>  & current, vector<vector<int>> & results  ){

      static int size = nums.size();

      if (current.size() == size){
            results.push_back(current);
            return;
      }

      for (int i = 0 ; i < size; i++ ){
            if (visited[i]) continue;

            visited[i] = true;
            current.push_back(nums[i]);
            find_all_permutationsDFS(nums,visited,current,results);

            visited[i] = false;
            current.pop_back();
      }

      return;
}


//Coding Interview Patterns - Backtracking Pg. 302
//2^n subsets * n since 2 decisions 
void findAllSubsetsDFSV1(const vector<int>& nums,int index, vector<int>& current,vector<vector<int>> & results  ); 
void findAllSubsetsDFSV2(const vector<int>& nums,int index, vector<int>& current,vector<vector<int>> & results  ); 

vector<vector<int>> findAllSubsets(const vector<int>& nums) {
    

      vector<vector<int>> results;
      vector<int> current;
      current.reserve(nums.size());

      findAllSubsetsDFSV1(nums, 0 ,  current, results );
      findAllSubsetsDFSV2(nums, 0 ,  current, results );

      return results;
}

// run on the vector and include/exlude the next item. increasing the index making sure we dont inclue previous nums
void findAllSubsetsDFSV1(const vector<int>& nums,int index, vector<int>& current,vector<vector<int>> & results ){

      static int size = nums.size();

      if (index == size){
            results.push_back(current);
            return;
      }

      // Exclude the current element at each step and recurse
      findAllSubsetsDFSV1(nums, index + 1 ,  current, results );

      //include the element 
      current.push_back(nums[index]);
      findAllSubsetsDFSV1(nums, index + 1 ,  current, results );
      
      //backtrack - go up the graph - return the previous state of the vector
      current.pop_back();
} 

void findAllSubsetsDFSV2(const vector<int>& nums,int start, vector<int>& current,vector<vector<int>> & results ){

      static int size = nums.size();
      //adding subset
      results.push_back(current);

      // start all other subsets options 
      // initally we start with including and  the pop back exclude the number and keep traversing 
      for (int index = start; index < size; index++){
            current.push_back(nums[index]);

            findAllSubsetsDFSV2(nums, index + 1 ,  current, results );

            current.pop_back();
      }
      return;
}


//Coding Interview Patterns - Backtracking Pg. 305
void nQueensDFS(int n, int row, vector<int> & column,vector<int> & diag1 , vector<int> & diag2, int & count); 
// time complexity n! each row n-a n-b n-c 
int nQueens(int n) {
    // Write your code here
    
    vector<int> column(n,0);
    vector<int> diag1(2*n -1,0);
    vector<int> diag2(2*n -1,0);
    int count = 0;

    nQueensDFS(n,0,column,diag1,diag2,count);

    return count;
}

// we traverse all options we continue if not valid path and back track to check other flows
void nQueensDFS(int n, int row, vector<int> & column,vector<int> & diag1 , vector<int> & diag2 ,int & count) {

      if (row == n){
            count++;
            return;
      }

      for (int col = 0; col < n; col ++){

            //check if each column fit. skip it 
            if ( column[col] || diag1[col + row] || diag2[ row - col +n -1]  )
                  continue;

            column[col] = diag1[col + row] = diag2[ row - col +n -1] = 1 ;

            // next column
            nQueensDFS(n,row +1,column,diag1,diag2,count);

            //backtrack remove the queen
            column[col] = diag1[col + row] = diag2[ row - col +n -1] = 0 ;
      }

      return ;
}


// Coding Interview Patterns - Dynamic Programming Pg. 311
// Considered top down since starting from the main problem 
//recursion tree with depth n. 2^n
//with memoization n subproblems so O(n)
int climbingStairsHelper(int n,vector<int> &memo);

int climbingStairs(int n) {
    
      vector<int> memo(n+1,0); 

      return climbingStairsHelper(n,memo);
}

int climbingStairsHelper(int n,vector<int> & memo) {

//base case
      if (n <= 2)
            return n;

      if (memo[n] != 0)
            return memo[n];

      // from this step only 1 step is possible so it is the same step so we add this two
      memo[n] = climbingStairsHelper(n-1,memo) + climbingStairsHelper(n-2,memo);

    return  memo[n];

}
//Bottom up DP
int climbingStairsV2(int n) {
    
      vector<int> memo(n+1,0); 

      memo[1] = 1;
      memo[2] = 2;


      for (int i =3; i <= n; i++ )
            memo[i] = memo[i-2] +memo[i-1];

      return memo[n];
}

//optimization 2 variable no vector
int climbingStairsV3(int n) {
    
      if (n<=2)
            return n;
      
      int oneStep = 2; 
      int twoStep = 1;
      
      int current = 0;
      for (int i = 3; i <= n; i++ ){
            current = oneStep + twoStep;

            oneStep = twoStep;
            twoStep = current;
      }

      return current;
}

//Coding Interview Patterns - Dynamic Programming Pg. 316

int minCoinCombinationHelper(const std::vector<int>& coins, int target,unordered_map<int,int> & memo);

int minCoinCombination(const std::vector<int>& coins, int target) {
    // Write your code here
    
    unordered_map<int,int> memo;

    numeric_limits<int>::max();

    int res = minCoinCombinationHelper(coins,target,memo);
    

    return res == numeric_limits<int>::max() ? -1 : res  ; // Placeholder return
}

int minCoinCombinationHelper(const std::vector<int>& coins, int target,unordered_map<int,int> & memo){

      constexpr int INF = numeric_limits<int>::max(); 

      if ( target == 0) return 0;
      if ( target < 0) return INF; //nopt valid path

      if (memo.find(target) != memo.end())
            return memo[target];

      int minCoins = INF;

      //comparing all coins
      for ( int coin:coins ){
            
            int res = minCoinCombinationHelper(coins,target - coin,memo); //new target

            if (res != INF)
                  minCoins =  min(minCoins,res +1) ; //new target + 1 = current target

      }

      memo[target] = minCoins;
      return minCoins; // return current target     

}
//time compexity - max target subproblems for each target we loop n coins  = n * target
//Bottom up solution can be acheived by translating memoizatin to DP array. 


int minCoinCombinationV2(const std::vector<int>& coins, int target) {

      constexpr int INF = numeric_limits<int>::max();

      vector<int> dp(target + 1, INF);

      dp[0] = 0; //base case

      //populating the  target array if we can. 
      
      for (int currTarget = 1 ; currTarget <= target; currTarget++){

            for (int coin: coins){
                  //
                  if (currTarget - coin >= 0 )
                        dp[currTarget] = min(dp[currTarget], dp[currTarget - coin] + 1 ); 
                        //might be more than one path to reach the target  
            }
      }

return dp[target] != INF ? dp[target]: -1;

}
//time complexity target * coins

//Coding Interview Patterns - Dynamic Programming Pg. 321

int matrixPathways(int m, int n) {

      vector<vector<int>> dp(m,vector<int>(n,0)); 

      for (int i = 0; i < m; i ++  )
            dp[i][0] = 1;

      for (int i = 0; i < n; i ++  )
            dp[0][i] = 1;

      for (int i = 1; i < m; i ++)
            for (int j=1; j < n ; j++)
                  dp[i][j] = dp[i][j-1] + dp[i-1][j]; // the num of paths to each cell is the num above and to the left
      
      return dp[m-1][n-1];
}

//space optimization we only need to keep top row and current row
//we can take one from current row and prev row
int matrixPathwaysV2(int m, int n) {

      vector<int> prevRow (n,1); //base
      vector<int> currRow (n,0); //being populated

      for (int i = 1; i < m; i ++){
            currRow[0] = 1;
            for (int j=1; j < n ; j++)
                  currRow[j] =   currRow[j-1] + prevRow[j];
            swap(prevRow ,currRow);
      }

      return prevRow[n-1];    
 }

 //Coding Interview Patterns - Dynamic Programming Pg. 325
//we found sub problems if we rob the last house or skip it
//and we defined a formula acording to it based on house i and previous DP. 
// to populate dp D[i]  = max (DP[i-1] , DP[i-2]  + house[i] )
 
 int neighborhood_burglary(const std::vector<int>& houses) {
 
      int size = houses.size();
      vector<int> dp(size,0);

      if (size ==1)
            return houses[0];
      
      //base cases
      dp[0] = houses[0];
      dp[1] = max(houses[0],houses[1]); // the most money that can be stolen

      for (int i = 2; i < size; i ++)
            dp [i] = max(dp[i-1], houses[i] + dp[i-2]);

    return dp[size -1]; 
}

//Coding Interview Patterns - Dynamic Programming Pg. 329
// if s1[i] == s2[i]
      // dp[i][j] = 1 + dp[i+1][j+1]; 
//else
      //case when  not equal excluding each character 
      //dp[i][j] = max(dp[i][j+1] , dp[i+1][j])
//
int longest_common_subsequence(const std::string& s1, const std::string& s2) {
    
      int s1Size = s1.size();
      int s2Size = s2.size();
      
      vector<vector<int>> dp(s1Size +1,vector<int> (s2Size+1,0));
      
      for (int i = s1Size-1;  i >= 0 ; i--)
            for (int j = s2Size-1;  j >= 0 ; j--){
                  
                  if (s1[i] == s2[j])
                        dp[i][j] = 1 + dp[i+1][j+1]; 
                  else  
                      dp[i][j] = max(dp[i][j+1] , dp[i+1][j]);
            }

    return dp[0][0]; // Placeholder return
}

//optimization - we don't 2d matrix 
//we can take one from current row and prev row
int longest_common_subsequenceV2(const std::string& s1, const std::string& s2) {
    
      int s1Size = s1.size();
      int s2Size = s2.size();
      
      vector<int> prev(s2Size+1,0) , curr(s2Size+1,0);
      
      for (int i = s1Size-1 ;  i >= 0 ; i--){
            for (int j = s2Size-1;  j >= 0 ; j--){
                  if (s1[i] == s2[j])
                        curr[j] = 1 + prev[j+1]; //  represents dp[i+1][j+1]
                  else  
                        curr[j] = max(curr[j+1] , prev[j]);
            }
            swap(prev,curr);
      }

    return prev[0]; // Placeholder return
}

//Coding Interview Patterns - Dynamic Programming Pg. 334

//. i-j range dp[i][j] -> true if s[i] == s[j] and dp[i+1][j-1] == true  -inner and outer
//base case all length 1 are true dp[i][i] =true
// if s[i] == s[i+1] then dp[i][i+1]
// length 0 1 2 o inner substring 

string longestPalindromeCodingInterview(string s) {

      int size = s.size();

      if (size == 0)
            return "";

      vector<vector<bool>> dp(size,vector<bool>(size,false));

      int start = 0, maxLen = 1;

      for (int i =0; i < size; i++) dp[i][i] = true;

      for (int i = 0; i < size-1; i++ )
            if (s[i] == s[i+1]){
                  dp[i][i+1] = true;
                  start = i;
                  maxLen = 2;
            }

      // we have the base case dp now increase len and check all combination

      for (int len = 3 ; len <= size  ; len++)
            for (int i = 0 ; i <= size - len   ; i++){  // 8 -3  = 5
                  int j = i + len-1;
                  if (s[i] ==s[j] && dp[i+1][j-1]){
                        dp[i][j] = true;
                        start = i;
                        maxLen = len;
                  }
            }
      
      return s.substr(start,maxLen);
}


//optimized version expand from base cases
/*
string longestPalindromeV2(string s) {

      int len = s.length();

      if (len == 0) return "";

      int start = 0 ;
      int maxLen = 1;


      for (int i = 0; i < len; i++ ){

            if ( i  < len-1 && s[i] == s[i+1]){


            }



      }


}*/