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

//test
using namespace std;
//using std::vector;

class TrieNode;
class Trie;


struct TreeNode {
     int val;
     TreeNode *left;
     TreeNode *right;
     TreeNode() : val(0), left(nullptr), right(nullptr) {}
     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};


struct ListNode {
          
      int val;
      ListNode *next;
      ListNode() : val(0), next(nullptr) {}
      ListNode(int x) : val(x), next(nullptr) {}
      ListNode(int x, ListNode *next) : val(x), next(next) {}
 };


vector<int>  sortedSquares(vector<int>& nums);
vector<int>  sortedSquaresV2(vector<int>& nums);
vector<int>  sortArrayByParity(vector<int>& nums);
vector<int>  sortArrayByParityV2(vector<int>& nums);
int removeElement(vector<int>& nums, int val);
vector<int> twoSum(vector<int>& nums, int target);
vector<int> twoSumMap(vector<int>& nums, int target);
int maxProfit(vector<int>& prices);
void moveZeroes(vector<int>& nums);
int partitionDisjoint(vector<int>& nums);
int partitionDisjointV2(vector<int>& nums);
int partitionDisjointV3(vector<int>& nums);
bool checkIfExist(vector<int>& arr);      
vector<int> replaceElements(vector<int>& arr);
int removeDuplicates(vector<int>& nums);
void merge(vector<int>& nums1, int m, vector<int>& nums2, int n);
void mergeSort(vector<int>& nums1, int m, vector<int>& nums2, int n);    
void duplicateZeros(vector<int>& arr);    
bool validMountainArray(vector<int>& arr);
int heightChecker(vector<int>& heights);
int findMaxConsecutiveOnes(vector<int>& nums);   
int thirdMax(vector<int>& nums); 
vector<int> plusOne(vector<int>& digits);
vector<int> findDisappearedNumbers(vector<int>& nums);
bool isLongPressedName(string name, string typed);
int pivotIndex(vector<int>& nums);
int dominantIndex(vector<int>& nums);
vector<int> findDiagonalOrder(vector<vector<int>>& mat);
vector<int> spiralOrder(vector<vector<int>>& matrix);
vector<vector<int>> generate(int numRows);
string addBinary(string a, string b);
int strStr(string haystack, string needle);
string longestCommonPrefix(vector<string>& strs);
string longestCommonPrefix(vector<string>& strs,int l, int r);
bool isCommonPrefix(vector<string>& strs,int len);
int arrayPairSum(vector<int>& nums);
void countingSort(std::vector<int>& arr);
int minSubArrayLen(int target, vector<int>& nums);
int binarySearch( vector<int>& nums,int target);
void rotate(vector<int>& nums, int k);
vector<int> getRow(int rowIndex);
int binomialCoefficient(int n , int k);
string reverseWords(string s);
bool isIsomorphic(string s, string t);
vector<string> findRestaurant(vector<string>& list1, vector<string>& list2);
int firstUniqChar(string s);
int search(vector<int>& nums, int target);
bool backspaceCompare(string s, string t);


int hammingWeight(int n) {
        
      // use running mask
      int mask = 1;
      int count = 0;

      for (int i =0; i < 32; i++ ){
            
            if ((n & mask) == mask)
                  count++;

           mask <<= 1;  

      } 
      return count ;

    /* //removeing lsb
    int bitCount = 0;
      
    while (n !=0){    
      n&= n-1; 
      bitCount++;
            
    }  
    return bitCount;
    */
}


bool isSameTree(TreeNode* p, TreeNode* q) {

//DFS
if (!p && !q ) return true;

if (!p || !q) return false; 

stack<TreeNode *> stackP;
stack<TreeNode *> stackQ;

stackP.push(p);
stackQ.push(q);


while (!stackP.empty() && !stackQ.empty()  ){

      TreeNode * nodeP = stackP.top();
      TreeNode * nodeQ = stackQ.top();

      stackP.pop();
      stackQ.pop();

      if (nodeP->val != nodeQ ->val) return false;


      if (nodeP->left && nodeQ->left){
            stackP.push(nodeP->left);
            stackQ.push(nodeQ->left);
      }
      else if (nodeP->left || nodeQ->left)
            return false;

      if (nodeP->right && nodeQ->right){
            stackP.push(nodeP->right);
            stackQ.push(nodeQ->right);
      }
      else if (nodeP->right || nodeQ->right)
            return false;


}

return stackP.empty() && stackQ.empty();


/* BFS
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
*/

/*recrusive
if (!p && !q )
      return true;

if (!p || !q )
      return false;

bool left = isSameTree(p->left,q->left);
bool right = isSameTree(p->right,q->right);

if (p->val != q->val)
      return false;

return left && right;
*/
}


vector<int> countBits(int n) {

//popcount
vector<int> result;
for (int i= 0; i<= n ; i++){

      int bitCount = 0;
      int num = i;

      while (num !=0){    
            num&= num-1;
            bitCount++;
            
      }  
      result.push_back(bitCount);
}


/*brute force

vector<int> result;
for (int i= 0; i<= n ; i++){

      int bitCount = 0;
      int num = i;

      while (num !=0){    
            bitCount+=( num&1 );
            num = num >> 1;
      }  
      result.push_back(bitCount);
}
*/
return result;

}

string processString(string & s){

string result;

for (int i =0; i < s.size(); i++){

      if(s[i]  == '#' && (!result.empty()) )
            result.pop_back();
      else
            result.push_back(s[i]);
}
return result;

}

bool backspaceCompare(string s, string t) {

return processString(s) == processString(t);

}


int romanToInt(string s) {

unordered_map<char,int> map;

map['I'] =1;
map['V'] =5;
map['X'] =10;
map['L'] =50;
map['C'] =100;
map['D'] =500;
map['M'] =1000;



/*
map["I"] =1;
map["V"] =5;
map["X"] =10;
map["L"] =50;
map["C"] =100;
map["D"] =500;
map["M"] =1000;
map["IV"] =4;


map["IX"] =9;
map["XL"] =40;
map["XC"] =90;
map["CD"] =400;
map["CM"] =900;
*/
//right to left


int i =s.size()-1;
int sum =0;


while ( i >=0 ){
      int lastNum =0;
      char ch = s[i];
      
      int num = map[ch];
      
      if (num < lastNum )
            sum -=num;
      else 
            sum += num;

      lastNum= num;
      i--;
}
return sum;

/*left to right


int i =0;
int sum =0;

while ( i < s.size()){
      if (i < s.size() -1){
            string doubleSymbol = s.substr(i,2);

            if (map.count(doubleSymbol) > 0){
                  i+=2;
                  sum += map[doubleSymbol];
                  continue;
            }
      }

      string singleSymbol = s.substr(i,1);
      i+=1;
      sum+= map[singleSymbol];
}



return sum;
*/
}



 bool canAttendMeetings(vector<vector<int>>& intervals) {

      if (intervals.empty()) {
            return true;
        }
        // Sort intervals based on their start times
     sort(intervals.begin(),intervals.end());


      for(int i =0; i <intervals.size() -1; i++ ){
            if (intervals[i][1] >   intervals[i+1][0])
                  return false;
      }

return true;
 }


bool containsDuplicate(vector<int>& nums) {

unordered_set<int> set;

for (int num:nums)
      if (set.count(num) >0)
            return true; 
      else
            set.insert(num);

return false;

/*sort
sort(nums.begin(),nums.end());

for (int i =0; i < nums.size() -1; i++)
      if (nums[i] == nums[i+1])
            return true;

return false;
*/
}

int majorityElement(vector<int>& nums) {

//bit manipulation
int majElem = 0;
for (int i = 0 ; i <32 ; i++){

      int count =0;

      for (int elem:nums){
            if ((elem & (1 << i)) != 0) 
                  count++;
            if (count > (nums.size()/2))
                  majElem |= (1<< i);
      }
}
return majElem;

/*since the majority > n/2 will be in the middle in sorted array
sort(nums.begin(),nums.end());

return nums[nums.size()/2];
*/
}



int longestPalindrome(string s) {

vector<int> lettersCount(52,0);


for (char ch:s){
  if (isupper(ch) )
      lettersCount[26 + ch -'A']++;
  else
      lettersCount[ch -'a']++;
}

int mid =0;
int length = 0;
for (int sum:lettersCount){

      if ((sum %2) == 0)
            length+= sum;
      else{
            mid =1;
            length +=sum -1;
      }
}

return length + mid;
/* hash map
unordered_map<char,int> map;

int even = 0 ,odd = 0;

for (char ch:s){
      map[ch]++;
}


for (auto it = map.begin(); it != map.end() ; it++){

      if ((it->second %2) == 0  )
            even += it->second ;
      else {
            even+= it->second -1;
            odd++;
      }
            
}

return even + (odd>0?1:0)  ;

*/

}


int climbStairsHelper(int n,vector<int> & memo) {

if (n == 0)
      return 1;
if (n == -1)
      return 0;

if (memo[n] != -1)
      return memo[n];
memo[n] = climbStairsHelper(n-1,memo) + climbStairsHelper(n-2,memo);

return memo[n];


}

int climbStairs(int n) {

 if (n == 1) {
            return 1;
        }

vector<int> dp(n+1,1);

dp[1] = 1;
dp[2] = 2;

for (int i =3; i <=n; i++ )
      dp[i] = dp[i-1] + dp[i-2];

return dp[n];

/* recursive and  memoization

vector<int> memo(n+1,-1);

return climbStairsHelper(n,memo);

*/

     /* no memoization time limit excedded
      if (n ==0)  
            return 1;
      if (n == -1)
            return 0;      

      return climbStairs(n-1)+climbStairs(n-2);
       */

}


void floodFillHelper(vector<vector<int>>& image, int sr, int sc, int originalColor, int newColor) {

int rows = image.size();
int cols = image[0].size();

if (sr < 0 || sr >= rows || sc < 0 || sc >=cols
 || image[sr][sc] != originalColor || image[sr][sc] == newColor     )
      return;

image[sr][sc] = newColor ;

floodFillHelper(image,sr -1 ,sc,originalColor ,newColor);
floodFillHelper(image,sr +1  ,sc,originalColor ,newColor);
floodFillHelper(image,sr ,sc -1,originalColor ,newColor);
floodFillHelper(image,sr ,sc +1 ,originalColor ,newColor);



}

bool canConstruct(string ransomNote, string magazine) {

if (ransomNote.length() > magazine.length())
      return false;

unordered_map<char,int> map;

for (char ch:magazine){
      map[ch]++;
} 


for (char ch:ransomNote){

      if(map[ch] > 0 )  
            map[ch]--;
      else  
            return false;
}
return true;

}


/*
int firstBadVersion(int n) {

      int i = 1; 
      int j =n;
      
      while (i<j){

            int mid = i +  (j-i)/2;
            if (isBadVersion(mid)){
                  j = mid ;
            }
            else
                  i = mid +1;
      }
        return i;

        
}
*/


class MyQueue {

private:
//stack<int>  & stackRefMain;
//stack<int>  & stackRefSec;

stack<int>  enqueueStack;
stack<int>  dequeueStack;

public:
   
    MyQueue() {
         
    }
    
    void push(int x) {
      enqueueStack.push(x);
    }
    
    int pop() {
       int val;
      val = peek();
      dequeueStack.pop();
      return val;
    }
    
    int peek() {
        int val;

      if (!dequeueStack.empty()){
            val = dequeueStack.top();
      }
      else{
            while (!enqueueStack.empty()){
                  dequeueStack.push(enqueueStack.top());
                  enqueueStack.pop();
            }
            val = dequeueStack.top();
            
      }
      return val; 
    }
    
    bool empty() {
        
      if (dequeueStack.empty() && enqueueStack.empty())
            return true;
      else
            return false;
    }
}
;



vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int color) {

if (image.empty())
      return image;

int rows = image.size();
int cols = image[0].size();

if (sr < 0 || sr >= rows || cols < 0 || sc >=cols )
      return image;

int originalColor = image[sr][sc];

if (originalColor == color)
      return image;

floodFillHelper(image,sr,sc,originalColor ,color);

return image;



}


//binary search
int search(vector<int>& nums, int target) {


int start = 0;
int end = nums.size() -1;


while (start <= end )
{
      int mid = (start + end) /2;

      if (target == nums[mid])
            return mid;
      
      if (target > nums[mid] )
            start = mid +1;
      else
            end = mid -1;
}


return -1;



}

bool isAnagram(string s, string t) {

unordered_map<char,int> counter;

if (s.length() != t.length())
      return false;

for (char ch:s)
      counter[ch]++;

for (char ch:t){
      
      if (counter.find(ch) == counter.end() || counter[ch] == 0)
            return false;
      counter[ch]--;

}

for (auto & pair:counter)
      if(pair.second != 0)
            return false;


return true;

/*sort
      if (s.length() != t.length())
      return false;

      sort(s.begin(),s.end());
      sort(t.begin(),t.end());

      return s == t;
*/
}


int firstUniqChar(string s){

unordered_map<char,int> map ;
//s ="loveleetcode";
int ret = -1;

for (char ch: s){
      map[ch]++;
}


for ( int i =0; i < s.size(); i++  ){

      if (map[s[i]] == 1)
            return i;
}

return ret;


}


class DisJointSet{

private:
      vector<int> parent;
      vector<int> root;
      vector<int> rank;
public:
      DisJointSet(int n):parent(n),root(n),rank(n,0){
            for (int i =0; i< n; i++){
                  parent[i] = i;
                  root[i] = i;
                  
            }
      }

      int find(int x) { // O(n)
            if (parent[x] == x)
                  return x;
            return find(parent[x]);
      }

      void unionSets(int a, int b) { // O(n)
            int rootA = find(a);
            int rootB = find(b);      

            if (rootA != rootB)
                  parent[rootB] = rootA;
      }

      bool isConnected(int a, int b) {
            return find(a) == find(b);
      }

      //Quick Find
     
      int quickFind(int x) { //O(1) 
        return root[x];
      }

      void quickFindUnionSets(int a, int b) { // O(n) 
            int rootA = quickFind(a);
            int rootB = quickFind(b);      

            if (rootA != rootB){
            
                  for (int i = 0; i < root.size(); ++i  ){
                              if (rootA == rootB)
                                    root[i] = rootA;     
                  }
            }
      }


       bool isQuickConnected(int a, int b) {
            return quickFind(a) == quickFind(b);
      }


      //Quick Union

      int quickUnionFind(int x) { //O(1) 
         while(root[x] != x )
            x = root[x];
        return x;
      }

      void quickUnionSets(int a, int b) { // O(1)
            int rootA = find(a); //can be O(n)
            int rootB = find(b);      

            if (rootA != rootB)
                  parent[rootB] = rootA;
      }

      //union by rank
      bool unionbyRank(int a, int b) { // O(1)
            int rootA = findPathCompression(a); //can be O(lgn)
            int rootB =  findPathCompression(b);      

            if (rootA == rootB)
                  return false;

            if (rootA != rootB){
                  if(rank[rootA] > rank[rootB] )
                        root[rootB] = rootA;
                  else if(rank[rootB] > rank[rootA] )
                        root[rootA] = rootB;
                  else {
                        rank[rootA]++;
                        root[rootB] = rootA;
                  } 
            }
            return true;
      }

      //path compression
      int findPathCompression(int x){

            if(x == root[x])
                  return x;

            return root[x] = findPathCompression(root[x]);
      }
};


/*

class DisJointSet{

private:
      vector<int> parent;
      vector<int> root;
      vector<int> rank;
public:
      DisJointSet(int n):parent(n),root(n),rank(n,0){
            for (int i =0; i< n; i++){
                  parent[i] = i;
                  root[i] = i;       
            }
      }
 
      bool isConnected(int a, int b) {
            return findPathCompression(a) == findPathCompression(b);
      }

      //union by rank
      bool unionbyRank(int a, int b) { // O(1)
            int rootA = findPathCompression(a); //can be O(lgn)
            int rootB =  findPathCompression(b);      

            if (rootA == rootB)
                  return false;

            if (rootA != rootB){
                  if(rank[rootA] > rank[rootB] )
                        root[rootB] = rootA;
                  else if(rank[rootB] > rank[rootA] )
                        root[rootA] = rootB;
                  else {
                        rank[rootA]++;
                        root[rootB] = rootA;
                  } 
            }
            return true;
      }

      //path compression
      int findPathCompression(int x){

            if(x == root[x])
                  return x;

            return root[x] = findPathCompression(root[x]);
      }
};


*/



int findCircleNum(std::vector<std::vector<int>>& isConnected) {

      int size = isConnected.size();
      DisJointSet set(size);

      for (int i =0 ; i < size; i++)
            for (int j = i +1; j <size; j++ ) // symmetrical matrix
                  if (isConnected[i][j])
                        set.unionbyRank(i,j);

      int province = 0;

      for (int i =0; i < size; i++)
            if  ( set.findPathCompression(i) == i)
                  province++;
      
      return province;
}     





bool validTree(int n, vector<vector<int>>& edges) {
         
      if ( (n-1) != edges.size() )
            return false;

       DisJointSet set(n);

       for (int i =0; i <edges.size(); i++)
            if (!set.unionbyRank(edges[i][0] , edges[i][1] ))
                  return false;

      return true;
    }

       int countComponents(int n, vector<vector<int>>& edges) {
        
         int size = edges.size();
      DisJointSet set(n);


        for (int i =0; i <edges.size(); i++)
            set.unionbyRank(edges[i][0] , edges[i][1] );

      int province = 0;

      for (int i =0; i < n; i++)
            if  ( set.findPathCompression(i) == i)
                  province++;
      
      return province;

    }




bool compareByTimestamp (const vector<int> & a, const vector<int>& b )
{
      return a[0] < b[0]; 
}

int earliestAcq(vector<vector<int>>& logs, int n) {
        
      int timeStamp = -1;
      int nodes = n;

      sort(logs.begin(),logs.end(),[](const vector<int> & a, const vector<int>& b )
{
      return a[0] < b[0]; 
});

      DisJointSet set(n);
      
      for (int i = 0; i < logs.size(); i ++){
            int a = logs[i][1];
            int b = logs[i][2];

           if ( set.unionbyRank(a,b)){
                  timeStamp = logs[i][0];
                  nodes--;
            }
            if (nodes == 1)
                  return timeStamp;
      }
      return -1;
    }


  string smallestStringWithSwaps(string s, vector<vector<int>>& pairs) {

      DisJointSet set(s.size());
      if (pairs.size() == 0)
            return s;

      string ret(s.length(),' ');

      for ( const auto & pair  : pairs)
            set.unionbyRank(pair[0],pair[1]);

      unordered_map<int,vector<int>> map;

      for (int i = 0 ; i <  s.size(); i++)
            map[set.find(i)].push_back(i); 

      for (const auto & it :map  ){
            //auto vec = it.second;

            string temp;

            for (int element :it.second)
                       temp.push_back(s[element]);

            sort(temp.begin(),temp.end());

            for (int i =0; i < it.second.size(); i++)
                  ret[it.second[i]] =temp[i];

      }

       return ret;

    }


class UnionFind {
private:

      unordered_map<string,string> parent;
      unordered_map<string,double> weight;
public:

      //Parent:   {"a": "b", "b": "c", "c": "c"}
      //Weight:   {"a": 2.0, "b": 3.0, "c": 1.0}

      // Initialize the parent and weight for a node if not already done
    void add(string x) {
      if (parent.find(x) == parent.end()){
            parent[x] = x;
            weight[x] = 1.0;
      }
    }

    // Find the root of a node and apply path compression
    string find(string x) {
      
            if ( parent[x] == x){
                  string origParent = parent[x];

                  parent[x] = find(parent[x]); //path compression

            }
            return parent[x];
    }

       // Union the sets of two nodes with a given ratio
    void unite(string x, string y, double value) {
       add(x);
       add(y);

       string rootX = find (x);
       string rootY = find (y);
       if (rootX != rootY){
            parent[rootX] = rootY;

       }
    }



};


bool validPathDfs(int curr, int dest, unordered_map<int,vector<int>>& graph, unordered_set<int> & visited ){

      if (curr == dest )
            return true;
      visited.insert(curr);

      for (int neighbor : graph[curr] ){

            if (visited.find(neighbor) == visited.end())
                 if (validPathDfs(neighbor, dest, graph, visited))
                  return true;
      }

return false;

}
bool validPath(int n, vector<vector<int>>& edges, int source, int destination) {

      unordered_map<int,vector<int>> graph;

      for (const auto & edge : edges ){
            int u = edge[0];
            int v = edge[1];
            graph[u].push_back(v);
            graph[v].push_back(u);
      }

      unordered_set<int> visited;

      return validPathDfs(source, destination,graph,visited);



}

//Stack solution

bool validPathStack(int n, vector<vector<int>>& edges, int source, int destination) {

      unordered_map<int,vector<int>> graph;

      for (const auto & edge : edges ){
            int u = edge[0];
            int v = edge[1];
            graph[u].push_back(v);
            graph[v].push_back(u);
      }

      stack<int> st;
      unordered_set<int> visited;

      st.push(source);

      while (!st.empty()){

            int curr = st.top();
            st.pop();

            if (curr == destination)
                              return true;

            if (visited.find(curr) != visited.end())
                  continue;

            visited.insert(curr);

            for (int neighbor :graph[curr])
                        st.push(neighbor);
      }

      return false;
}


void allPathsDfs(int curr, int dest, vector<vector<int>>& graph, vector<vector<int>>& ret,vector<int> & path ){

      if (curr == dest ){
            
            ret.push_back(path);
            return ;
      }

      for (const auto & neighbor : graph[curr] ){
            
            path.push_back(neighbor);

            allPathsDfs(neighbor, dest, graph,ret,path );
            path.pop_back();
      }

return ;

}


//Backtracking 

 vector<vector<int>> allPathsSourceTarget(vector<vector<int>>& graph) {

      vector<vector<int>> result;
      
      vector<int> path{0};
      
      allPathsDfs(0 ,graph.size()-1 ,graph ,result,path);


      return result;
 }




class SolutionCloneGraph {
public:

      class Node {
      public:
      int val;
      vector<Node*> neighbors;
      Node() {
            val = 0;
            neighbors = vector<Node*>();
      }
      Node(int _val) {
            val = _val;
            neighbors = vector<Node*>();
      }
      Node(int _val, vector<Node*> _neighbors) {
            val = _val;
            neighbors = _neighbors;
      }
      };


    unordered_map< Node* , Node* > visited;

    Node* cloneGraph(Node* node) {
        
            if (!node)  
                  return nullptr;

            if (visited.find(node) != visited.end()) 
                  return visited[node];
            
            Node* newNode  = new Node(node->val); 
            visited[node] = newNode;

            for (const auto & neighbor : node->neighbors){                 
                  newNode->neighbors.push_back(cloneGraph(neighbor)) ;
            }

            return newNode;
    }
};


class FindItinerarySolution {
public:
 vector<string> findItinerary(vector<vector<string>>& tickets) {

      for (const auto & ticket: tickets){
            graph[ticket[0]].push_back(ticket[1]);
      }

      for ( auto & pair :graph) {
            sort(pair.second.begin(),pair.second.end());
            visited[pair.first] = vector<bool>(pair.second.size(),false);      
      }
      ticketCount = tickets.size();
      dfs("JFK");
      return result;
 }


private: 

unordered_map<string,vector<string>> graph;
unordered_map<string,vector<bool>> visited;
vector<string> result;
bool bFound = false;
int ticketCount = 0;

bool dfs (const  string & from){
      
      result.push_back(from);

      if (result.size() == ticketCount + 1){
            bFound = true;
            return bFound;
      } 


      auto & dest  =  graph[from];
      auto & bitMap = visited[from];

      for ( int i = 0  ; i < dest.size(); i++ ){
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


class LeadsToDestinationSolution {
public:

bool leadsToDestination(int n, vector<vector<int>>& edges, int source, int destination) {

      graph.resize(n);
      visited.resize(n,false);

      for (const auto &   edge : edges ){
            graph[edge[0]].push_back(edge[1]); 
      }     
      return dfs(source,destination);
}

private:

vector<bool> visited; 

bool dfs(int curr ,int dest){

      // If the node has no outgoing edges, it must be the destination
      if (graph[curr].empty())
            return curr == dest;

      // If the node is currently being visited, a cycle is detected
      if (visited[curr] == true)
            return false;
      visited[curr] = true; 
      
      for (auto & neighb : graph[curr] ){
            if ( !dfs(neighb,dest))
                  return false;
      }
      visited[curr] = false;
      return true;
}

vector<vector<int>> graph;

};


class NumIslandsSolution {
public:
    
    int numIslands(vector<vector<char>>& grid) {
      m = grid.size();
      n = grid[0].size();
      islands = 0;

      for (int i = 0; i < m; i++)
            for (int j = 0; j < n ; j++)
                  if (grid[i][j] ==  '1'){
                        islands++;
                        dfs(grid,i,j);
                  }
            
      return islands;
    }

private:

void dfs(vector<vector<char>>& grid, int i, int j){

      grid[i][j] = '0';

      if (i + 1 < m && grid[i+1][j] == '1' ) dfs(grid,i+1,j);
      if (j + 1 < n && grid[i][j+1] == '1' ) dfs(grid,i,j+1);
      if (i - 1 >= 0 && grid[i-1][j] == '1' ) dfs(grid,i-1,j);
      if (j -1 >= 0 && grid[i][j-1] == '1' ) dfs(grid,i,j-1);
      return ;
}

int islands;
int m;
int n;

};


class FindMinHeightTreesSolution {

public:
      vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges){

            if (n==1)
                  return {0};

            numOfNodes = n;
            vector<vector<int>> graph;
            vector<int> result;
           
            graph.resize(n);
            //build graph  for dfs
            for ( const auto & edge : edges){
                  graph[edge[0]].push_back(edge[1]);
                  graph[edge[1]].push_back(edge[0]);
            }
            //vector<bool> visited(n,false);
            int minHeight = n;
            for (int i =0; i < n; i++){
                 
                  int height = treeHeight(i,-1,graph);

                  if (height < minHeight ){
                        minHeight = height ;
                        result.clear();
                        result.push_back(i);
                  } else if (height == minHeight) {
                result.push_back(i);
            }
            }

            return result;

      }

private:
      
      int numOfNodes;
      int treeHeight (int node ,int parent  , vector<vector<int>> & graph){
                     
            int depth = 0;
            for (const auto & leaf: graph[node]){
                  if (leaf != parent){
                         depth = max(depth, 1 + treeHeight(leaf,node, graph));
                  }
            }
            return depth;    
      }

};


class PatternsToSolveLeetCode15 {


//1. Prefix Sum pattern
public:
    PatternsToSolveLeetCode15(vector<int>& nums) {
        if (nums.empty())
            return;
        sums.resize(nums.size());
        sums[0] = nums[0];
        for (int i = 1; i < nums.size(); ++i)
            sums[i] = sums[i - 1] + nums[i];
    }
    
    int sumRange(int left, int right) {
        if (left == 0)
            return sums[right];
        return sums[right] - sums[left - 1];
    }

private:
    vector<int> sums;

 int findMaxLength(vector<int>& nums) {
        unordered_map<int,int> map;
        int maxLength = 0;

        int size = nums.size();
        if (nums.empty())
            return 0;
      
        sums.resize(size);
        
        //think of base case for edges can help 
        // Base case: prefix sum 0 at index -1
        map[0] = -1;
        sums[0] = nums[0] == 1? 1:-1;

        for (int i =1; i <size; i ++)    
            sums[i] = sums[i-1] +  (nums[i] == 1? 1:-1);

        for (int i = 0 ; i < size; i++ ){
            
            if(map.find(sums[i]) == map.end() )       
                  map[sums[i]] =  i;
            else{
                 maxLength= max(maxLength,i- map[sums[i]]);
            }
        }

        return maxLength;
    }

int subarraySum(vector<int>& nums, int k) {
       
        int size = nums.size();
        vector<int> sums(size +1,0);
        int count = 0;
        sums[0] = 0;
       
      //build prefix sum.
        for (int i = 1; i <=  size; i++){     
            sums[i] = sums[i-1] +nums[i-1];
        }
        //looping and using the prefix sum to calculate sub array the brute force
        // is to to calculate for the range
        for (int i = 0; i < size; i++)
            for (int j =i+1; j<= size; j++ ){
                  if ( k == sums[j] - sums[i])
                        count++;
            }
    return count;
    }
    
int subarraySumBrueteForce(vector<int>& nums, int k) {
      //calculating  sum on the fly 

      int size = nums.size();
      int count = 0;
      for (int start = 0 ; start < size; start++ ){
            int sum = 0;
            for (int end = start ; end < size; end++ ){
                  sum += nums[end]; //savinf the extra loop 
                  if (sum == k)
                        count++;      
            }
      }
      return count;
}

int subarraySumhasMap(vector<int>& nums, int k) {

      int size = nums.size();
      int count =0;
      unordered_map<int,int> sumMap;
      int sum =0;
      sumMap[0] = 1;
      
      for (int i =0; i <size; i++){
            sum += nums[i];

            if (sumMap.find(sum - k) != sumMap.end())       
                  count += sumMap[sum - k]; // we are adding all frequencies every time we find a match. 

            sumMap[sum]++;  
      }
      return count;
}

//2. Two pointers Pattern

 vector<int> twoSum(vector<int>& numbers, int target) {

      int size = numbers.size();
      vector<int> res(2,0);
      
      for (int i =0, j= size-1; i <j;  ){
            int sum = numbers[i] +  numbers[j];
            if (  sum == target    ){
                  res[0] = i+1;
                  res[1] = j+1;
                  return res;
            }
            else if (sum > target)
                        --j;
                  else  
                        i++;
      }

      return res;
 }

vector<int> twoSumBinarySearch(vector<int>& numbers, int target) {

      int size = numbers.size();

      for (int i = 0; i < size; i++){

            int complementary = target - numbers[i];

            //binary search complementary. /lgn performance
            int start = i +1;
            int end = size -1;
            
            while (start <= end){

                  int mid = start + (end-start) /2;
                  if (numbers[mid] == complementary)
                        return {i+1,mid+1};
                  else if (numbers[mid] < complementary)
                        start = mid +1;
                  else
                        end = mid -1;

            }
      }

return {};

}

 vector<vector<int>> threeSum(vector<int>& nums) {
      vector<vector<int>> res;
      //sort vector
      int size = nums.size();
      sort(nums.begin(),nums.end());
      
      // since the aray sorted we can only search the nums right to i we 
      //avoid duplicates combination by that 
      for (int i = 0; i < size - 2; i++   ){ //last 2 dont count
            if (i > 0 && nums[i] == nums[i-1]) continue;  //fix a number and skip duplicated after
            // i = i+1 wont work since you want to fix a num for i but j can be the same number but not i again    
            //so we want to fix the number first.
            int target = -nums[i];

            int start = i + 1;
            
            int end = size - 1;
              
            while (start < end){

                  int currSum = nums[start] + nums[end];
                  
                  if (currSum == target){
                        res.push_back({nums[i], nums[start], nums[end]});

                        start++;
                        end--;
                        while (start < end && nums[start] == nums[start - 1]) start++;
                        // skip duplicates of found number 
                        while (start < end && nums[end] == nums[end + 1]) end--;
                  }    
                  else if (currSum < target)
                        start++;
                  else
                        end--;
            }
      }  
return res;
      // fixed number and sum 2 problem hashmap/  
 }

//binary search
 vector<vector<int>> threeSumBinarySearch(vector<int>& nums) {
      vector<vector<int>> res;
      //sort vector
      int size = nums.size();
      sort(nums.begin(),nums.end());
      
      // since the aray sorted we can only search the nums right to i we 
      //avoid duplicates combination by that 
      for (int i = 0; i < size - 2; i++   ){ //last 2 dont count
            if (i > 0 && nums[i] == nums[i-1]) continue;  //fix a number and skip duplicated after
            // i = i+1 wont work  
            
            for (int j = i + 1; j < size - 1; j++){ // size -1 since k should be after
                  if ( (j> i+1) && nums[j] == nums[j-1]) continue;  //skip duplicated

                  int start = j + 1;
            
                  int end = size - 1;

                  int target = -nums[i]  -nums[j];

                  while (start <= end){
                        int mid = start + (end - start) /2 ;

                        if (nums[mid] == target){
                              res.push_back({nums[i],nums[j],nums[mid]});
                              break;
                        }
                        else if (nums[mid] < target)
                                    start = mid+1;
                              else
                                    end = mid -1;
                  }
             }  
      }
      return res;
      // fixed number and sum 2 problem hashmap/  
 }

//Hash Set
 vector<vector<int>> threeSumHashSet(vector<int>& nums) {
      vector<vector<int>> res;
      //sort vector
      int size = nums.size();
      sort(nums.begin(),nums.end());
      
      // since the aray sorted we can only search the nums right to i we 
      //avoid duplicates combination by that 
      
      for (int i = 0; i < size - 2; i++   ){ //last 2 dont count
            unordered_set<int> set;
            if (i > 0 && nums[i] == nums[i-1]) continue;  //fix a number and skip duplicated after
            
            for (int j = i+1; j < size ; j++){  // k could be before or after j

                  int complement = -(nums[i]  + nums[j]);

                  if (set.find(complement) != set.end()){
                        res.push_back({nums[i],nums[j],complement});
                         while (j + 1 < size && nums[j] == nums[j+1] ) j++;  // j+1 means k ,  
                  }
                  
                  set.insert(nums[j]); // add any j since it can be a complement 
             }  
      }
      return res;
      // fixed number and sum 2 problem hashmap/  
 }

//Brute Force
int maxAreaBruteForce(vector<int>& height) {

      int max = 0;
      int size = height.size();

      for (int i = 0; i < size -1; i++ ){
            for (int j = i + 1;  j <  size; j++ ){
                  max = std::max(max, (j-i) * (std::min(height[i],height[j])))   ;

            }
      }
return max;
}

int maxAreaTwoPointers(vector<int>& height) {

      //moving the shorter line inward
      //by doing that we skip all permutuations that are not relevant since the shorter line always 
      //dominates the size of the container. 
      int max = 0;
      int size = height.size();

      int i = 0 , j = size -1;
      while ( i < j ){
            
            max = std::max(max, (j-i) * (std::min(height[i],height[j])));
            if( height[i] > height[j])
                  j--;
            else 
                  i++;
      }
return max;
      

}

//Sliding window

double findMaxAverage(vector<int>& nums, int k) {

       int size = nums.size();
      
      double windowSum = 0.0;

      for (int i = 0; i < k; i++ )
            windowSum += nums[i];
      double max = windowSum;

      for (int i = 1; i <= size - k; i++){
            
            windowSum += (-(nums[i-1]) + nums[i+k-1] );

            max =std::max(windowSum,max);
      }
      return max/ k;
}

//BruteForce
int lengthOfLongestSubstringBF(string s) {

      if (s.empty())
            return 0;
      
      int size = s.size();

      if (size == 1)
            return 1;

      int maxLen = 0;
      // maximum length until a duplicate is found
      for (int i = 0 ; i < size; i++){
            unordered_set<char> seen; //reset for every i
            int currLen = 0;
            
            for (int j = i  ; j < size ; j++){
                  
                  if (seen.find(s[j]) != seen.end()  )
                        break;      //duplicate found; 
                  currLen++;
                  seen.insert(s[j]);
            }

            maxLen =  std::max(maxLen, currLen);
      }
      return maxLen;
}
//sliding window approach saves from recounting the length
int lengthOfLongestSubstring(string s) {

      if (s.empty())
            return 0;
      
      int size = s.size();

      if (size == 1)
            return 1;

      int maxLen = 0;
      // maximum length until a duplicate is found
      
      unordered_map<char,int> prevLoc; // Map to store the last seen index of each character
      int currLen = 0;
      int start = 0 ;
      // the end is progressing the start jums between duplicates (windows)
      for (int end = 0 ; end  < size; end ++ ){
            char currCh = s[end];

             // If the character is already seen and within the current window
            if (prevLoc.find(currCh) != prevLoc.end() && prevLoc[currCh] >= start ){
                  start = prevLoc[currCh] +1; //Move the start to the right of the duplicate
            }

            // Update the last seen index of the current character
            prevLoc[currCh] = end;
      
            maxLen =  std::max(maxLen, end - start +1);
      }
      return maxLen;
}




//4. Fast and slow pointers array/Linked List

bool hasCycle(ListNode *head){

      ListNode * slow = head;
      ListNode * fast = head;

      while (fast  &&   fast->next  ){ // if null we know not a cycle also this is the step .
            
            slow = slow-> next;
            fast = fast->next->next;
            
            if (fast == slow)
                  return true;
      }

      return false;

}



//hash set
bool hasCycleSet(ListNode *head) {
      unordered_set<ListNode *> set;

      ListNode * curr = head;

      while (curr != NULL){
            if (set.find(curr) != set.end())
                  return true;
            
            set.insert(curr);
            curr = curr->next;
      }

      return false;
}



bool isHappy(int n) {
      unordered_set<int> set; 
      
      while (1){
            int sum;
            sum =0;
            
            while( n){
                  int digit = n%10;
                  n/=10;
                  sum +=(pow(digit,2));
            }
            
            n = sum;
            if (sum == 1  )
                  return true;
            else if (set.find(sum) != set.end() )
                        return false;
                  else
                        set.insert(sum);
      }
}     
 
/* 
 int findDuplicate(vector<int>& nums) {

      int n =  nums.size();
 }
*/

 int removeDuplicates(vector<int>& nums) {

            
            //{0,0,1,1,1,2,2,3,3,4};
              int insertIndex = 1;

    for (int i = 1; i < nums.size(); i++) {
        if (nums[i] != nums[i - 1]) {
            nums[insertIndex] = nums[i];
            insertIndex++;
        }
    }
    return insertIndex; // This directly gives the new length
}

 ListNode* reverseBetween(ListNode* head, int left, int right) {

      return head;
 }

 ListNode* swapPairs(ListNode* head) {
  
      return head;     
 }

//6. Monotonic(increasing /decreasing)  Stack
vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2) {


      return {};
}




//7. K largest  /k smallest /most frequent 
int findKthLargest(vector<int>& nums, int k) {

      int size = nums.size();

      priority_queue<int,vector<int>,greater<int>> minHeap;

      for (int num : nums) {
            minHeap.push(num); // Add the current number to the heap
            if (minHeap.size() > k) 
                  minHeap.pop(); // Remove the smallest element if size exceeds k
      }
      return minHeap.top();
}

vector<int> topKFrequent(vector<int>& nums, int k) {

      unordered_map<int,int> freqMap;
      vector<int> result;
      for (int num:nums) 
            freqMap[num]++;

      priority_queue<pair<int,int>,vector<pair<int,int>>,greater<pair<int,int>>> minHeap;

      for (const auto & pair :freqMap){

            int num = pair.first;
            int freq = pair.second;

            minHeap.push({freq,num});
            if (minHeap.size() > k)
                  minHeap.pop();
      }

      while(!minHeap.empty()){

            result.push_back(minHeap.top().second);
            minHeap.pop();
      }

      return result;
}

vector<int> topKFrequentBucketSort(vector<int>& nums, int k) {

      vector<int> result;
      unordered_map<int,int> freqMap;
      int size = nums.size();

      for (int num:nums)
            freqMap[num]++;

      vector<vector<int>> buckets(size+1);  //  that is the maximum freq if all were the same number

      for (const auto & pair:freqMap){
            int num =  pair.first;
            int freq =  pair.second;
            buckets[freq].push_back(num);
      }


      for (int i = size; size > 0 && k > result.size(); i--   )
            for (int num:buckets[i]){
                  
                  result.push_back(num);
                  if (result.size() == k)
                        break;
            }
      return result;
}

 vector<vector<int>> kSmallestPairs(vector<int>& nums1, vector<int>& nums2, int k) {
 
      vector<vector<int>> result;

      struct heapStruct {
            int sum;       // Task priority (lower value = higher priority)
            int x;
            int y;
            heapStruct(int x, int y) : x(x), y(y),sum(x+y) {}
      };
      
      struct compareHeap{
            bool operator()(const  heapStruct & h1,const  heapStruct & h2 ){
                  return h1.sum < h2.sum;
            }
      };

      priority_queue<heapStruct,vector<heapStruct>,compareHeap> maxHeap; 

      for (int x:nums1)
            for (int y:nums2){
                  heapStruct h(x,y);
                  maxHeap.push(h);

                  if (maxHeap.size() > k  )
                        maxHeap.pop();
            }
      
      
      while(!maxHeap.empty()){
            heapStruct h = maxHeap.top();
            result.push_back({h.x,h.y});      
            maxHeap.pop();
            
      }
       reverse(result.begin(), result.end());
      return result;

 }    

vector<vector<int>> kSmallestPairsEfficent(vector<int>& nums1, vector<int>& nums2, int k) {
      

      vector<vector<int>> result;

      int m = nums1.size();
      int n = nums2.size();

      set<pair<int,int>> visited;

      priority_queue<pair<int,pair<int,int>>,vector<pair<int,pair<int,int>>>,greater<pair<int,pair<int,int>>> > minHeap;

      minHeap.push({ nums1[0] + nums2[0] ,{0,0} } );
      visited.insert({0,0});
      
      while (k-- > 0 && !minHeap.empty()   ){

            auto top =  minHeap.top();
            minHeap.pop();
            
            int i = top.second.first; 
            int j = top.second.second;
            result.push_back({nums1[i], nums2[j]});

            if ( i +1 < m && visited.find({i+1,j}) == visited.end()){
                  minHeap.push({ nums1[i+1] + nums2[j] ,{i+1,j} });
                  visited.insert({i+1,j});
            }


            if ( j +1 < n && visited.find({i,j+1}) == visited.end()){
                  minHeap.push({ nums1[i] + nums2[j+1] ,{i,j+1} });
                  visited.insert({i,j+1});
            }
      }
 return result;     
}


static bool compareIntervals(const vector<int>  & a , const vector<int> & b ){
      return a[0] < b[0];
}

//Overlapping Intervals
vector<vector<int>> merge(vector<vector<int>>& intervals) {

//Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
//Output: [[1,6],[8,10],[15,18]]

      vector<vector<int>> res;
      sort(intervals.begin(),intervals.end(),compareIntervals);

      res.push_back(intervals[0]);

      for (int i = 1; i < intervals.size() ; i ++ ){

            auto & last =  res.back();  
            if (intervals[i][0] <= last[1]  )
                 last[1] = max(intervals[i][1],last[1]);
            else     
                  res.push_back(intervals[i]);
             
      }

      return res;
}


 vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval) {

      //Input: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
      //Output: [[1,2],[3,10],[12,16]]
      vector<vector<int>> merged;

      if (newInterval.empty())
            return merged;

      if (intervals.empty() ){
            merged.push_back(newInterval);
            return merged;   
      }
      
      
      int left = 0 ;
      int right = intervals.size() - 1;  
      int newStart = newInterval[0];

      while (left <= right){
            int mid = left + (right - left)/2;

            auto & interval = intervals[mid];

      }


      return merged;

 }


//10. Binary Tree Traversal

vector<string> binaryTreePaths(TreeNode* root) {

//Input: root = [1,2,3,null,5]
//Output: ["1->2->5","1->3"]

return {};

}




};








class PatternsToSolveLeetCode8 {

public:

//Sliding Window
int maxSumSubarray(const vector<int>& nums, int k) {
    int n = nums.size();
    if (n < k) return -1;  // Invalid if array size is smaller than k

    int windowSum = 0;
    for (int i = 0; i < k; i++) {
        windowSum += nums[i];
    }

    int maxSum = windowSum;

    for (int i = k; i < n; i++) {
        windowSum += nums[i] - nums[i - k];
        maxSum = max(maxSum, windowSum);
    }

    return maxSum;
}


//Subset pattern 
void findSubsets(const vector<int>& nums, vector<int>& current, int index, vector<vector<int>>& result) {

      if (index == nums.size()){
            result.push_back(current); 
      }
      current.push_back(nums[index]);

      findSubsets(nums,current,index+1,result);

      current.pop_back();
      findSubsets(nums,current,index+1,result);
}

//Modified Binary Search
int search(vector<int>& nums, int target) {

      int size = nums.size();
      int left = 0;
      int right = size -1;

      while (left <= right){
            int mid = left + (right - left)/2;  //  == (left + right) /2

            if (  nums[mid] == target)
                  return mid;

            if (nums[left] <= nums[mid]) //left sorted can do binary search
                  if (nums[left] <= target && target < nums[mid] ) //on left
                        right = mid - 1;
                  else  
                        left = mid + 1; // on right
            else  // right sorted
                  if (nums[mid] < target && target <= nums[right] ) 
                        left = mid + 1;
                  else
                        right = mid -1;
      }           
      return -1;
}
//K largest from vector 
vector<int> kLargestElements(vector<int>& nums, int k) {

      vector<int> result(k);
      priority_queue<int,vector<int>,greater<int>> minHeap;

      for (const auto & num: nums ){
            minHeap.push(num);

            if (minHeap.size() > k )
                  minHeap.pop();
      }

      while (!minHeap.empty()){
            result.push_back(minHeap.top());
            minHeap.pop();
      }
      return result;
}


bool dfs(int course,vector<vector<int>>& graph,vector<int>& visited){
      //leafs stopping condition
      if (visited[course] == 1 ) //cycle detected
            return false;
      if (visited[course] == 2 ) //
            return true;

      visited[course] = 1; //visiitng now
      for (int next : graph[course] ){
            if (!dfs(next,graph,visited))
                  return false; //false propgates all the way  up  
      }

      visited[course] = 2; //mark as visited after traveresring
      // on all descendants no cycle detected at this point 
      
      return true;
}
 
bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {

      vector<int> visited(numCourses,0);

      vector<vector<int>> graph(numCourses);

      for (const auto & pre:prerequisites){
            graph[pre[1]].push_back(pre[0]);
      }
        
      for (int i = 0; i < numCourses ; i++)
            if (visited[i] == 0 && !dfs(i,graph,visited)) //independent subgraphs 
                  return false;

      return true;
}



vector<vector<int>> subsets(vector<int>& nums) {
      vector<vector<int>> result;
      vector<int> curr;
      findSubsets(nums,curr,0,result );
      return result;
}


vector<vector<int>> levelOrder(TreeNode* root) {

       vector<vector<int>> result;
      if (!root)
            return result;

      queue<TreeNode *> queue;

      queue.push(root);

      while (!queue.empty()){

            int size = queue.size();
            vector<int> level;
           
            for (int i = 0 ; i < size; i ++)
            {
                  TreeNode *  curr = queue.front();
                  level.push_back(curr->val);
                  queue.pop();
                  if (curr->left) queue.push(curr->left);
                  if (curr->right)queue.push(curr->right);
            }

            result.push_back(level);
      }
      return result;
}


};

class CodingInterviewPatterns{
public:


class GraphNode {
public:
    int val;
    vector<GraphNode*> neighbors;

    GraphNode(int _val) : val(_val) {}
};

//BFS
GraphNode* graphDeepCopyBFS(GraphNode* node) {
    if (!node) 
      return nullptr;

      GraphNode * copy = new GraphNode(node->val);
      unordered_map<GraphNode*,GraphNode*> visited; // map of all connection between the 2 graphs 
      
      queue<GraphNode*> queue;
      visited[node] = copy;
      queue.push(node);
    
      while (!queue.empty()){
            
            GraphNode * currNode = queue.front();
            queue.pop();
           
            for (const auto & neighbor:currNode->neighbors ){

                  if (visited.find(neighbor) == visited.end()){
                        
                        visited[neighbor] = new GraphNode(neighbor->val);
                        queue.push(neighbor);
                  }

                  visited[currNode]->neighbors.push_back(visited[neighbor]);
            }
      }
      return copy;
}

//DFS

GraphNode* graphDeepCopyDFS(GraphNode * node) {

      if (!node)
            return nullptr;

      unordered_map<GraphNode*, GraphNode* > visited; 
   
      GraphNode * copy  = DFSDeepCopy(node,visited);

      return copy;

}

GraphNode * DFSDeepCopy( GraphNode * node , unordered_map<GraphNode*, GraphNode* > & visited  ){

      if ( visited.find(node) != visited.end())
            return visited[node]  ;

      GraphNode * newNode = new GraphNode(node->val); 
      visited[node] = newNode;

      for (const auto & neighbor: node->neighbors ){
            
            newNode->neighbors.push_back(DFSDeepCopy(neighbor,visited));
      }

      return   newNode;
}


void countIslandsDFS(vector<vector<int>>& matrix, int i , int j) {

      //check boundries and 
      if (i < 0 || i >= matrix.size() || j < 0 || j >= matrix[0].size() || matrix[i][j] == 0 )
            return;

      matrix[i][j] = 0;

      countIslandsDFS(matrix,i+1,j);

      countIslandsDFS(matrix,i,j+1);
      
      countIslandsDFS(matrix,i-1,j);
      
      countIslandsDFS(matrix,i,j-1);
}

// Count Islands
int countIslands(vector<vector<int>>& matrix ) {

      if (matrix.empty() || matrix[0].empty())
            return 0;

      int islands = 0;
      for (int i = 0; i < matrix.size(); i ++ )
            for (int j = 0; j < matrix[0].size(); j ++ )
                if (matrix[i][j] ==1){
                        islands++;
                        countIslandsDFS(matrix,i,j);
                  }

return islands;

}

int matrixInfection(vector<vector<int>>& matrix) {

      if (matrix.empty() || matrix[0].empty()  )
            return 0;

      int rows = matrix.size();
      int cols = matrix[0].size();

      queue<pair<int,int>> queue;
      int ones = 0;
      int seconds = 0;
     
      for (int i = 0 ; i < matrix.size() ; i++)
            for (int j = 0 ; j < matrix[0].size() ; j++){
                  if (matrix[i][j] == 2)
                        queue.push({i,j});
                  
                  if (matrix[i][j] == 1)
                        ones++;
            }

        // If no uninfected cells exist
      if (ones == 0) {
         return 0;
      }

      while (!queue.empty())     {

            int levelSize = queue.size();
            bool infectionFound = false;
            for (int k = 0 ; k <  levelSize; k ++){ // give me level by level logic 

                  auto pos = queue.front();
                  int i = pos.first;
                  int j = pos.second;
                   
                  queue.pop(); 

                  if (i+1 < rows && matrix[i+1][j] == 1 ){
                        queue.push({i+1,j}); 
                        ones--; 
                        matrix[i+1][j] = 2; 
                        infectionFound = true;
                  }
                  
                  if (i-1 >=0 && matrix[i-1][j] == 1 ){
                        queue.push({i-1,j});
                        ones--; 
                        matrix[i-1][j] = 2; 
                        infectionFound = true;
                  }

                  if (j +1 < cols && matrix[i][j+1] == 1 ){
                        queue.push({i,j+1}); 
                        ones--; 
                        matrix[i][j+1] = 2; 
                        infectionFound = true;
                  }
                  if (j-1 >=0 && matrix[i][j-1] == 1 ){
                        queue.push({i,j-1}); 
                        ones--; 
                        matrix[i][j-1] = 2; 
                        infectionFound = true;
                  }
            
            }
            if (infectionFound)
                seconds++;
      }

      return ones == 0 ? seconds : -1;
}


bool bipartiteDFS(int node ,int color ,const vector<vector<int>>& graph , vector<int> & colors ){
     
      // If already colored, check if it's the same color
      if (colors[node] != 0 )
            return color == colors[node];
      
      // Color the current node
      colors[node] = color;

      for (int  neighbor:graph[node])
             if (!bipartiteDFS(neighbor, -color  , graph,colors ) )
                 return false;

      return true;
}

bool bipartite_graph_validation(const vector<vector<int>>& graph) {

      bool  isValid = true;

      int size = graph.size();
      vector<int> colors(size,0);

      for (int i = 0;i <  size && isValid; i++ ){
            if (colors[i] == 0){
                  isValid = bipartiteDFS(i,-1,graph,colors);
            }
      }

      return isValid;
}

int lipDFS(int i ,int j,int prevNum, const vector<vector<int>> & matrix , vector<vector<int>> & memo) {

      int rows = matrix.size();
      int cols = matrix[0].size();

      if (i < 0 || i >= rows || j < 0 || j >= cols || prevNum >= matrix[i][j]  )
            return 0;

      if (memo[i][j] != -1)
            return memo[i][j];
            
      vector<pair<int,int>> directions({{-1, 0}, {1, 0}, {0, -1}, {0, 1}});
      int res = 1; // if it's the last cell the value is 1

      for ( auto [dx,dy]:directions )
            res = std::max(res, 1 + lipDFS(i + dx, j + dy,matrix[i][j],matrix,memo ));

      memo[i][j] = res;
      return res;
}

int longest_increasing_path(const vector<vector<int>>& matrix) {

      if (matrix.empty() || matrix[0].empty() )
            return 0;
      
      int rows = matrix.size();
      int cols = matrix[0].size();
      
      vector<vector<int>> memo(rows,vector<int>(cols,-1));

      int maxPath = 0;
      for (int i = 0; i < rows;  i++)
             for (int j = 0; j < cols;  j++)
                  maxPath = std::max(maxPath,lipDFS(i,j,-1,matrix,memo));

      return maxPath;
}

//Shortest Transformation Sequence
int shortestTransformationSequence(string start, string end, vector<string>& dictionary) {
      // Convert dictionary to a set for O(1) lookup
      unordered_set<string> wordSet(dictionary.begin(),dictionary.end());
      
      if (wordSet.find(end) == wordSet.end())
            return -1;

      queue<pair<string,int>> queue;

      queue.push({start,1});

      while(!queue.empty()) {

            auto [currentWord,steps] = queue.front();
            queue.pop();

            if (currentWord == end)
                  return steps;

            // Try changing each character in the current word and search it
            for (int i = 0; i < currentWord.length(); ++i ){
                  string tempWord = currentWord;
                  for (char ch = 'a'; ch <= 'z'; ch++ ){
                        tempWord[i] = ch;

                        if (wordSet.find(tempWord) != wordSet.end()){
                              queue.push({tempWord,steps +1});
                              wordSet.erase(tempWord);// Remove to prevent revisiting
                        }
                  }
            }
      }
      return -1;
}

class MergingCommunities {
private:
    std::vector<int> parent, rank, size;

public:
    // Constructor
    MergingCommunities(int n) {
        // Initialize parent and size vectors
             
            size.resize(n,1);
            rank.resize(n,0);
            parent.resize(n);
            for (int i = 0; i < n; i++) 
               parent[i]= i; // Each node is its own parent
    }

      int find(int x){
            if (parent[x] == x)
                  return x;
            parent[x] = find(parent[x]);
            return parent[x];
      }
   
    // Connect two nodes
      void connect(int x, int y) {
        // Implementation for union operation
        int rootX = find(x);
        int rootY = find(y);

        if (rootX == rootY)
            return;

        if (rank[rootX] == rank[rootY] ){
            rank[rootX]++;
            parent[rootY] = rootX;
            size[rootX] += size[rootY];
            }
        else if (rank[rootX] > rank[rootY]) {
                  parent[rootY] = rootX;
                  size[rootX] += size[rootY];
            }
        else{ 
                  parent[rootX] = rootY;
                  size[rootY] += size[rootX];
            }
       return ;
    }

    // Get the size of the community of a given node
    int getCommunitySize(int x) {
        // Implementation to find the size of the community

        return size[find(x)];; // Placeholder
    }
};

};

class Headlands{

public:

/*
static bool compareChains(const  vector<pair<vector<int>,int>>  & a, const  vector<pair<vector<int>,int>> & b ){
       if (a.empty() || b.empty()) 
            return false;
      
      return a[0].second > b[0].second ;
}
*/







};


class SocialNetwork{
private:
    // A map of user name to their friends (set ensures no duplicate friends)
    unordered_map<string, unordered_set<string>> network;

public:
    // Add a new user to the network
    void addUser(const string& user) {
        if (network.find(user) == network.end()) {
            network[user] = unordered_set<string>();
            cout << user << " added to the network.\n";
        } else {
            cout << user << " already exists in the network.\n";
        }
    }

    // Add a friendship between two users
    void addFriendship(const string& user1, const string& user2) {
        if (network.find(user1) == network.end() || network.find(user2) == network.end()) {
            cout << "Both users must exist in the network.\n";
            return;
        }

        // Add the friendship (bidirectional)
        network[user1].insert(user2);
        network[user2].insert(user1);
        cout << "Friendship added between " << user1 << " and " << user2 << ".\n";
    }

    // Display a user's friends
    void displayUser(const string& user) const {
        if (network.find(user) == network.end()) {
            cout << user << " does not exist in the network.\n";
            return;
        }

        cout << user << "'s friends: ";
        if (network.at(user).empty()) {
            cout << "No friends yet.\n";
        } else {
            for (const auto& friendName : network.at(user)) {
                cout << friendName << " ";
            }
            cout << endl;
        }
    }

    // Suggest friends (friends of friends) for a user
    void suggestFriends(const string& user) const {
        if (network.find(user) == network.end()) {
            cout << user << " does not exist in the network.\n";
            return;
        }

        unordered_set<string> suggestions;
        const auto& friends = network.at(user);

        for (const auto& friendName : friends) {
            // Check each friend of the current user's friends
            for (const auto& fof : network.at(friendName)) {
                // If the friend of a friend is not the user and not already a direct friend
                if (fof != user && friends.find(fof) == friends.end()) {
                    suggestions.insert(fof);
                }
            }
        }

        // Display suggestions
        cout << "Friend suggestions for " << user << ": ";
        if (suggestions.empty()) {
            cout << "No suggestions available.\n";
        } else {
            for (const auto& suggestion : suggestions) {
                cout << suggestion << " ";
            }
            cout << endl;
        }
    }
};




ListNode* middleNodeHelper(ListNode* slow , ListNode * fast) {

if (!fast || !fast->next)
      return slow;

      return middleNodeHelper(slow->next, fast->next->next);

}



ListNode* middleNode(ListNode* head) {

//recursive
if (!head || !head->next )
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

class DiameterSolution{ 

private:

int maxDiameter = 0;

int diameterOfBinaryTreeHelper(TreeNode* root) {

      if (root == nullptr)
            return 0;

      int leftHeight  = diameterOfBinaryTreeHelper(root->left);
      int rightHeight = diameterOfBinaryTreeHelper(root->right);

      maxDiameter = max(maxDiameter,leftHeight +rightHeight);

      return max(leftHeight,rightHeight) +1;


}

int diameterOfBinaryTree(TreeNode* root) {

      /* 1 recursion not 2 
      if (root == nullptr)
            return 0;

      diameterOfBinaryTreeHelper(root);

return maxDiameter;
*/


/**/
      if (root == nullptr)
            return 0;

      //height of each subtree = edges to the root 
      int leftHeight = treeHeight(root->left);
      int rightHeight = treeHeight(root->right);

      int diameterThroughRoot = leftHeight + rightHeight;

      return max(max(diameterOfBinaryTree(root->left),diameterOfBinaryTree(root->right)),diameterThroughRoot) ;
}

int treeHeight(TreeNode* root){
      if (root ==nullptr)
            return 0;

      return max(treeHeight(root->left),treeHeight(root->right)) +1; 
}





};


class Node{

public:
      //single linked list
      /*
      Node():val(0),next(NULL){}
            
      Node(int num):val(num),next(NULL){}
        */    

      //double
      Node():val(0),next(NULL),prev(NULL),child(NULL),random(NULL),left(NULL),right(NULL){}
            
      Node(int num):val(num),next(NULL),prev(NULL),child(NULL),random(NULL),left(NULL),right(NULL){}

      int val;
      Node * left;
      Node * right;
      Node * next;
      Node * prev;
      Node * child;
      Node* random;
};


 // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
        if (root == nullptr)
            return "null";
      string str =  to_string(root->val) + ",";

      str+= serialize(root->left);
      str+=",";
      str+= serialize(root->right);

      return str;
    }

    
int treeHeight(TreeNode* root){

      if (root == nullptr)
            return 0;

      return max(treeHeight(root->left),treeHeight(root->right)) + 1;
}

bool isBalancedHelper(TreeNode* root, int & height){

      if (root == nullptr){
            height = 0;
            return true;
      }

      int leftHeight;
      int rightHeight;

      bool  bLeftBalanced  =  isBalancedHelper(root->left,leftHeight);
      bool  bRightBalanced =  isBalancedHelper(root->right,rightHeight);
      
      if( bLeftBalanced && bRightBalanced && abs(leftHeight - rightHeight) <2){
            height = max(leftHeight,rightHeight) + 1; 
            return true;
      }
      return false;
      
}

bool isBalanced(TreeNode* root) {
//Bottom up

int height;

if (root == nullptr)
      return true;
      
return isBalancedHelper(root,height);





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


//BST
class LCA {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {


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

     //Iterative 
      if (root == nullptr || root == p || root == q  )
            return root;
      int pVal = p->val;
      int qVal = q->val;
            
      while (root != nullptr){
            
            if (root == p || root == q)   
                  return root;

            int rVal = root->val;

            if ( (pVal > rVal && qVal < rVal)  || (pVal < rVal && qVal > rVal) )
                   return root;

            if ( (pVal > rVal && qVal > rVal)   )
                  root= root->right;
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




//Binary Tree not BST

//base case 
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




void connectHelper(Node* left,Node *right) {

if (!left || !right)    
      return  ;

left->next = right;

connectHelper(left->left, left->right);
connectHelper(left->right, right->left);
connectHelper(right->left, right->right);


}


Node* connect(Node* root) {

//not a prefect tree 

if (!root)
      return root;

Node * levelStart = root;

while (levelStart != nullptr){

      Node* curr = levelStart;  //start of each level
      Node * prev = nullptr;
      Node * nextLevelStart = nullptr;

      while (curr != nullptr){
            if (curr->left != nullptr){
                  if (prev !=nullptr)
                        prev->next = curr->left;
                  else
                        nextLevelStart = curr->left;

                  prev = curr->left;
            }

            if (curr->right != nullptr) {
                  if (prev != nullptr) 
                    prev->next = curr->right;
                  else 
                    nextLevelStart = curr->right;

                  prev = curr->right;
            }

            curr = curr->next;
      }

      levelStart =nextLevelStart;
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


bool countUnivalSubtreesHelper(TreeNode* root,int & count) {

        if (!root)  
            return true;

      bool isLeftSub = countUnivalSubtreesHelper(root->left,count);
      bool isRightSub = countUnivalSubtreesHelper(root->right,count);

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



class SolutionA {
public:
   
    TreeNode* buildTreePost(vector<int>& inorder, vector<int>& postorder) {
        // Store the indices of inorder elements for quick lookup
        unordered_map<int, int> inorder_map;
        for (int i = 0; i < inorder.size(); ++i) {
            inorder_map[inorder[i]] = i;
        }
        return buildTreeHelper(inorder, postorder, 0, inorder.size() - 1, 0, postorder.size() - 1, inorder_map);
    }

       TreeNode* buildTreeHelper(vector<int>& inorder, vector<int>& postorder, int inStart, int inEnd, int postStart, int postEnd, unordered_map<int, int>& inorder_map) {
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


      
      TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
      // Store the indices of inorder elements for quick lookup
        unordered_map<int, int> inorder_map;
        for (int i = 0; i < inorder.size(); ++i) {
            inorder_map[inorder[i]] = i;
        }

      return buildTreeHelperPre(preorder ,inorder , 0, preorder.size() - 1, 0, inorder.size() - 1, inorder_map);
      
      }
      
      TreeNode* buildTreeHelperPre(vector<int>& preorder, vector<int>&  inorder, int preStart , int preEnd , int inStart, int inEnd, unordered_map<int, int>& inorder_map) {
            if (inStart > inEnd || preStart > preEnd) 
                return nullptr;
        
            // Create the root node from the last element of postorder
            TreeNode* root = new TreeNode(preorder[preStart]);

             // Find the index of the root value in the inorder traversal
            int rootIndexInInorder = inorder_map[root->val];
            int leftSubtreeSize = rootIndexInInorder - inStart;

            // Recursively build left and right subtrees
            root->left  = buildTreeHelperPre(preorder , inorder, preStart +1 , preStart + leftSubtreeSize, inStart, rootIndexInInorder - 1, inorder_map);
            root->right = buildTreeHelperPre(preorder,  inorder, preStart + leftSubtreeSize+1, preEnd, rootIndexInInorder + 1, inEnd , inorder_map);
        
            return root;

      }


   
};



int countUnivalSubtrees(TreeNode* root) {

      if (!root)        
            return 0;

    int count =0 ;
    
    countUnivalSubtreesHelper(root,count);
    return count;
       
    }





bool hasPathSum(TreeNode* root, int targetSum) {

//iterative stack
 if (!root) 
  return false; // If the tree is empty, there's no path
    
    stack<pair<TreeNode*, int>> nodeStack;
    nodeStack.push({root, root->val}); // Initialize the stack with the root node and its value

    while (!nodeStack.empty()) {
        auto topElem = nodeStack.top();
            
        nodeStack.pop();

        if (!topElem.first->left && !topElem.first->right && targetSum ==topElem.second)
            return true; 

        if (topElem.first->left)
            nodeStack.push({topElem.first->left,topElem.second+ topElem.first->left->val});

        if (topElem.first->right)
            nodeStack.push({topElem.first->right, topElem.second + topElem.first->right->val});

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



bool isMirror(TreeNode* t1,TreeNode* t2) {

if (t1 == nullptr && t2 == nullptr)
      return true;

if (t1 == nullptr || t2 == nullptr)
      return false;

return t1->val == t2->val  && isMirror(t1->left,t2->right) 
&& isMirror(t1->right, t2->left);

}
 
bool isSymmetric(TreeNode* root) {

//return isMirror(root,root);


//iteratrive queue every 2 consecutive nodes should be equal

queue<TreeNode *> queue;

queue.push(root->left);
queue.push(root->right);

while (!queue.empty()){

TreeNode * t1 = queue.front();
queue.pop();
TreeNode * t2 = queue.front();
queue.pop();

if (!t1 && !t2 )
      continue;

if (!t1 || !t2 )
      return false;
if(t1->val != t2->val)
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


TreeNode* invertTree(TreeNode* root) {

//BFS

if (!root)
      return root;

queue<TreeNode*> queue;

queue.push(root);
//we add to queue row by row 
while (!queue.empty()){

      int size =  queue.size();

      for (int i =0 ; i < size; i++){

            TreeNode * node = queue.front();
            queue.pop();
            TreeNode* temp = node->left;
            node->left = node->right;
            node->right = temp;

            if(node->left)
                  queue.push(node->left);
            
            if(node->right)
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


int maxDepth(TreeNode* root) {

/*      //DFS
      if (!root)        
            return 0;

         return    max( maxDepth(root->left) +1 ,maxDepth(root->right) +1   );  
*/

//itertive 
  stack<pair<int, TreeNode*>> stack;
      

      if (!root)
            return 0;

      stack.push({1,root});

      int maxDepth = 0;
      while (  !stack.empty()){
            //depth, node
            auto stackTop = stack.top();

            stack.pop();
            maxDepth = max(stackTop.first,maxDepth);

            if (stackTop.second->left)         
                  stack.push({stackTop.first +1,stackTop.second->left});
            
            if (stackTop.second->right)         
                  stack.push({stackTop.first +1,stackTop.second->right});
            

      }      
      return maxDepth;

}




void levelOrderHelper(TreeNode* root,int level,vector<vector<int>> & result) {

if (!root)
      return;

if (result.size() <= level){
      result.push_back(vector<int>{});

}
//BFS inside recursion 
result[level].push_back(root->val);

levelOrderHelper(root->left,level+1,result);
levelOrderHelper(root->right,level+1,result);

}

vector<vector<int>> levelOrder(TreeNode* root) {


vector<vector<int>> result;

if (!root)
      return result;
//levelOrderHelper(root, 0, result);

//BFS
//
queue<TreeNode*> queue;

queue.push(root);
//we add to queue row by row 
while (!queue.empty()){

      int size =  queue.size();
      vector<int> level;

      for (int i =0 ; i < size; i++){

            TreeNode * node = queue.front();
            queue.pop();
            level.push_back(node->val);

            if(node->left)
                  queue.push(node->left);
            
            if(node->right)
            queue.push(node->right);
      }

      result.push_back(level);
}
      
return result;

 }




class Solution {
public:


void postOrderTraversalHelper(TreeNode* root,vector<int> &result){
            
            if (!root ) 
                  return;

            
            postOrderTraversalHelper(root->left,result);                  
            postOrderTraversalHelper(root->right,result);
            result.push_back(root->val);

      }

vector<int> postorderTraversal(TreeNode* root) {

vector<int> result;
postOrderTraversalHelper(root,result);    

      


return result;
}

void inOrderTraversalHelper(TreeNode* root,vector<int> &result){
            
            if (!root ) 
                  return;

            
            inOrderTraversalHelper(root->left,result);      
            result.push_back(root->val);
            inOrderTraversalHelper(root->right,result);

      }

vector<int> inorderTraversal(TreeNode* root) {

vector<int> result;
//inOrderTraversalHelper(root,result);      

      stack<TreeNode*> stack;
      TreeNode * current = root;

      if (!root)
            return result;
      
      while ( current || !stack.empty()){

            while(current){
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

      

      void preorderTraversalHelper(TreeNode* root,vector<int> &result){
            
            if (!root ) 
                  return;

            result.push_back(root->val);
            preorderTraversalHelper(root->left,result);      
            preorderTraversalHelper(root->right,result);

      }

    vector<int> preorderTraversal(TreeNode* root) {
        
      vector<int> result;
      stack<TreeNode*> stack;

      if (!root)
            return result;
      
      stack.push(root);

      while (!stack.empty()){

            TreeNode * node = stack.top();
            stack.pop();

            result.push_back(node->val);

            if (node->right)
                  stack.push(node->right);
            if (node->left)
                  stack.push(node->left);

      }      

      


      ///preorderTraversalHelper(root,result);


return result;
    }
};


 bool isPalindrome(string s) {

      //s ="A man, a plan, a canal: Panama";

      string filteredString;

      for (char ch:s)
            if (isalnum(ch))
                  filteredString.push_back(tolower(ch));


      string reversedString(filteredString);

      reverse(reversedString.begin(),reversedString.end());

      return filteredString.compare(reversedString);

}


//paranthesis 
bool isValid(string s) {

      if (s.length() %2 == 1)
            return false;

      stack<char> stack;

      map<char,char> map;
      map['('] = ')';
      map['{'] = '}';
      map['['] = ']';

      for (char ch:s){
            if (ch == '(' || ch == '[' || ch == '{')
                  stack.push(ch);
      
            if (ch == ')' || ch == ']' || ch == '}'){
                  if (stack.empty())
                        return false;
             
                  if (map[stack.top()] != ch   )
                        return false;
                  stack.pop();
            }
      }

      if (!stack.empty())
            return false;
      
      return true;
}



vector<string> findRestaurant(vector<string>& list1, vector<string>& list2) {

map<int,vector<string>> map;

vector<string>& smallerList = list1.size() > list2.size() ? list2:list1; 

vector<string>& greaterList = list1.size() > list2.size() ? list1:list2; 

//vector<string> * smasfsdlerList = list1.size() > list2.size() ? &list2:&list1; 

for (int i = 0; i < smallerList.size(); i++){

      auto it =  find(greaterList.begin(),greaterList.end(),smallerList[i]);
      if (it != greaterList.end()){
            int j = distance(greaterList.begin(),it);

            map[i+j].push_back(smallerList[i]);
      }
}     

return  (map.begin()->second);



 }


bool isIsomorphic(string s, string t) {

      unordered_map<char,char> mapStoT ;
      unordered_map<char,char> mapTtoS ;
      //badc 
      //baba 
      //paper
      //title
      if (s.length() != t.length())
            return false;


      for (int i = 0; i < s.length(); i++ ){

            char chS = s[i];
            char chT = t[i];
            
            int countS = mapStoT.count(chS);
            int countT = mapTtoS.count(chT);
            
            if (countS == 0)
                  mapStoT[chS] = t[i];
            
            if (countT == 0)
                  mapTtoS[chT] = s[i];
            
            if (mapStoT[chS] != chT || mapTtoS[chT] != chS     )
                  return false;
            
      }

      return true;

}


vector<vector<string>> groupAnagrams(vector<string>& strs) {

vector<vector<string>> ret;
//strs = {"eat","tea","tan","ate","nat","bat"};
array<int,26> arr{};
unordered_map<string,vector<string>> map;

for(string word: strs){

      for (char ch: word){
            arr[ch - 'a']++;
      }
      string key;
      for (int i:arr){
            key.push_back(i +'0');
            key.push_back('#');
      }

      map[key].push_back(word);      
      fill(arr.begin(),arr.end(),0);
      key.clear();
}

for (auto it = map.begin(); it != map.end(); it++){
      ret.push_back(it->second);
}

return ret;

/*sorting
vector<vector<string>>  ret;
unordered_map<string,vector<string>> map;
strs = {"eat","tea","tan","ate","nat","bat"};
for (string word: strs){
      
      string sortedWord = word; 
      sort(sortedWord.begin(),sortedWord.end());
      map[sortedWord].push_back(word);
}

 for ( auto  it =  map.begin() ; it != map.end(); it++  ){
      ret.push_back(it->second);
 }


return ret;
*/



}








vector<int> intersection(vector<int>& nums1, vector<int>& nums2) {

vector<int> ret;

if (nums1.size() > nums2.size()){
      unordered_set<int>  set(nums2.begin(),nums2.end());
      for (int num: nums1){
            auto it = set.find(num);
            if ( it != set.end()){
                  ret.push_back(num);
                  set.erase(it);
            }
      } 
}


return ret;
}


int singleNumber(vector<int>& nums) {

//use XOR

int ret = 0;

for (int num:nums)
      ret^=num;
return ret;


//2∗(a+b+c)−(a+a+b+b+c)=c
/* MAth
set<int> set;
int sum = 0;
int uniqueSum = 0;

for (int num: nums ){

      sum += num;
      auto result =  set.insert(num);

      if (result.second)
            uniqueSum += num;
}
return (2 *uniqueSum) - sum;  
*/



/* delete duplicate 
unordered_set<int> set;


for (int num :nums){
      if (set.count(num) ==1 )
          set.erase(num);  
      else
            set.insert(num);
}

return *set.begin();
*/

}







ListNode* reverseList(ListNode* head);

class MyLinkedList {


private:
      
      Node * tail ;            
    
public:
int size;

      Node * head ;
    MyLinkedList():size(0) {
        head = new Node();
        tail = new Node();  
        head->next = tail;
        tail->prev = head;
    }
    
    int get(int index) {
        
      if (index < 0 || index >= size)
           return -1; 
    
      int mid = size/2;
      Node * curr;
      if (index < mid  ){

            curr = head->next;
            
            for (int i = 0; i < index; i++ ){
                  curr = curr->next;
            }
      } 
      else {
            curr = tail->prev;
            for (int i = 0; i < size  - index-1 ; i++)
                  curr = curr->prev;
      } 

    return curr->val;     
        
    }
struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode* left, TreeNode* right) : val(x), left(left), right(right) {}
};

class BST {
public:
    TreeNode* root;

    BST() : root(nullptr) {}

    bool search(int val) {
        return _search(root, val) != nullptr;
    }

    TreeNode* _search(TreeNode* node, int val) {
        if (node == nullptr || node->val == val)
            return node;

        if (val < node->val)
            return _search(node->left, val);
        else
            return _search(node->right, val);
    }

    void insert(int val) {
        root = _insert(root, val);
    }

    TreeNode* _insert(TreeNode* node, int val) {
        if (node == nullptr)
            return new TreeNode(val);

        if (val < node->val)
            node->left = _insert(node->left, val);
        else
            node->right = _insert(node->right, val);

        return node;
    }

    void deleteNode(int key) {
        root = _delete(root, key);
    }

    TreeNode* _delete(TreeNode* root, int key) {
        if (root == nullptr)
            return root;

        if (key < root->val)
            root->left = _delete(root->left, key);
        else if (key > root->val)
            root->right = _delete(root->right, key);
        else {
            if (root->left == nullptr) {
                TreeNode* temp = root->right;
                delete root;
                return temp;
            } else if (root->right == nullptr) {
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

class LinkedList {
private:
    struct Node {
        int data;
        int key;
        Node* next;
        Node(int data) : data(data), next(nullptr) {}
    };

    Node* head;

public:
    LinkedList() : head(nullptr) {}

    ~LinkedList() {
        while (head) {
            Node* temp = head;
            head = head->next;
            delete temp;
        }
    }

    // Insert data at the front of the list
    void insertFront(int data) {
        Node* newNode = new Node(data);
        newNode->next = head;
        head = newNode;
    }

    // Remove the first occurrence of data from the list
    void remove(int data) {
        if (!head)
            return;

        if (head->data == data) {
            Node* temp = head;
            head = head->next;
            delete temp;
            return;
        }

        Node* prev = head;
        Node* current = head->next;

        while (current) {
            if (current->data == data) {
                prev->next = current->next;
                delete current;
                return;
            }
            prev = current;
            current = current->next;
        }
    }

    // Search for data in the list
    bool search(int data) const {
        Node* current = head;
        while (current) {
            if (current->data == data)
                return true;
            current = current->next;
        }
        return false;
    }

  
  
};

class Pair{

public:
int first;
int second;

Pair():first(0),second(0){}

Pair(int key, int value):first(key),second(value){}

};


class MyHashMap {

private:

#define primeBase  769
array<LinkedList  *,primeBase > arr;

public:

    MyHashMap() {
        
        for (int i =0 ; i< primeBase ; i++)
            arr[i] = new LinkedList;

    }
    
    void put(int key, int value) {
      int mappedVal = key%primeBase;
      
      LinkedList * bucket = arr[mappedVal];
      bucket->insertFront(key);
    }
    
    void remove(int key) {
        int mappedVal = key%primeBase;
        LinkedList * bucket = arr[mappedVal];
        bucket->remove(key);
    }
    
    bool contains(int key) {
      int mappedVal = key%primeBase;
      LinkedList * bucket = arr[mappedVal];
      if( bucket->search(key))
            return true;

        return false;
    }
};



class MyHashSet {


private:

#define primeBase  769
//array<LinkedList*,primeBase > arr;
array<BST*,primeBase > arr;



public:

    MyHashSet() {
        
        for (int i =0 ; i< primeBase ; i++)
            arr[i] = new BST;

    }
    
    void add(int key) {
      int mappedVal = key%primeBase;
      
      BST * bucket = arr[mappedVal];
      bucket->insert(key);
    }
    
    void remove(int key) {
        int mappedVal = key%primeBase;
        BST * bucket = arr[mappedVal];
        bucket->deleteNode(key);
    }
    
    bool contains(int key) {
      int mappedVal = key%primeBase;
      BST * bucket = arr[mappedVal];
      if( bucket->search(key))
            return true;

        return false;
    }
};





 ListNode* rotateRight(ListNode* head, int k) {

      if (!head)
            return nullptr;

      int size =1;
      ListNode * curr =head;

      while(curr->next){
            size++;
            curr= curr->next;
      }
      k = k % size;

      curr->next = head;
      curr = head;
      for (int i = 0; i <  size - k -1; i++, curr= curr->next   );

      head = curr->next;
      curr->next = nullptr;
      
      return head;

 }




unordered_map<Node*,Node* >  visited;


Node * getClonedNode(Node* oldNode,unordered_map<Node*,Node* > * hashMap   ){

      if (oldNode == nullptr)
            return nullptr;

      if (hashMap->find(oldNode) == hashMap->end()) 
            hashMap->insert({oldNode,new Node(oldNode->val)});
             
      return (*hashMap)[oldNode];
} 


 Node* copyRandomList(Node* head) {

      // Weave the two lists so not O(N) space needed 

      if (!head)
            return head;

      Node * curr = head;
      Node * newHead = nullptr;
      
      while(curr){

            Node * newNode = new Node(curr->val);

            newNode->next = curr->next;

            curr->next = newNode;

            curr = newNode->next;
      }

      ///set random
      curr = head;
      while(curr){
            curr->next->random = curr->random ?   curr->random->next : nullptr;

            curr= curr->next->next;
      }

      //unweave
      curr = head;
      newHead = curr->next;
      Node * newCurr = newHead;
      while(curr){

            curr->next = newCurr->next;
            curr= curr->next;

            if(curr){
                  newCurr ->next =curr->next;
                  newCurr = curr->next;
            }
            else
                  newCurr ->next = nullptr;

            

            
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



Node* insertCircularList(Node* head, int insertVal) {

Node * newNode = new Node(insertVal);

if (!head)
{
      newNode->next = newNode;
      return newNode;
}

Node * curr= head->next, * prev = head;

//case 1  3 <i < 5
while (!(prev->val <=insertVal && insertVal <= curr->val)  ){
 
     if (prev->val > curr->val){
      //case 2   9 < 10 < 1
            if (prev->val <= insertVal  && insertVal >= curr->val)
                  break;
      
      //case 2.1 9 < 0 < 1
            if (prev->val >= insertVal  && insertVal <= curr->val)
                  break;
     }
      //case 3 same values closed a loop
      if (head == curr)
            break;

      prev = curr;
      curr = curr->next;
}

prev->next= newNode;
newNode->next = curr; 

return head;



}



/* return the tail of the flatten list */
  Node* flattenDFS(Node* prev, Node* curr) {
      if (!curr)
            return prev;

      prev->next = curr;
      curr->prev = prev;

      Node * temp = curr->next;
      Node * tail = flattenDFS(curr,curr->child);
      curr->child = nullptr;

      return flattenDFS(tail,temp);
  }



Node* flatten(Node* head) {

      if (!head)
            return nullptr;

      Node * sentinal = new Node(0);

      sentinal ->next = head;

      Node * prev =sentinal  ;
      stack<Node *> stack;
      stack.push(head);


      while (!stack.empty()){

            Node * curr = stack.top();
            stack.pop();               
            prev->next = curr;
            curr->prev = prev;

            if (curr->next){
                  stack.push(curr->next);
            }
            if (curr->child){
                  stack.push(curr->child);
                  curr->child = nullptr;
            }
 
            prev = curr;


      }




      /* recursive 
      flattenDFS(sentinal,head);
      */

      Node * tmp = sentinal->next;
      tmp->prev = nullptr;
      delete sentinal;
      return tmp;
}
  


      









ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {

ListNode* head = new ListNode(-1);
ListNode * curr  = head;

int carry =0;
      while(l1 || l2 || carry!=0 ){
            
            int x = (l1 == nullptr)?0:l1->val;
            int y = (l2 == nullptr)?0:l2->val;      

            int sum = (carry + x + y);
            carry = sum/10;

            curr->next = new ListNode(sum%10);

            curr = curr->next;

            l1 = l1?l1->next:nullptr;
            l2 = l2?l2->next:nullptr; 
      }

return head->next;


 }

ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {

//recursive



if (!list1 ) 
      return list2;
if (!list2 ) 
      return list1;

if (list1->val <= list2->val){
      list1->next =    mergeTwoLists(list1->next,list2);
      return list1;
}
else{      
    list2->next = mergeTwoLists(list1,list2->next);
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



 bool isPalindrome(ListNode* head) {

// 2 pointers

ListNode * slow = head, * fast = head;

if(!head)
      return false;


while(fast->next && fast->next->next){

      fast= fast->next->next;
      slow = slow->next;
}



ListNode * headSec =  reverseList(slow->next);

slow->next = nullptr;

ListNode * first = head;
ListNode * sec = headSec;

bool ret = true;

while(first && sec){
      if (first->val != sec->val){
            ret = false;
            break;
      }         
            first = first->next;
            sec = sec->next;
} 

headSec =  reverseList(headSec);
slow->next = headSec;


return ret;


//recursive;

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




ListNode* oddEvenList(ListNode* head) {

      if (!head)
            return nullptr;

      ListNode* odd = head;
      ListNode* even = head->next;
      ListNode* evenHead =even;

      //minimum step
      while (even  && even->next ){

            odd->next = even->next;
            odd = odd->next;
            even ->next = odd->next;
            even = even ->next;
      }

      odd->next = evenHead;
return head;


}



 ListNode* removeElements(ListNode* head, int val) {


      ListNode * sentinel = new ListNode(0);
      sentinel->next = head;
      ListNode * prev, * curr, * toDel = nullptr;
      prev = sentinel;
      curr = sentinel->next;
      
      while (curr){
            if (curr->val == val){
                  prev->next =curr->next;
                  delete curr;
                  curr = prev->next;
            }else{
                  curr = curr->next;
                  prev = prev->next;
            }
      }

ListNode * ret= sentinel->next;

delete sentinel;

return ret;



 }





ListNode* reverseList(ListNode* head) {

//iterative

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

//recursive

      if (!head || !head->next  )
            return head;

      ListNode *  p = reverseList(head->next);
      head->next->next = head;
      head->next = nullptr;

      return p;

}


ListNode* removeNthFromEnd(ListNode* head, int n) {

      //one pass algorithem

      ListNode * ptr1 = head;
      ListNode * ptr2 = head;

      for (int i =0; i < n; i++)
            ptr1 = ptr1->next;

      if (ptr1 == nullptr)
            return head->next;


      while(ptr1->next != nullptr){
            ptr1 = ptr1->next;
            ptr2 = ptr2->next;
      }
      
      ptr2->next=ptr2->next->next;

      return head;


      
      
      //two pass algorithem 
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

      //two pass algorithem V2






}



ListNode * getIntersectionNode(ListNode *headA, ListNode *headB) {


      //two pointers
      ListNode * ptr1 = headA;
      ListNode * ptr2 = headB;

      while (ptr1 != ptr2){
            ptr1 = ptr1 == nullptr ? headB:ptr1->next;
            ptr2 = ptr2 == nullptr ? headA:ptr2->next;
      }

      return ptr1;



      //hash set
      /*
      unordered_set<ListNode* > set;

      ListNode* curr = headA;

      while(curr != nullptr){
            
            set.insert(curr);

            curr = curr->next;
      }

      curr = headB;

      while (curr != nullptr){

            if(set.find(curr)!= set.end())
                  return curr;

            curr = curr->next;
      }       
      return nullptr;
      
          */  

      
      



}



 
 ListNode * detectCycle(ListNode *head) {

      /*
      unordered_set<ListNode *> set;

      ListNode * curr = head;

      while (curr != NULL){

            if (set.find(curr) != set.end())
                  return curr;
            
            set.insert(curr);

            curr = curr->next;
      }

      return NULL;
      */

      //Floyd's Tortoise and Hare

      ListNode * slow =head;
      ListNode * fast =head;

      while  (fast != NULL && fast->next != NULL){
            
            slow = slow->next;
            fast = fast->next->next; 
            if(slow == fast)
                   break;
      }

      if (fast == NULL || fast->next == NULL)
            return NULL;

      slow = head;

      while( slow != fast ){
            slow = slow->next;
            fast= fast->next;
      }

      return slow;




 }






string reverseWords(string s){

//  hello   world  

// in place with indices

reverse(s.begin(),s.end());

int n = s.size();
int idx = 0; // where we copy slower than start due to zeros


for (int start =0 ; start < n; start++){

      if (s[start] != ' '){
            
            // add space ignore begining
            if(idx != 0)
                  s[idx++] = ' ';
            
            //word
            int end = start;
            while(s[end] != ' ' && end < n  )
                  s[idx++] = s[end++];


            reverse(s.begin()+idx - (end - start),s.begin() + idx);      
            
            //end
             start = end;

      }
}

s.erase(s.begin()+idx, s.end());
return s;







//in place using built functions

/*
size_t start = 0;
size_t end =0 ;

start =  s.find_first_not_of(" \t",start);

s.erase(s.begin(),s.begin() + start);
reverse(s.begin(),s.end());
start =0;
start =  s.find_first_not_of(" \t",start);
s.erase(s.begin(),s.begin() + start);

start =0;
end =0 ;

while (end != string::npos){

      start =  s.find_first_not_of(" \t",start);
          if (start == string::npos)
            break;
      end = s.find_first_of(" \t",start);
      if (end == string::npos){

            end = s.length(); 
            reverse(s.begin()+start, s.begin() +end);
            break;
      }
      reverse(s.begin()+start, s.begin() +end);

  

      start = end+1;
}

start =0;
end =0 ;

while (end != string::npos){
      start = s.find_first_of(" \t",start);
        if (start == string::npos)
            break;
      end = s.find_first_not_of(" \t",start);
      start++;
      s.erase(start, end-start  );
}

return s;

*/
// not in place
/*
size_t start = 0;
size_t end =0 ;
string ret;
vector<string> vec; 

while (end != string::npos){

      start =  s.find_first_not_of(" \t",start);

      if (start == string::npos)
            break;

      end = s.find_first_of(" \t",start);

      string word = s.substr(start, end - start );
      start = end+1;

      vec.push_back(word);
}

reverse(vec.begin(),vec.end());

for (string w: vec){

      ret +=( w + ' '); 
}

ret.pop_back();

return ret;

*/


}




int key(int i, int j){
    size_t hash_i = hash<int>{}(i), hash_j = hash<int>{}(j);
    int hashed = (int)(hash_i ^ (hash_i >> 32));
    return (hashed << 5) - 1 + (int)(hash_j ^ (hash_j >> 32));
  }


//(n choose k)
int binomialCoefficient(int n , int k){

int rowCol = key(n,k);
static unordered_map<int,int> cache;

if (cache.count(rowCol) > 0)
      return cache[rowCol];


if (k == 0 || n==0 ||  n==k)
      return cache[rowCol] = 1;

return cache[rowCol] =( binomialCoefficient(n-1,k-1) + binomialCoefficient(n-1,k) ); 

}

vector<int> getRow(int rowIndex){



//pascal  
//rowIndex == 0 return 1    

//writing to same row 

vector<int> ans(rowIndex + 1,1); 

for (int i =1 ; i < rowIndex ; i++  ){

      for (int j = i; j >0; j-- ){
            ans[j ] = ans[j ] +ans[j-1 ] ;
      }
}

return ans;


// save one row instaed of caching everything

/*
vector<int> curr,prev= {1};


for (int i=1; i <= rowIndex; i++){

      curr.assign(i+1,1);

      for (int j =1; j <i; j++ ){

          curr[j] = prev[j-1] + prev[j];  
      }

      prev = move(curr);
}

return prev;
*/

 //recursive
/*
static unordered_map<int,int> cache;
cache.clear();

vector<int> ret(rowIndex);
for (int i =0 ; i <= rowIndex; i++){

      ret.push_back(binomialCoefficient(rowIndex,i));
}

return ret;
*/

/* build the triangle

vector<vector<int>> tri;

for (int i =0 ; i < rowIndex +1; i++  ){

      vector<int> row(i + 1,1); 

      for (int j =1; j <i; j++ ){

          row[j] = tri[i-1][j-1]  +tri[i-1][j];  
      }

      tri.push_back(row);
}

return tri[rowIndex]; 
*/


}

void rotate(vector<int>& nums, int k){

//1 <= nums.length <= 105
//-231 <= nums[i] <= 231 - 1
//0 <= k <= 105



//123 2
int size = nums.size();
if (k == 0 || k%size == 0 )
      return;
k%=size;
//using reverse
reverse(nums.begin(),nums.end());
reverse(nums.begin(),nums.begin() +k);
reverse(nums.begin()+k,nums.end());





//using cyclic replacements
//123 2
/*
int size = nums.size();

if (k == 0 || k%size == 0 )
      return;

int start = 0;
int curr = 0 ;
int num = nums[start];

for (int i =0; i < size ;i ++){

      int step = (curr+k) % size ;
      
      int temp = nums[step];
      nums[step ] = num;
      num = temp;
      if (step == start){

            start++;
            curr = start; 
            num = nums[curr];     
      }
      else
            curr = step;

}      

*/




/*

//extra vector
int size = nums.size();

vector<int> temp(size,0);

if (k == 0 || k%size == 0 )
      return;

for (int i = 0; i < size ; i++){

     temp[(i+k)%size] = nums[i]; 
}

nums.assign(temp.begin(),temp.end());
*/


/*
for (int i =0 ; i < k; i++){

      int last = nums[nums.size() -1];

      nums.pop_back();

      nums.insert(nums.begin(),last);

}

*/



/*
int size = nums.size();
if (k == 0 || k%size == 0 )
      return;

for (int j =0 ; j < k ; j++){

      int num = nums[0];

      for (int i = 0; i < size ; i++   ){

            int temp = nums[((i+1) % size) ];

            nums[(i+1) % size] = num;

            num = temp;
      }

}
*/

}




 int minSubArrayLen(int target, vector<int>& nums) {

 int len = numeric_limits<int>::max(); 

//2 pointer 

int left =0;

int sum = 0;

for ( int i =0; i <nums.size(); i ++) {

      sum += nums[i];

      while (sum >= target ){

            len = min(len,(i - left +1));
            sum -= nums[left++];
      }
}





//using sum array

/*
vector<int> sums(nums.size());

sums[0] = nums[0];


for (int i =1; i < nums.size(); i++)
      sums[i] =sums[i-1] + nums[i]; 


 for ( int i = 0 ; i< nums.size(); i ++){
      for ( int j = i; j < nums.size(); j ++){

            int sum = sums[j] - sums[i] +nums[i] ;

            if (sum >= target){
                  len = min(len,(j-i+1));
                  break;
            }
      }
 }

*/

//Brute force
/*
 for ( int i = 0 ; i< nums.size(); i ++){
      for ( int j = i; j < nums.size(); j ++){

            int sum = 0 ;
            for (int k = i; k <= j; k++)
                  sum += nums[k];      
            
            if (sum >= target){
                  len = min(len,(j-i+1));
                  break;
            }
      }
 }


 */

return len ==  numeric_limits<int>::max() ? 0: len; 
 }



int binarySearch( vector<int>& nums,int target){

      int low =0; 
      int high = nums.size() -1 ;


      while (low <= high){

           int  mid = low + (high -low )/2;

            if (nums[mid] == target)
                  return mid;
            else if (nums[mid] > target)
                        high = mid -1;
                  else
                        low = mid + 1;

      }

      
return -1;

}

void countingSort(std::vector<int>& arr) {
 // Find the maximum element in the array
    //int maxElement = *std::max_element(arr.begin(), arr.end());
//10^4
 //4, 2, 1, 0, 3, 3, 1, 2
      for (int element : arr)
            element += pow(10,4);

    // Create a count array to store the occurrences of each element
      vector<int> count(pow(10,4) +1,0);

    // Count the occurrences of each element
      for (int element : arr)
            count[element]++;

    // Calculate the cumulative sum in the count array
      for (int i =1; i < count.size(); i++)
            count[i] += count[i-1]; 

    // Build the sorted array using the count array
    //Iterate through the original array in reverse order:
    //For arr[7] = 2, its final position is count[2] - 1 = 2, so place 2 at
    vector<int> sortedArr(arr.size());
    for (int i = arr.size() - 1; i >= 0; --i) {
        int element = arr[i];
        int position = count[element] - 1;
        sortedArr[position] = element;
        count[element]--;
    }

      for (int element : sortedArr)
            element -= pow(10,4);

    // Copy the sorted array back to the original array
      arr =sortedArr;
}
 int arrayPairSum(vector<int>& nums) {
 
      sort(nums.begin(),nums.end());

      int maxSum = 0;

      for (int i= 0; i <nums.size() ; i+=2)
            maxSum += nums[i];


     return maxSum; 

 }




bool isCommonPrefix(vector<string>& strs,int len){


      string str1 = strs[0].substr(0,len); 

      for (int i =1; i < strs.size(); i++ ){

            if (strs[i].find(str1) != 0     )    
                  return false;
      }

      return true;
}







void addAtIndex(int index, int val) {
       
      if (index < 0 || index > size )
            return;

      Node * n = new Node(val);

      Node * pred;

      int mid = size/2;
      Node * curr;
      if (index < mid  ){
            pred = head;   
            for (int i = 0; i < index; i++)
                pred = pred->next;   

            n->next = pred ->next;
            n->next->prev =n;
            pred ->next = n;
            n->prev = pred;  

      }else{
           pred = tail;   
           for (int i = 0;  i < size - index ; i++)
                  pred = pred->prev;
            
            n->prev = pred ->prev;
            n->prev->next =n;
            pred ->prev = n;
            n->next = pred; 

      }
      size++;
    }
    
    void addAtHead(int val) {
      addAtIndex(0,val); 
    }
    
    void addAtTail(int val) {
         addAtIndex(size,val); 
    }
    
   
    
      void deleteAtIndex(int index) {
        
            if (index < 0 || index >= size )
              return;

            size--;
        
            int mid = size/2;
            Node * pred;
            Node * i;
            if (index < mid  ){

                  pred = head;        
                  for (int i = 0; i < index; i++)
                        pred = pred->next;
                  
                  i = pred->next;

                  pred->next = pred->next->next;
                  pred->next->prev =pred;
            }else{
                  pred = tail;        
                  for (int i = 0; i < size - index ; i++)
                        pred = pred->prev;
                  
                  i = pred->prev;

                  pred->prev = pred->prev->prev;
                  pred->prev->next =pred;
            }
      
            delete i;
        }
    
};



/**
 * Your Trie object will be instantiated and called as such:
 * Trie* obj = new Trie();
 * obj->insert(word);
 * bool param_2 = obj->search(word);
 * bool param_3 = obj->startsWith(prefix);
 */



class TrieNode {
public:
    unordered_map<char, TrieNode*> children;
    bool isEndOfWord;

    TrieNode() : isEndOfWord(false) {}
};




class Trie {

private:
public:
      TrieNode * root; 

    Trie():root( new TrieNode()) {
            
    }
    
    void insert(string word) {
            
            TrieNode * current = root;
            
            for (char ch: word){
                  if( current->children.find(ch) == current->children.end()){
                  
                        current->children.insert({ch,new TrieNode()});
                  }
                  current =current->children[ch];
            }  
            current->isEndOfWord = true;
    }
    
    bool search(string word) {
        
            TrieNode * current = root;
            
            for (char ch: word){
                  if( current->children.find(ch) == current->children.end())
                        return false;

                  current =current->children[ch];                        
            }
            return current->isEndOfWord;
    }
    
    bool startsWith(string prefix) {
      
       TrieNode * current = root;
            
      for (char ch: prefix){
            if( current->children.find(ch) == current->children.end())
                  return false;

            current =current->children[ch];                        
      }
      return true;


    }
};


string longestCommonPrefix(vector<string>& strs,int l, int r){

      //divide and conquer
      if (l == r)
            return strs[l];
      
      int mid = l + (r-l) / 2;

      string leftStr =  longestCommonPrefix(strs,l,mid);
      string rightStr = longestCommonPrefix(strs,mid +1,r);

      int minLen =  min(leftStr.length(),rightStr.length());

      for (int i=0; i < minLen ; i++){

            if (leftStr[i] != rightStr[i])
                  return leftStr.substr(0,i);
      }

      return leftStr.substr(0,minLen);
}


bool isCommonPrefix(vector<string>& strs,int len){

      string prefix =strs[0].substr(0,len + 1);

      for (int j =1; j < strs.size(); j++){

            if (prefix != strs[j].substr(0,len+1))          
                  return false;
      }

return true;

}
string longestCommonPrefix(vector<string>& strs){

//using binary search.

if(strs.empty()) 
return "";

int minLen = INT_MAX;
for (string str: strs)
      minLen = min((int)str.length(),minLen);

int low = 0;
int high = strs.size() -1;

int mid =0;

while (low <= high) {

      int mid = low + (high -low)/2;

      if (isCommonPrefix(strs,mid))
            low =mid+1;
      else
            high = mid -1;
}

return strs[0].substr(0,mid +1);



string ret;
// flower","flow","flight"


//1 <= strs.length <= 200
//0 <= strs[i].length <= 200



//Using Trie

Trie t;

for (string str : strs)
      t.insert(str);

TrieNode * current = t.root;

while (current->children.size() == 1 ){
      ret += current->children.begin()->first;
      current = current->children.begin()->second;
}

return ret;

/*
//Binary search
size_t minLen = numeric_limits<size_t>::max() ;

for (int i =0; i < strs.size(); i++)
      minLen = min(minLen,strs[i].length());
      

int low = 01;
int high = minLen;

while (low <= high){
      int mid = (low + high) / 2;
      
      if (isCommonPrefix(strs,mid))
            low = mid + 1 ;
      else
            high = mid - 1;
}

ret = strs[0].substr(0,(low+high)/2);


//Recursion Divide n Conquer 

/// ret = longestCommonPrefix(strs,0,strs.size()-1);


*/






//Horizantal scanning compare 1 string to all and remove characters
/*
string lcp = strs[0];

if (lcp.empty())
      return lcp;

// flower","flow","flight"
for (int i = 1; i < strs.size(); i++){

      string nStr = strs[i];


      while ( nStr.find(lcp) != 0   ){
            lcp.erase(lcp.end() - 1);

            if (lcp.empty())
                  return lcp;
      }
}

return lcp;
*/

//Vertical scanning  - compare charters instead of ull strings 
/*
string ret = "";

for (int i =0; i <strs[0].length(); i++){

      for (int j =1; j < strs.size(); j++  ){

            if (strs[0][i] != strs[j][i] || i == strs[j].length())
                  return strs[0].substr(0, i);


      } 
}

return strs[0];
*/

/*. Brute force
int minLen = 201;

string ret = "";

for (int i = 0; i < strs.size(); i ++){

      if (strs[i].length() < minLen)
            minLen = strs[i].length();
}

if (minLen == 201 || minLen == 0   )
      return ret;

for (int j = 0; j<minLen ; j++ ){

      char ch = strs[0][j];  
      int i = 0;
      for (   ;i < strs.size(); i++ ){
            
            if (ch != strs[i][j])
                  break; 
            
      }
      
      if (i != strs.size() )
            break;
    
      ret.push_back(ch);



}

return ret;
*/

      }




int strStr(string haystack, string needle) {

int hL = haystack.length();
int nL = needle.length();

int window = 0;
for (; window < hL - nL+ 1; window++   ){

      int j = 0;

      while (haystack[ window +j] == needle[j] && j < nL     ){
            j++;
      }

      if (j == nL)
            break;
      j = 0;

}
 if (window == hL - nL +1 || hL < nL )
      window = -1;

return window;

}

string addBinary(string a, string b){


//add bit bit and push to string 
int maxL = max(a.length(),b.length());
// padding zeros

a = string(maxL - a.length() ,'0') + a;
b = string(maxL - b.length() ,'0') + b;
string ret;

int carry = 0;

for ( int i = 0;  i < maxL; i++){

      int A = static_cast<int>(a[maxL -i -1] - '0');
      int B = static_cast<int>(b[maxL -i -1]- '0');

      int result = A+B + carry;

      if (result == 3 ){
            carry = 1;
            ret.push_back('1');
      }else if (result == 2){
            ret.push_back('0');
            carry =1;
      }else if (result == 1){
            ret.push_back('1');
            carry =0;
      }else if (result ==0){
            ret.push_back('0');
            carry =0;
      }
}

if (carry)
      ret.push_back('1');
reverse(ret.begin(),ret.end());
return ret;

//1111
//0010
/*
int maxL = max(a.length(),b.length());

a = string(maxL - a.length() ,'0') + a;
b = string(maxL - b.length() ,'0') + b;

int A = 0; 
int B = 0; 

for ( int i = 0;  i < maxL; i++){
      
      A = (A << 1) + a[i] - '0';
      B = (B << 1) + b[i] - '0'; 
}

int carry = A & B;

A^=B;

while (carry != 0 ){

      B = carry << 1;
      carry = A & B;
      A^=B;
}


      string ret; 
while (A) {

      ret = static_cast<char>(A%2 + '0') + ret;

      A/=2;
}

string zero("0");
return ret.empty()? zero : ret;

*/

/*
bitset<32> binaryNumber1(stoi(a));
bitset<32> binaryNumber2(stoi(b));

bitset<32> sum = binaryNumber1. binaryNumber2;
return sum.to_string(); 

*/
/*
int maxLength = max(a.length(),b.length());


string paddedA(maxLength-a.length(),'0');
string paddedB(maxLength-b.length(),'0');

a = paddedA + a;
b = paddedB + b;

string ret;

int carry = 0;

for (int i = maxLength-1 ; i >= 0; i--){
      
      
      int bitA = a[i] - '0';
      int bitB = b[i] - '0';

      int sum = carry + bitA + bitB;

      ret.push_back(sum%2 + '0');
      carry =sum/2;   

}

if (carry)
      ret.push_back('1'); 

reverse(ret.begin(),ret.end());
return ret;
*/



 }

vector<vector<int>> generate(int numRows) {

vector<vector<int>> triangle;

//Pascal
for (int i = 0; i < numRows; i++){
vector<int> row(i+1,1);


      for (int j =1; j < i; j++ ){
            row[j] = triangle[i-1][j-1] +  triangle[i-1][j] ;
      }

triangle.push_back(row);

}

return triangle;

}

vector<int> spiralOrder(vector<vector<int>>& mat){
//1,2,3,6,9,8,7,4,5


const int VISITED = 101;

vector<int> ret; 

int rows = mat.size();
int cols = mat[0].size();

int directions[4][2] = {{0,1},{1,0},{0,-1},{-1,0}};
int currDirection = 0;

int i = 0 , j = 0;

ret.push_back(mat[0][0]);
mat[0][0] = VISITED;


for (int k = 1; k < rows * cols; ){

      while (i + directions[currDirection][0] >= 0 && i + directions[currDirection][0] < rows &&
       j + directions[currDirection][1] >= 0 && j + directions[currDirection][1] < cols &&
       mat[i + directions[currDirection][0]][j + directions[currDirection][1]] != VISITED ) {      
                  
                  k++;
                  i += directions[currDirection][0];
                  j += directions[currDirection][1];
                  ret.push_back(mat[i][j]);
                  mat[i][j] = VISITED;
      }

      currDirection = (currDirection +1)%4;

}

return ret;

}

vector<int> findDiagonalOrder(vector<vector<int>>& mat) {
      

vector <int> ret;
//1,2,4,7,5,3,6,8,9]
int rows = mat.size();
int cols = mat[0].size();

for (int k = 0; k < rows + cols -1 ; k++){

      int i = k < cols ? 0 : k - cols +1;   
      int j = k < cols ? k : cols -1;

      vector<int> reverse;

      while (i < rows && j > -1 ){
            if(k%2 ==0)
                  reverse.insert(reverse.begin(),mat[i][j]);
            else
                  ret.push_back(mat[i][j]);
            j--;
            i++;
      }  

      if(k%2 ==0){

            for (int a:  reverse)
                  ret.push_back(a);      
            reverse.clear();
      }          
}
      return ret;
}


 vector<int> plusOne(vector<int>& digits) {
 
int size = digits.size();
int i = size - 1;


for ( ;i >= 0; i-- ){


      if (digits[i]== 9)
            digits[i]=0;
      else{
            digits[i]++;
            break;
      }     
}

if (digits[0] == 0)
      digits.insert(digits.begin(),1);

return digits;

 
 
 
 //while(digits.rbegin() != digits.rend() )


 }

int dominantIndex(vector<int>& nums) {

//[3,6,1,0]


int index = -1;

int max = 0;

int next = 0;

for (int i =0; i < nums.size(); i++){

      int num = nums[i];
      
      if (num > max){
            next = max;
            max = num;
            index = i;
      }else if(num > next)
            next = num;
            
}

if (max >= 2 * next)     
      return index;
else 
      return -1;

}


int pivotIndex(vector<int>& nums) {

int size = nums.size();
int pivot = -1;

int sum = 0, leftSum = 0;

for (int i  : nums )
      sum += i;


for (int i = 0  ; i < size ; i++){

      if (leftSum == (sum - leftSum - nums[i]    )){
            pivot = i;
            break;
      }

      leftSum += nums[i];

}


/*

vector<int> leftSum(size,nums[0]);
vector<int> rightSum(size,nums[size-1]);



for (int i = 1; i < size ; i++){

      leftSum[i] = nums[i] + leftSum[i-1]; 
      
      rightSum[size -i -1] = rightSum[size - i] + nums[size - i -1]  ;    
}


for (int i =  0;i < size-1; i++ ){

      if (leftSum[i] == rightSum[i])
      {
            pivot = i; 
            break;
      }

}

if (pivot == -1)
      if (size > 1){

            if (leftSum[size-2] == 0)
                  pivot = size -1;
            else if ((rightSum[1] == 0))
                  pivot = 1;
      }else
            pivot = 0; 


*/

return pivot;

}


vector<int> findDisappearedNumbers(vector<int>& nums) {

//4,3,2,7,8,2,3,1

set<int> mySet;


vector<int> ret;

int size = nums.size();

for (int i = 0; i < size; i++){

      mySet.insert(nums[i]);
} 

for (int i = 1; i <= size; i++){

      if (mySet.find(i) == mySet.end())
            ret.push_back(i);
}

return ret;

}



int thirdMax(vector<int>& nums) {

//1,2,2,5,3,5
int max =0;
set<int> mySet;
long  l ;
pair<int,int> a;


l = numeric_limits< int>::min();

for (int i =0; i < nums.size(); i++){

      int num = nums[i];

      if(mySet.size() <3 ){

            mySet.insert(num);
            continue;
      }
      else{
            if (num > *mySet.begin()  ){
                  if (mySet.find(num) == mySet.end()){

                       mySet.erase(mySet.begin());
                       mySet.insert(num);

                  }
                  
            }
      }
}
 
if (mySet.size()==3)    
      max = * mySet.begin();
else
      max = *mySet.rbegin();


/*
priority_queue<int, vector<int>, greater<int>> minHeap;
unordered_set<int> mySet;
//3,2,1

      int max = 0 ;

      for (int i = 0; i <nums.size(); i++){

            int num = nums[i];

            if(mySet.find(num) != mySet.end())
                  continue;

            if(minHeap.size() != 3){
                  minHeap.push(num);
                  mySet.insert(num);
            }
            else{
                  if (minHeap.top() < num){
                        minHeap.pop();
                        minHeap.push(num);
                        mySet.insert(num);
                  }                  
            }

      }

      if(minHeap.size() != 3){
            while(!minHeap.empty() ){
                  max = minHeap.top();
                  minHeap.pop();
            }
      }
      else 
            max = minHeap.top();
*/

return max;




}

 int findMaxConsecutiveOnes(vector<int>& nums) {

//1,0,1,1,0,1,1,1,0,0,0

//1,0,1,1,0

      bool bFlag = false; 
      int max = 0; 
      int count = 0;

      int minCount = 0;

      for (int i =0 ; i < nums.size(); i++ ){

            
            if (nums[i] == 1){
                  count++;
                  continue;
            }  

            if (nums[i] == 0 ){

                  if (bFlag == false ){
                        bFlag = true; 
                        count++;
                        minCount = count;
                        continue;
                  }
                  else{
                        bFlag = false;
                        i--;
                        if (count> max)
                              max = count;
                        count = count - minCount;                        
                  }
            }

      }

       if (count> max)
            max = count;

/*
      int max =0;

        for (int i = 0; i< nums.size(); i++){
    
            int iZeros = 0;

            for (int j = i; j< nums.size(); j++){
                
                if(nums[j] == 0)
                    iZeros++;
                
                if (iZeros <= 1){
                    max = std::max(max,j - i + 1 );
                }



            }
        }
      
*/
return max;
 }


 int heightChecker(vector<int>& heights) {
        //1,1,4,2,1,3

      vector<int> expected(heights.begin(),heights.end());

      sort(expected.begin(),expected.end());

      int num=0;

      for (int i =0; i< heights.size(); i++  ){

            if (heights[i] != expected[i])
                  num++;
      } 

      return num;
        
    }


 void moveZeroes(vector<int>& nums) {
 //0,1,0,3,12
 //1,3,12,0,0
 
 
int writePtr = 0;

      for (int readPtr= 0; readPtr < nums.size(); readPtr++ ){
            
            if (nums[readPtr] != 0 ){
                  nums[writePtr] = nums[readPtr];
                  writePtr++;
            }
      }


      for (; writePtr < nums.size(); writePtr++ )
            nums[writePtr] = 0;

 }


vector<int> replaceElements(vector<int>& arr) {
//17,18,5,4,6,1
//18,6,6,6,1,-1

      int size = arr.size();
      int max = arr[size-1] ;
      arr[size-1] = -1;


      for (int i = size - 2 ;  i >=0; i--)
      {     
            int temp;

            temp = arr[i];


            if (arr[i] > max){
                  max += arr[i]; 
                  arr[i] = max - arr[i];
                  max -= arr[i]; 
            
            } else if (arr[i] < max)
                  arr[i] = max;
                  
      }
      return arr;
}


bool validMountainArray(vector<int>& arr) {

      int size =arr.size();

      if (size < 3)
            return false;

      //3,1,1
      
      int i = 1;
      
      while (arr[i-1] < arr[i])
      {
            i++;
      } 

      if (i == 1 || i == size)    
            return false;

       while (arr[i-1] > arr[i] && i < size)
      {
            i++;
      }


      return size  == i ;

      }

bool checkIfExist(vector<int>& arr) {

      int size = arr.size();
      map<int,int> myMap;

//[7,1,14,11]
      for (int i= 0; i < size; i++){

            int num = arr[i];

            int iDouble = num * 2;
            

            if (myMap.find(num) == myMap.end())
                  myMap[num] = i;

            if (myMap.find(iDouble) != myMap.end() && myMap[iDouble] !=i )
                  return true;


            if( ( num%2 ==0 &&  myMap.find(num/2) != myMap.end() )  )
            {

                  if (myMap[num/2] !=i)
                   return true;

            } 
      }

      return false;
      
        
    }

       int removeDuplicates(vector<int>& nums) {

            //{0,0,1,1,1,2,2,3,3,4};
            int size =  nums.size();
            int duplicates = 0;
            int insertIndex = 1;
            for (int i=1 ; i < size; i++ ){

                  if (nums[i] != nums[i-1] ){
                      nums[insertIndex] = nums[i];
                      insertIndex++  ;      
                  }     
                  
               
            }
            return insertIndex ;
       }


    int removeElement(vector<int>& nums, int val) {
      int size = nums.size();
        
      int reversePtr = 0;
      for (int i = 0 ; i <  size -reversePtr ; i ++){
            

            if (nums[i] != val  ){
                  continue;
            }else if (nums[size - 1 - reversePtr] == val){ // both equal 
                  --i;
                  ++reversePtr;
                  
      
            }
            else{
                  nums[i] = nums[size -1 - reversePtr];
                  ++reversePtr;
                  
            }
                  

      }
      //[0,1,2,2,3,0,4,2] 2

      //3,2,2,3 3 
       
      nums.resize(size - reversePtr );
      return  size - reversePtr;

      
    }

 void duplicateZeros(vector<int>& arr) {
        
      int size = arr.size();
      int zeros = 0;

        for (int i = 0; i < size; i++){
            if (arr[i] == 0)
                  zeros++;
            
        }
      
      vector<int> temp(size+ zeros);
//{1,0,2,3,0,4,5,0};
      int i=1;
      for (;   zeros >0 ; i++ ){

           if (arr[size-i] == 0 ){
            temp[size+zeros-i]   = 0;
            temp[size+zeros-i-1] = 0;
            zeros--;
           }  
           else     
            temp[size+zeros-i]= arr[size-i];

      }

      for (int j = size - i+1 ; j < size; j++ )
            arr[j] = temp[j];
        

    }


void mergeSort(vector<int>& nums1, int m, vector<int>& nums2, int n){

      for (int i =0; i < n; i++){
            nums1[m+i] = nums2[i];
      }

      sort(nums1.begin(),nums1.end());


}      

void merge(vector<int>& nums1, int m, vector<int>& nums2, int n){
      
    /*  int r =  m + n;
      int i = m-1;
      int j = n-1; 

      for ( r = r-1 ; r > 0, j >=0 ; r--    ){
            
            if ( nums2[j] >= nums1[i] ){
                  nums1[r] = nums2[j];
                  j--;
            }
            else{
                  nums1[r] = nums1[i];
                  nums1[i] = 0;
                  i--;
            }
      }
      */
}



 int maxProfit(vector<int>& prices) {

      int size = prices.size();
      int min = pow(10,4);
      //int max = 

      int profit = 0;

      for (int i = 0; i < size; i ++ ){
      
            if (prices[i] < min)
                  min = prices[i];
            else if ( (prices[i]  - min > profit) ){
                  profit = prices[i]  - min;

            }

            

      }

      return profit;

}



vector<int> twoSumMap(vector<int>& nums, int target){

      map<int,int> myMap;

      int size = nums.size();

      vector<int> indices(2);
      //&&  myMap[compliment] != i 
      for (int i = 0 ; i < size; i++  ){
            
            int compliment = target - nums[i];
            
            
            if (myMap.find(compliment)  != myMap.end() && myMap[compliment] != i   ){

                  indices[0] = myMap[compliment];
                  indices[1] = i;
                  break;
            }

            myMap[nums[i]] = i;
      }

      return indices;

}


vector<int> twoSum(vector<int>& nums, int target){


//numbers[low] > (1 << 31) - 1 - numbers[high]


//2147483647
/*
int a = (2<<30) -  1;
int p = (1<<31) -  1;
long b = pow(2,32) /2;

unsigned int       c = pow(2,32) ;


int d = numeric_limits<int>::max();
long e = numeric_limits<long>::min();
long long f = numeric_limits<long long>::min();
double g = numeric_limits<double>::min();
float h = numeric_limits<float>::min();
size_t i = numeric_limits<size_t>::min();
*/
//long d = ((1<<32) -1)>>1 ;

//2,7,11,15
// two pointer solution for sorted array

      int size = nums.size();

      vector<int> indices(2);

      for (int i =0, j = size -1; i < j; ){
           
           //check overflow 
            if (nums[i]  > (1L<<31) -1  - nums[j] ){
                  j--;
                  continue;
            }
           
           
            int sum =nums[i] + nums[j];

            if ( sum == target){
                  indices[0]=i;
                  indices[1]=j;

            }else if (sum < target)
                  i++;
            else 
                  j--;
      }



/*
      int size = nums.size();

      vector<int> indices(2);

      for (int i = 0 ; i < size-1; i++  ){
            for (int j = i+1; j < size; j++){

                  if (nums[i] + nums[j] == target){
                        indices[0] = i;
                        indices[1] = j;
                        return indices;
                  }
                        

            } 

      }

*/

      return indices;

}

 bool isLongPressedName(string name, string typed) {

      bool bIs= true;
      
      int i=0;
      for (int  j=0;  j < typed.length(); ){
            if ((i < name.length()) && name[i] == typed[j] ){
                  i++;
                  j++;

            }else if (j != 0 &&  typed[j] == typed[j-1])
                  j++;
            else
                  return false;
      }

      return i == name.length();

 }


 int partitionDisjointV3(vector<int>& nums) {
 
      int size = nums.size();

      int currentMax;
      int possibleMax;
      int length = 1;
      currentMax = possibleMax = nums[0];
      

      
      for (int i= 1; i < size ; i++ ){
      
            if (nums[i] < currentMax ){
                  length = i+1;
                  currentMax = possibleMax;
            }
            else{
                 possibleMax =   max(possibleMax,nums[i]);
            }
      
      }
            return length;

     
 }




 int partitionDisjointV2(vector<int>& nums) {
 
      int size = nums.size();

      int currentMax;
      int minRight[size];

      currentMax = nums[0];
      minRight[size-1] = nums[size-1];

      for (int i= size-2; i >= 0 ; i-- ){
            minRight[i] = min(nums[i],minRight[i+1]);
      }
  
      for (int i=1; i < size; i++ ){
         
            if (currentMax <= minRight[i])
                   return i;

            currentMax = max(nums[i],currentMax);
      }
      return -1;
 }





 int partitionDisjoint(vector<int>& nums) {
 
      int size = nums.size();

      int maxLeft[size];
      int minRight[size];

      maxLeft[0] = nums[0];
      minRight[size-1] = nums[size-1];

      for (int i=1; i < size; i++ ){
            maxLeft[i] = max(nums[i],maxLeft[i-1]);
      }

      for (int i= size-2; i >= 0 ; i-- ){
            minRight[i] = min(nums[i],minRight[i+1]);
      }
  
      for (int i=1; i < size; i++ )
            if (maxLeft[i-1] <= minRight[i])
            return i;
      
      return -1;
 }





bool compareParity(int a, int b){
      return (a%2 == 0) && (b%2 ==1);
}

vector<int> sortArrayByParityV2(vector<int>& nums) {

     // sort(nums.begin(),nums.end(),compareParity);

      return nums;


 }

 vector<int> sortArrayByParity(vector<int>& nums) {
 
      int size = nums.size();

      for (int i = 0 , j = size -1; i<j;  ){

            if ( (nums[i]%2 == 1 ) && (nums[j]%2 == 0)){

                  int temp =  nums[i];
                  nums[i] = nums[j];
                  nums[j] = temp; 
            }
            if (nums[i]%2 == 0)
                        ++i;
            if (nums[j]%2 == 1)
                        --j;

      }
      return nums;
 
 }


 vector<int> sortedSquaresV2(vector<int>& nums){

      int size = nums.size();

      vector<int> res(size,0);
      int left = 0;
      int right = size -1;

      for (int i = size -1 ; i >= 0  ; --i){

            int leftSq = pow(nums[left],2); 
            int rightSq = pow(nums[right],2); 

            if (leftSq > rightSq){
                  res[i] = leftSq;
                  left++;
            }
            else{
                  res[i] = rightSq;
                  right--;
            }

      }
      return res;
 }




 vector<int> sortedSquares(vector<int>& nums){
	
     vector<int> res(nums.size(),0); 
     
     vector<int>::iterator it = nums.begin();
     vector<int>::reverse_iterator rit = nums.rbegin();

      if(it == nums.end() || rit == nums.rend() || nums.empty()  ){
            cout<< "empty vector"; 
            return nums;
      }
     
     //-15,-10,0,1, 2, 3, 4, 5
      for (auto currIt = res.rbegin() ;it <= rit.base() && currIt != res.rend() ;currIt++ ){
            
            int leftSq = (*it)*(*it); 
            int rightSq = (*rit)*(*rit);

            if (leftSq > rightSq){
                  *currIt = leftSq;
                  it++;
            }
            else{
                  *currIt = rightSq;
                  rit++;
            }
            
      }
     

      return res;
}  


        
	


/*****************************************************************************************/

//HH:MM:SS
string timeFormat(int seconds){

      using namespace std;
      int hours = seconds/3600;
      int mins  =  (seconds % 3600)/60;
      int secs  = seconds % 60;

      return ((hours <10) ? "0":"") + to_string(hours) + ":" + ((mins < 10) ? "0":"") + to_string(mins) + ":"     
       + ((secs <10) ? "0":"") + to_string(secs);
      std::ostringstream oss;
      oss << std::setw(2) << std::setfill('0') << hours << ":"
        << std::setw(2) << std::setfill('0') << mins << ":"
        << std::setw(2) << std::setfill('0') << secs;

    return oss.str();
}

struct Chain{

      vector<size_t> jobIds;
      size_t totalRuntime;
};

struct JobInfo{

      size_t runtimeSeconds;
      size_t nextJobId; 
};
/*
int main(){   

      using namespace std;
    
      
      //string inputStr = "#job_id,runtime_in_seconds,next_job_id\n1,10,2\n2,20,3\n3,30,4\n4,40,0\n5,15,6\n6,25,0";
      //string inputStr = "#job_id,runtime_in_seconds,next_job_id\n1,60,23\n2,23,3\n3,12,0\n23,30,0";
      //string inputStr = "#job_id,runtime_in_seconds,next_job_id\n1,100,0"; 
      //string inputStr = "#job_id,runtime_in_seconds,next_job_id\n1,60,2\n2,30,0\n3,45,4\n4,20,0\n5,10,0"; 
      //string inputStr = "#job_id,runtime_in_seconds,next_job_id\n1,10,2\n2,20,3\n3,30,4\n4,40,0"; 
      // Expected Output: Chain 1: [1 -> 2 -> 3 -> 4] Total Runtime: 100 seconds
      
      //edge Cases
      //string inputStr = "#job_id,runtime_in_seconds,next_job_id"; 
      //string inputStr = "#job_id,runtime_in_seconds,next_job_id\n1,0,0"; 
      // Expected Output: Chain 1: [1] Total Runtime: 0 seconds
      //string inputStr = "#job_id,runtime_in_seconds,next_job_id\n1,60,99"; 
      // Expected Output: Error: Job 99 referenced by next_job_id is missing from the input.
      string inputStr = "#job_id,runtime_in_seconds,next_job_id\n1,60,2\n1,30,0\n2,20,0"; 
      // Expected Output: Error: Duplicate job_id 1 found in input.
      //string inputStr = "1,60,23\n2,30,0"; 
      // Expected Output: Error: Missing header row.
      //string inputStr = "#job_id,runtime_in_seconds,next_job_id\n1,abc,23\n2,30,0"; 
      // Expected Output: Error: Malformed input on line 2. Non-integer value found for runtime_in_seconds.
      //string inputStr = "#job_id,runtime_in_seconds,next_job_id\n1,-10,2\n2,30,0"; 
      // Expected Output: Error: Negative values are not allowed for runtime_in_seconds or job_id.
      //string inputStr = "#job_id,runtime_in_seconds,next_job_id\n1,60\n2,30,0,extra"; 
      // Expected Output: Error: Malformed input. Incorrect number of columns on line 2.
      //string inputStr = "#job_id,runtime_in_seconds,next_job_id\n1,10,2\n2,20,1"; 
      // Expected Output: Error: Circular chain detected involving job_id 1.
      //string inputStr = "#job_id,runtime_in_seconds,next_job_id\n1,1,2\n2,1,3\n3,1,4\n4,1,5\n5,1,6\n6,1,7\n7,1,8\n8,1,9\n9,1,10\n...9998,1,9999\n9999,1,0"; 
      // Expected Output: Chain 1: [1 -> 2 -> 3 -> ... -> 9999] Total Runtime: 9999 seconds
      //string inputStr = "#job_id,runtime_in_seconds,next_job_id\n1,2,10\n10,15,20\n20,25,30\n30,35,10";
      // Expected Output: Error: Circular chain detected involving job_id 10, 20, and 30.


    #job_id,runtime_in_seconds,next_job_id
      1,10,2
      2,20,3
      3,30,4
      4,40,0
      5,15,6
      6,25,0
      */
    // Create a stringstream to simulate getline reading from a string
    //stringstream inputStream(inputStr);
    
    //getline(inputStream, inputLine);
     
     
     /*
      string inputLine;
      getline(cin, inputLine);

      if ( inputLine != "#job_id,runtime_in_seconds,next_job_id" )
          throw std::logic_error("Missing Header");            

      unordered_map<int,JobInfo> logContainer;  
      vector<int> startJobs;
      unordered_set<int> nextJobSet;

      while(getline(cin, inputLine)){

            stringstream ss(inputLine);
            int jobId;
            int runtimeSeconds;
            int nextJobId; 
            char delimiter1;
            char delimiter2;

            if (! (ss >> jobId >> delimiter1 >> runtimeSeconds >> delimiter2 >> nextJobId) )
                  throw std::logic_error("Log Input Malformed: " + inputLine );
            if (jobId <= 0 || runtimeSeconds < 0 || nextJobId < 0 || delimiter1 != ',' || delimiter2 != ',' ) 
                         throw std::logic_error("Log Input Malformed: " + inputLine );

            JobInfo jobInfo =  {static_cast<size_t>(runtimeSeconds),static_cast<size_t>(nextJobId)};
            auto result1 = logContainer.insert({jobId,move(jobInfo)}) ;
           
            if (!result1.second)
			      throw std::logic_error("Duplicate job ID detected: " + to_string(jobId) );

            if (nextJobId > 0 ){
                  auto result2  =  nextJobSet.insert(nextJobId);
                  if (!result2.second)
			      throw std::logic_error("Duplicate next job ID detected: " + to_string(nextJobId));
            }
      }

      if (logContainer.empty())
            throw std::logic_error("No log entry found.");
            

      // Find all starting jobs (jobs that aren't "next" jobs for others)
      for (const auto & [jobId,jobInfo] : logContainer){
            if (jobInfo.nextJobId !=0 && logContainer.find(jobInfo.nextJobId) == logContainer.end() )
                  throw std::logic_error("Job ID: " + to_string(jobId) 
                  +  " referencing non existing next Job ID: " + to_string(jobInfo.nextJobId)  );
      
            if ( nextJobSet.find(jobId) == nextJobSet.end() )
                  startJobs.push_back(jobId);
      }

      if (startJobs.empty())
             throw std::logic_error("No independet starting Jobs.") ;
            
      vector<Chain> chains;
      chains.reserve(startJobs.size());
      
       for (const int startJob : startJobs) {
            
            Chain chainStruct;
            chainStruct.totalRuntime = 0;             
            size_t currentJob = startJob;
            
            while(currentJob != 0 ){
            
                  const auto & jobInfo =  logContainer[currentJob];      

                  chainStruct.jobIds.push_back(currentJob);
                  currentJob = jobInfo.nextJobId;                  
                  chainStruct.totalRuntime += jobInfo.runtimeSeconds;                 
            }
            chains.push_back(move(chainStruct));
      }

      sort(chains.begin(),chains.end(),[](const  Chain  & a, const  Chain & b ){
            return a.totalRuntime > b.totalRuntime ;
             }
      );
      
      //vector<pair<vector<int>,int>>
      for (const auto & chain:chains ){
            cout << "-" << endl;
            cout << "start_job: "<< chain.jobIds.front() << endl;
            cout << "last_job: " << chain.jobIds.back() << endl;
            cout << "number_of_jobs: " << chain.jobIds.size() << endl;
            string a = timeFormat(chain.totalRuntime) ;
            cout << "job_chain_runtime: " << timeFormat(chain.totalRuntime) << endl;
            int averageTime = chain.jobIds.size() ? chain.totalRuntime / chain.jobIds.size() : 0;
            string b = timeFormat(averageTime);
            cout << "avarage_job_time: " << timeFormat(averageTime) << endl;
            string res = "start_job: " + to_string(chain.jobIds.front()) + 
             " last_job: " + to_string(chain.jobIds.back()) + 
             " number_of_jobs: " + to_string(chain.jobIds.size()) + 
             " job_chain_runtime: " + a + 
             " average_job_time: " + b;


            int c =6;
      }

      return 0;
      
}
*/

struct File{

public:
     File(const string & path,const string & opaqueID):path(path),opaqueID(opaqueID){}            
     
      string path;
      string opaqueID;
};

struct Commit{

      uint64_t id;
      uint64_t timeStamp;
      vector<File> files;
};

//class to unite commits to repo
class UnionFindDR{

private:
      //not using vector since we don't know the range of id's
      unordered_map<int, int> parent;
      unordered_map<int, int> rank;
public:

      bool addCommit(int x){
           auto [it, success] = parent.insert({x, x});
            if (success) {
                rank[x] = 1; // Initialize rank for the new commit
      }
            return success;
      }


      int find (int x){
            if (parent.find(x) == parent.end())
                  parent[x] = x;
            
            if (parent[x] != x)
                  parent[x] =  find(x);  // path compression optimization
            
            return parent[x];
      }
      
      //union by rank optimization
      bool unionSets(int x , int y){

            int rootX = parent[x];
            int rootY = parent[y];

            if ( rootX != rootY ){

                  size_t rankX = rank[rootX];
                  size_t rankY = rank[rootY]; 

                  if (rankX == rankY){
                        parent[rootY] =rootX;
                        rank[rankX]++;
                  }else 
                        if (rankX > rankY)
                              parent[rootY] =rootX;
                        else 
                              parent[rootX] =rootY;
                  return true;
            }
            return false;
      }
};

//Disaster Recovery
int main(){
      try{
      //every log entry is a commit 
      //every file is described by path and opaque id 
      //Two commits for different repositories may contain the same file path, but the file's opaque identifier should be distinct, and vice-versa 
      //matching path and opaque will go to the same repo 
      //ambigious - when 2 commits are united by 1 file but the other not fully matched. 
      // files are uniting 2 commits to the same repo . 

      size_t N;
      cin >> N;
      cin.ignore(); //ignore new line
      
      vector<Commit> commits;
      commits.reserve(N);

       // 2 level map to represent file name and opauqeId. file name -> opaque -> commit ID
       //commit id in union id connects to root which is the actuall Repo ID.
      unordered_map<string, unordered_map<string, int>> fileToRepo; 

      // union find to connect join commits to reposotories 
      UnionFindDR unionFind; 

      // contain the root(repisitory id) and all related commits
      unordered_map<int, vector<Commit>> repositories;

      for (int i = 0; i < N; i++){

            string inputLine;
            getline(cin, inputLine);

            istringstream ss(inputLine);
            Commit commit;
            //id 38024 timestamp 74820 foo.py ac819f bar.py 0d82b9
            
            //add validation 
            string param1 , param2;
            if (! (ss >> param1 >> commit.id >> param2 >> commit.timeStamp ) )
                   continue;  //discared malformed entries 
            
            if (unionFind.addCommit(commit.id))
                   continue;  //discared malformed entries. commit id need to be unique.

            while (ss >> param1 >> param2) 
                  commit.files.emplace_back(param1, param2); // emplace to avoid expensive copying
    
            //loop all files in current entry 
           for (const auto &file : commit.files) {
                  const string & filePath = file.path;
                  const string & opaqueId = file.opaqueID;

                  // search repo for file add it if doesn't exist 
                  //if it does exist add it to repo 
                  if (fileToRepo[filePath].count(opaqueId) == 0) 
                        fileToRepo[filePath][opaqueId] = commit.id; // the root in union find with be the Repo id.
                  else{ //find to where to add the commit in the unionFind set
                        int existingCommit = fileToRepo[filePath][opaqueId];
                        unionFind.unionSets(commit.id, existingCommit);
                  }
                // checking ambiguity for every file added
                // loop on all memebers of the 2nd level map that matches file path. covers the vicecersa requirment. 
                for (const auto &otherOpaque : fileToRepo[filePath]) {
                        if (otherOpaque.first != opaqueId) {
                              throw logic_error("AMBIGIOUS INPUT!");
                }
            }
           }
            commits.push_back(std::move(commit)); //move to avoid expensive copy of commit
      }     

      // building the newly organised repository  
      for (auto &commit : commits) {
            int repoId = unionFind.find(commit.id);
            repositories[repoId].push_back(std::move(commit));
      }

      //order each repository by increasing of timestamp,id.
      for (auto &repo : repositories) {
        sort(repo.second.begin(), repo.second.end(), [](const Commit &a, const Commit &b) {
            return a.timeStamp < b.timeStamp || (a.timeStamp == b.timeStamp && a.id < b.id);
      });
    }
      
    int R;
    cin >> R;
    cin.ignore();

     for (int i = 0; i < R; ++i) {
        uint64_t startTime, endTime;
        string filePath, opaqueId;
        cin >> startTime >> endTime >> filePath >> opaqueId;

        vector<int> result;
        if (fileToRepo[filePath].count(opaqueId) > 0) {
            int repoId = unionFind.find(fileToRepo[filePath][opaqueId]);
            for (const auto &commit : repositories[repoId]) {
                if (commit.timeStamp >= startTime && commit.timeStamp <= endTime) {
                    result.push_back(commit.id);
                }
            }
        }

        sort(result.begin(), result.end());
        for (int id : result) {
            cout << id << " ";
        }
        cout << endl;
    }
         
      } catch(const std::logic_error& e) {
            cout << "Caught a logic_error: " << e.what()<< endl; 

      }catch(...){
            cout << "caught an unknown exception"<<endl;
      }

      return 0;
}

