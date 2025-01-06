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

int romanToInt(string s)
{
      unordered_map<char, int> map;

      map['I'] = 1;
      map['V'] = 5;
      map['X'] = 10;
      map['L'] = 50;
      map['C'] = 100;
      map['D'] = 500;
      map['M'] = 1000;

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
      // right to left

      int i = s.size() - 1;
      int sum = 0;

      while (i >= 0)
      {
            int lastNum = 0;
            char ch = s[i];

            int num = map[ch];

            if (num < lastNum)
                  sum -= num;
            else
                  sum += num;

            lastNum = num;
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

//278. First Bad Version

// Mock global variable representing the first bad version
int firstBad = 4; // Example: Assume version 4 is the first bad version

// Implementation of the isBadVersion API
bool isBadVersion(int version) {
      return version >= firstBad;
}

int firstBadVersion(int n) {

      int i = 1;
      int j = n;

      while (i < j) {

            int mid = i + (j - i) / 2;
            if (isBadVersion(mid)) {
                  j = mid;
            }
            else
                  i = mid + 1;
      }
      return i;
}


class MyHashMap
{

private:
#define primeBase 769
      array<LinkedList*, primeBase> arr;

public:
      MyHashMap()
      {

            for (int i = 0; i < primeBase; i++)
                  arr[i] = new LinkedList;
      }

      void put(int key, int value)
      {
            int mappedVal = key % primeBase;

            LinkedList* bucket = arr[mappedVal];
            bucket->insertFront(key);
      }

      void remove(int key)
      {
            int mappedVal = key % primeBase;
            LinkedList* bucket = arr[mappedVal];
            bucket->remove(key);
      }

      bool contains(int key)
      {
            int mappedVal = key % primeBase;
            LinkedList* bucket = arr[mappedVal];
            if (bucket->search(key))
                  return true;

            return false;
      }
};


class MyHashSet
{

private:
#define primeBase 769

      array<BST*, primeBase> arr;

public:
      MyHashSet()
      {

            for (int i = 0; i < primeBase; i++)
                  arr[i] = new BST;
      }

      void add(int key)
      {
            int mappedVal = key % primeBase;

            BST* bucket = arr[mappedVal];
            bucket->insert(key);
      }

      void remove(int key)
      {
            int mappedVal = key % primeBase;
            BST* bucket = arr[mappedVal];
            bucket->deleteNode(key);
      }

      bool contains(int key)
      {
            int mappedVal = key % primeBase;
            BST* bucket = arr[mappedVal];
            if (bucket->search(key))
                  return true;

            return false;
      }
};



int key(int i, int j)
{
      size_t hash_i = hash<int>{}(i), hash_j = hash<int>{}(j);
      int hashed = (int)(hash_i ^ (hash_i >> 32));
      return (hashed << 5) - 1 + (int)(hash_j ^ (hash_j >> 32));
}

//(n choose k)
int binomialCoefficient(int n, int k)
{

      int rowCol = key(n, k);
      static unordered_map<int, int> cache;

      if (cache.count(rowCol) > 0)
            return cache[rowCol];

      if (k == 0 || n == 0 || n == k)
            return cache[rowCol] = 1;

      return cache[rowCol] = (binomialCoefficient(n - 1, k - 1) + binomialCoefficient(n - 1, k));
}



string longestCommonPrefix(vector<string>& strs, int l, int r)
{

      // divide and conquer
      if (l == r)
            return strs[l];

      int mid = l + (r - l) / 2;

      string leftStr = longestCommonPrefix(strs, l, mid);
      string rightStr = longestCommonPrefix(strs, mid + 1, r);

      int minLen = min(leftStr.length(), rightStr.length());

      for (int i = 0; i < minLen; i++)
      {

            if (leftStr[i] != rightStr[i])
                  return leftStr.substr(0, i);
      }

      return leftStr.substr(0, minLen);
}

bool isCommonPrefix(vector<string>& strs, int len)
{

      string prefix = strs[0].substr(0, len + 1);

      for (int j = 1; j < strs.size(); j++)
      {

            if (prefix != strs[j].substr(0, len + 1))
                  return false;
      }

      return true;
}
string longestCommonPrefix(vector<string>& strs)
{

      // using binary search.

      if (strs.empty())
            return "";

      int minLen = INT_MAX;
      for (string str : strs)
            minLen = min((int)str.length(), minLen);

      int low = 0;
      int high = strs.size() - 1;

      int mid = 0;

      while (low <= high)
      {

            int mid = low + (high - low) / 2;

            if (isCommonPrefix(strs, mid))
                  low = mid + 1;
            else
                  high = mid - 1;
      }

      return strs[0].substr(0, mid + 1);

      string ret;
      // flower","flow","flight"

      // 1 <= strs.length <= 200
      // 0 <= strs[i].length <= 200

      // Using Trie

      Trie t;

      for (string str : strs)
            t.insert(str);

      TrieNode* current = t.root;

      while (current->children.size() == 1)
      {
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

      // Horizantal scanning compare 1 string to all and remove characters
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

      // Vertical scanning  - compare charters instead of ull strings
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




int main()
{
      return 0;
}
