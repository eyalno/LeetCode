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


//2539. Count the Number of Good Subsequences
int countGoodSubsequences(string s) {
 
      int res = 0;




      return res;
}


//3006. Find Beautiful Indices in the Given Array I
vector<int> beautifulIndices(string s, string a, string b, int k) {

      vector<int> aIndex;
      vector<int> bIndex;
      vector<int> res;

      size_t index  = s.find(a);    
      while (index != string::npos){
            aIndex.push_back(index);      
            index  = s.find(a,index+1);    
      }

      index  = s.find(b);    
      while (index != string::npos){
            bIndex.push_back(index);      
            index  = s.find(b,index+1);    
      }

      for (const int i:aIndex )
            for (const int j:bIndex )
                  if (abs(i-j) <=k ){
                        res.push_back(i);
                        break;
                  }
      return res;
}


//2964. Number of Divisible Triplet Sums
int divisibleTripletCount(vector<int>& nums, int d) {
        
      int res = 0;
      int len = nums.size(); 

      for (int i = 0 ; i < len; i++  ){
            unordered_set<int> set;

            for (int j = i+1; j < len; j++){

                  //int comp = nums
                  



            }


      }


      return res;


}