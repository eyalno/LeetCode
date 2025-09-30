#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <map>
#include <unordered_map>
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
using namespace std;


//ordered by topics
//1. Two Sum.  Unsorted

vector<int> twoSumUnsorted(vector<int>& nums, int target) {

unordered_map<int,int> map;

for (int i = 0; i < nums.size(); i++){

    int comp = target - nums[i];

    if (map.find(comp) != map.end())
        return {i,map[comp]};
    else
        map[nums[i]] = i;
    
}

return {};
}

//O(n). map O(1)
//space O(n) 

//121. Best Time to Buy and Sell Stock

int maxProfit(vector<int>& prices)
{

    int min = 10000;
    int profit = 0;

    for (int price:prices){

        if (price < min)    
            min = price;
        else if ((price - min) > profit )
            profit = price - min;
    }
    return profit;
}

//57. Insert Interval
vector<vector<int>> insertG(vector<vector<int>>& intervals, vector<int>& newInterval){

    // Input: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
    // Output: [[1,2],[3,10],[12,16]]

    if (newInterval.empty())
       return intervals;

    if (intervals.empty()){
        intervals.push_back(newInterval);
        return intervals;
    }

    vector<vector<int>> res;

    int i = 0 , n = intervals.size() ;

    // current interval dont overlap. left edge case
    while (i < n &&  intervals[i][1] < newInterval[0]){
        res.push_back(intervals[i]);
        i++;
    }
    
    // overlap begins. consume the intervals
    while ( i < n && newInterval[1] >= intervals[i][0]  ){ 
        newInterval[0] = min(newInterval[0] ,intervals[i][0] );
        newInterval[1] = max(newInterval[1] ,intervals[i][1] );
        i++;
    }
    res.push_back(newInterval);

    //remainder 
    while ( i < n){
        res.push_back(intervals[i]);
        i++;
    }

    return res;
}

//15. 3Sum

vector<vector<int>> threeSumG(vector<int>& nums)
{
      vector<vector<int>> res;
      // sort vector
      int size = nums.size();
      sort(nums.begin(), nums.end());

      // since the aray sorted we can only search the nums right to i we
      // avoid duplicates combination by that
      for (int i = 0; i < size - 2; i++)
      { // last 2 dont count
            if (i > 0 && nums[i] == nums[i - 1])
                  continue; // fix a number and skip duplicated after
            // i = i+1 wont work since you want to fix a num for i but j can be the same number but not i again
            // so we want to fix the number first.
            int target = -nums[i];

            int start = i + 1;

            int end = size - 1;

            while (start < end)
            {

                  int currSum = nums[start] + nums[end];

                  if (currSum == target)
                  {
                        res.push_back({ nums[i], nums[start], nums[end] });

                        start++;
                        end--;
                        while (start < end && nums[start] == nums[start - 1])
                              start++;
                        // skip duplicates of found number
                        while (start < end && nums[end] == nums[end + 1])
                              end--;
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

// binary search
vector<vector<int>> threeSumBinarySearchG(vector<int>& nums)
{
      vector<vector<int>> res;
      // sort vector
      int size = nums.size();
      sort(nums.begin(), nums.end());

      // since the aray sorted we can only search the nums right to i we
      // avoid duplicates combination by that
      for (int i = 0; i < size - 2; i++)
      { // last 2 dont count
            if (i > 0 && nums[i] == nums[i - 1])
                  continue; // fix a number and skip duplicated after
            // i = i+1 wont work

            for (int j = i + 1; j < size - 1; j++)
            { // size -1 since k should be after
                  if ((j > i + 1) && nums[j] == nums[j - 1])
                        continue; // skip duplicated

                  int start = j + 1;

                  int end = size - 1;

                  int target = -nums[i] - nums[j];

                  while (start <= end)
                  {
                        int mid = start + (end - start) / 2;

                        if (nums[mid] == target)
                        {
                              res.push_back({ nums[i], nums[j], nums[mid] });
                              break;
                        }
                        else if (nums[mid] < target)
                              start = mid + 1;
                        else
                              end = mid - 1;
                  }
            }
      }
      return res;
      // fixed number and sum 2 problem hashmap/
}

// Hash Set
vector<vector<int>> threeSumHashSetG(vector<int>& nums)
{
      vector<vector<int>> res;
      // sort vector
      int size = nums.size();
      sort(nums.begin(), nums.end());

      // since the aray sorted we can only search the nums right to i we
      // avoid duplicates combination by that

      for (int i = 0; i < size - 2; i++)
      { // last 2 dont count
            unordered_set<int> set;
            if (i > 0 && nums[i] == nums[i - 1])
                  continue; // fix a number and skip duplicated after

            for (int j = i + 1; j < size; j++)
            { // k could be before or after j

                  int complement = -(nums[i] + nums[j]);

                  if (set.find(complement) != set.end())
                  {
                        res.push_back({ nums[i], nums[j], complement });
                        while (j + 1 < size && nums[j] == nums[j + 1])
                              j++; // j+1 means k ,
                  }

                  set.insert(nums[j]); // add any j since it can be a complement
            }
      }
      return res;
      // fixed number and sum 2 problem hashmap/
}


//238.Product of Array Except Self

vector<int> productExceptSelf(vector<int>& nums) {

    if (nums.empty())
        return {};  

        int size = nums.size();            
    if (size == 1)
        return nums;    

    
    vector<int> res(size) , prefix(size), suffix(size) ;

    prefix[0] = nums[0];
    for (int i = 1 ; i < size; i++)
        prefix[i] = prefix[i-1] * nums[i];
    
    suffix[size-1] = nums[size-1];  
    for (int i = size-2; i >= 0 ; i --)
        suffix[i] = suffix[i+1] * nums[i];

    res[0] = suffix[1 ];
    res[size-1] = prefix[size -2];

    // prefix suffix multiplcation of num before and after
    for (int i = 1 ; i < size-1; i++){
        
        res[i] = suffix[ i +1 ] * prefix[i-1];
    }

    return res;
}



//39. Combination Sum backtrack
void combinationSumDFS(vector<int>& candidates, int target,int i, vector<int>& curr,  vector<vector<int>> & res );

vector<vector<int>> combinationSum(vector<int>& candidates, int target) {

    sort(candidates.begin(),candidates.end());
    vector<vector<int>> res;

    vector<int> curr;

    combinationSumDFS(candidates,target,0,curr,res);

    return res;
}

void combinationSumDFS(vector<int>& candidates, int target,int i, vector<int>& curr,  vector<vector<int>> & res ){

    if (target == 0){
        res.push_back(curr);
        return;
    }

    if ( i >= candidates.size() || target < 0 )
        return;

    curr.push_back(candidates[i]);
    combinationSumDFS(candidates,target - candidates[i],i,curr,res);
    
    curr.pop_back(); 

    combinationSumDFS(candidates,target , i+1,curr,res);

}
    
//56.Merge Intervals
vector<vector<int>> merge(vector<vector<int>>& intervals) {

    vector<vector<int>> res;
    // sort by first num the internal vector is const
    sort(intervals.begin(),intervals.end(),[](const vector<int> & a, const vector<int> &b) { return a[0] < b[0]; }  );

    int size = intervals.size();

    res.push_back(std::move(intervals[0]));

    for (int i = 1 ; i < size; i++ ){

        auto & last = res.back();  // last element
        int start = intervals[i][0];

        if ( start <= last[1])
            last[1] = max(last[1],intervals[i][1]);
        else
            res.push_back(std::move(intervals[i]));
    }

    return res;
}

//169. Majority Element
int majorityElementG(vector<int>& nums)
{

      // bit manipulation
      //we check every bit and construct a number from all bits > 50%
      int majElem = 0;
      for (int i = 0; i < 32; i++)
      {
            int count = 0;

            for (int elem : nums)
            {
                  if ((elem & (1 << i)) != 0)
                        count++;
                  if (count > (nums.size() / 2))
                        majElem |= (1 << i);
            }
      }
      return majElem;

      /*since the majority > n/2 will be in the middle in sorted array
      sort(nums.begin(),nums.end());

      return nums[nums.size()/2];
      */

      /*  unordered_map<int,int> freq;
        int size = nums.size();

        for (int num :nums){
            freq[num]++;
            if (freq[num] > (size/2))
                return num;

        }*/
}

//75.Sort Colors\

void sortColors(vector<int>& nums) {
        
// counting sort freq map;
   /*
    vector<int> freq(3,0);
    
    int index=0;
    for (int num:nums)
        freq[num]++;

    for (int i =0; i < freq[0]; i++)
        nums[index++] = 0;


    for (int i =0; i < freq[1]; i++)
        nums[index++] = 1;

    
    for (int i =0; i < freq[2]; i++)
        nums[index++] = 2;


    */
    int size = nums.size();
    int low = 0, mid = 0, high  = size -1;


    //mid scanner dutch national flag we maintain 3 regions.
    while (mid <= high) { // mid reached the end. simulate i loop 

        if (nums[mid] == 0){
            swap(nums[mid],nums[low]);
            mid++;
            low++;
        }
        else if (nums[mid] == 1){
            mid++;
        }
        else{// 2
            swap(nums[mid],nums[high]);
            high--;
        }

    }
}

// 5. Longest Palindromic Substring
string longestPalindromeAmazon(string s) {

    int size = s.size();
    vector<vector<bool>> dp(size,vector<bool>(size,false));
    int maxLen = 1;
    int start = 0;

    if (size == 0)
    return "";


    //2 base cases
    for ( int i = 0 ; i < size; i ++)
        dp[i][i] = true;

    for ( int i = 0; i < size -1 ; i++ ) 
        if (s[i] == s[i+1]){      
            dp[i][i+1] = true;
            maxLen = 2;
            start = i;
        }
    
        //base cases for one and now build all dp for all lengths. 
    for (int len = 3; len <= size; len++){

        for (int i = 0; i <= size - len  ; i ++){ // 8 - 3
            int j = i + len-1;

            if (s[i] == s[j] && dp[i+1][j-1] ){
                dp[i][j] = true;
                maxLen = len;
                start = i;
            }
        } 
    }
    //complexity saving repeating calculations 
    return s.substr(start,maxLen);
}


//283. Move Zeroes

void moveZeroes1(vector<int>& nums) {
        
        int size = nums.size();
        int curr = 0;
        
        for (int i =0; i < size ; i++){

            if (nums[i] != 0)
                nums[curr++] = nums[i];
        }
        
        
        for (int i = curr; i <size; i++)
            nums[i] =0;



    }