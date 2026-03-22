#include <vector>
#include <algorithm>
#include <iostream>
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


//Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
//Output: 6

// 	•	positive → adding it helps → keep it
//  •	negative → adding it hurts → start fresh
//

int maxSubArray(vector<int>& nums) {

    int maxSum = nums[0];
    int currSum = nums[0]; // state     

    for (int i = 1; i < nums.size(); i ++ ){
        currSum = max(nums[i],currSum + nums[i]); // currSum + nums[i] Transition 
        maxSum = max (currSum, maxSum);     //currSum reuse
    }

    return maxSum;
}

//https://leetcode.com/problems/longest-increasing-subsequence/description/

//300. Longest Increasing Subsequence

/*
    Input: nums = [10,9,2,5,3,7,101,18]
    Output: 4
    Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4.

    State : dp[i] = length of the LIS ending at i
    Transition: 
            If nums[j] < nums[i]   
    Reuse:
            dp[i] = max(dp[i], dp[j] + 1) adding 1 more 

*/

int lengthOfLIS(vector<int>& nums) {

    int size = nums.size();
    if (size == 0) return 0;

    vector<int> dp(size,1); // for every index the minimum is 1 
    // State    
    
    int maxLength = 1;
    // we build base cases. iterative

    //for every i we check if previous dp  If nums[j] < nums[i]  
    for (int i = 1; i < size ; i++) //O(n^2)
        for (int j = 0; j < i ; j++){
            if (nums[i] > nums[j]){ //Transition
                dp[i] = max (dp[i], dp[j] +1);//Reuse
                maxLength = max (maxLength,dp[i]); 
            }
        }

return maxLength;
    
}

