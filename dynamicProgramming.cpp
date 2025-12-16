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