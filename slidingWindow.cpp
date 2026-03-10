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



//https://leetcode.com/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/description/

//1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit

//max(window) - min(window) <= limit
//max-min never decreases when expanding
// the predictable monotonic property is that the new item will increase decrease min/max
//So at every moment we need to know:
	//•	the minimum element in the window
	//•	the maximum element in the window
//We store the current window elements inside a multiset.

int longestSubarray(vector<int>& nums, int limit) {

    multiset<int> currWindow; //set keeps it sorted
    size_t res = 0;

    for (int beg = 0, end = 0; end < nums.size(); end++  ){ //expanding

        size_t maxVal = 0 , minVal = 0;
        currWindow.insert(nums[end]);
        maxVal = *currWindow.rbegin(); //since it is sorted
        minVal = *currWindow.begin();

        while ( (maxVal - minVal) > limit && beg <= end  ){ //shrinking invalid 
                currWindow.erase(currWindow.find(nums[beg])); // remove left of window
                beg++;
                minVal = *currWindow.begin();
                maxVal = *currWindow.rbegin();
        }

        res = max(res,currWindow.size());    
             
    }
    return res;
}


