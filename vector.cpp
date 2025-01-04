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




// 915. Partition Array into Disjoint Intervals
int partitionDisjoint(vector<int>& nums)
{
      int size = nums.size();
      int maxLeft[size];
      int minRight[size];
      int currentMax;

      switch (1) { //

      case 1: { //Two Arrays

            maxLeft[0] = nums[0];
            minRight[size - 1] = nums[size - 1];

            for (int i = 1; i < size; i++)
            {
                  maxLeft[i] = max(nums[i], maxLeft[i - 1]);
            }

            for (int i = size - 2; i >= 0; i--)
            {
                  minRight[i] = min(nums[i], minRight[i + 1]);
            }

            for (int i = 1; i < size; i++)
                  if (maxLeft[i - 1] <= minRight[i])
                        return i;

            return -1;
      }
      case 2: {  //one array
            int minRight[size];

            currentMax = nums[0];
            minRight[size - 1] = nums[size - 1];

            for (int i = size - 2; i >= 0; i--)
            {
                  minRight[i] = min(nums[i], minRight[i + 1]);
            }

            for (int i = 1; i < size; i++)
            {

                  if (currentMax <= minRight[i])
                        return i;

                  currentMax = max(nums[i], currentMax);
            }
            return -1;
      }

      case 3: {  // No array

            int currentMax;
            int possibleMax;
            int length = 1;
            currentMax = possibleMax = nums[0];

            for (int i = 1; i < size; i++)
            {
                  if (nums[i] < currentMax)
                  {
                        length = i + 1;
                        currentMax = possibleMax;
                  }
                  else
                  {
                        possibleMax = max(possibleMax, nums[i]);
                  }
            }
            return length;

      }
      }
}

//977. Squares of a Sorted Array
vector<int> sortedSquares(vector<int>& nums)
{
      int size = nums.size();
      vector<int> res(size, 0);

      switch (1) { //

      case 1: { // STL iterators (forward and reverse).
            vector<int>::iterator it = nums.begin();
            vector<int>::reverse_iterator rit = nums.rbegin();

            if (it == nums.end() || rit == nums.rend() || nums.empty())
            {
                  cout << "empty vector";
                  return nums;
            }

            //-15,-10,0,1, 2, 3, 4, 5
            for (auto currIt = res.rbegin(); it <= rit.base() && currIt != res.rend(); currIt++)
            {

                  int leftSq = (*it) * (*it);
                  int rightSq = (*rit) * (*rit);

                  if (leftSq > rightSq)
                  {
                        *currIt = leftSq;
                        it++;
                  }
                  else
                  {
                        *currIt = rightSq;
                        rit++;
                  }
            }

            return res;
      }
      case 2: { //Index-based two-pointer technique.

            vector<int> res(size, 0);
            int left = 0;
            int right = size - 1;

            for (int i = size - 1; i >= 0; --i)
            {
                  int leftSq = pow(nums[left], 2);
                  int rightSq = pow(nums[right], 2);

                  if (leftSq > rightSq)
                  {
                        res[i] = leftSq;
                        left++;
                  }
                  else
                  {
                        res[i] = rightSq;
                        right--;
                  }
            }
      }

      }
      return res;
}


//905. Sort Array By Parity
vector<int> sortArrayByParity(vector<int>& nums)
{
      switch (1) {

      case 1: { // sort

            sort(nums.begin(), nums.end(), [](int a, int b) {
                  return (a % 2 == 0) && (b % 2 == 1);
                  });

            return nums;
      }
      case 2: { // Two pointers

            int size = nums.size();

            for (int i = 0, j = size - 1; i < j;)
            {
                  if ((nums[i] % 2 == 1) && (nums[j] % 2 == 0))
                  {
                        int temp = nums[i];
                        nums[i] = nums[j];
                        nums[j] = temp;
                  }
                  if (nums[i] % 2 == 0)
                        ++i;
                  if (nums[j] % 2 == 1)
                        --j;
            }
            return nums;
      }
      }
}


//925. Long Pressed Name
bool isLongPressedName(string name, string typed)
{

      bool bIs = true;

      int i = 0;
      for (int j = 0; j < typed.length();)
      {
            if ((i < name.length()) && name[i] == typed[j])
            {
                  i++;
                  j++;
            }
            else if (j != 0 && typed[j] == typed[j - 1])
                  j++;
            else
                  return false;
      }

      return i == name.length();
}


//561. Array Partition
int arrayPairSum(vector<int>& nums)
{

      sort(nums.begin(), nums.end());

      int maxSum = 0;

      for (int i = 0; i < nums.size(); i += 2)
            maxSum += nums[i];

      return maxSum;
}

void countingSort(vector<int>& arr)
{
      // Find the maximum element in the array
      // int maxElement = *std::max_element(arr.begin(), arr.end());
      // 10^4
      // 4, 2, 1, 0, 3, 3, 1, 2
      for (int element : arr)
            element += pow(10, 4);

      // Create a count array to store the occurrences of each element
      vector<int> count(pow(10, 4) + 1, 0);

      // Count the occurrences of each element
      for (int element : arr)
            count[element]++;

      // Calculate the cumulative sum in the count array
      for (int i = 1; i < count.size(); i++)
            count[i] += count[i - 1];

      // Build the sorted array using the count array
      // Iterate through the original array in reverse order:
      // For arr[7] = 2, its final position is count[2] - 1 = 2, so place 2 at
      vector<int> sortedArr(arr.size());
      for (int i = arr.size() - 1; i >= 0; --i)
      {
            int element = arr[i];
            int position = count[element] - 1;
            sortedArr[position] = element;
            count[element]--;
      }

      for (int element : sortedArr)
            element -= pow(10, 4);

      // Copy the sorted array back to the original array
      arr = sortedArr;
}



int climbStairsHelper(int n, vector<int>& memo)
{
      if (n == 0)
            return 1;
      if (n == -1)
            return 0;

      if (memo[n] != -1)
            return memo[n];
      memo[n] = climbStairsHelper(n - 1, memo) + climbStairsHelper(n - 2, memo);

      return memo[n];
}

//70. Climbing Stairs
int climbStairs(int n)
{

      switch (1) {

      case 1: { //dynamic programming

            if (n == 1)
            {
                  return 1;
            }

            vector<int> dp(n + 1, 1);

            dp[1] = 1;
            dp[2] = 2;

            for (int i = 3; i <= n; i++)
                  dp[i] = dp[i - 1] + dp[i - 2];

            return dp[n];
      }
      case 2: { //recursive and  memoization 

            vector<int> memo(n + 1, -1);

            return climbStairsHelper(n, memo);
      }
      case 3: { //no memoization time limit excedded


            if (n == 0)
                  return 1;
            if (n == -1)
                  return 0;

            return climbStairs(n - 1) + climbStairs(n - 2);

      }
      }
}



//217. Contains Duplicate
bool containsDuplicate(vector<int>& nums)
{

      switch (1) {

      case 1: { //set
            unordered_set<int> set;

            for (int num : nums)
                  if (set.count(num) > 0)
                        return true;
                  else
                        set.insert(num);

            return false;
      }
      case 2: { //sort

            sort(nums.begin(), nums.end());

            for (int i = 0; i < nums.size() - 1; i++)
                  if (nums[i] == nums[i + 1])
                        return true;

            return false;
      }
      }
}


// binary search
int search(vector<int>& nums, int target)
{
      int start = 0;
      int end = nums.size() - 1;

      while (start <= end)
      {
            int mid = (start + end) / 2;

            if (target == nums[mid])
                  return mid;

            if (target > nums[mid])
                  start = mid + 1;
            else
                  end = mid - 1;
      }

      return -1;
}


vector<int> intersection(vector<int>& nums1, vector<int>& nums2)
{

      vector<int> ret;

      if (nums1.size() > nums2.size())
      {
            unordered_set<int> set(nums2.begin(), nums2.end());
            for (int num : nums1)
            {
                  auto it = set.find(num);
                  if (it != set.end())
                  {
                        ret.push_back(num);
                        set.erase(it);
                  }
            }
      }

      return ret;
}

int singleNumber(vector<int>& nums)
{

      // use XOR

      int ret = 0;

      for (int num : nums)
            ret ^= num;
      return ret;

      // 2∗(a+b+c)−(a+a+b+b+c)=c
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
