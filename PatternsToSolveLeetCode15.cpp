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



using namespace std;
//PatternsToSolveLeetCode15

class PatternsToSolveLeetCode15{
      // 1. Prefix Sum pattern
public:
      PatternsToSolveLeetCode15(vector<int>& nums)
      {
            if (nums.empty())
                  return;
            sums.resize(nums.size());
            sums[0] = nums[0];
            for (int i = 1; i < nums.size(); ++i)
                  sums[i] = sums[i - 1] + nums[i];
      }

      int sumRange(int left, int right)
      {
            if (left == 0)
                  return sums[right];
            return sums[right] - sums[left - 1];
      }

private:
      vector<int> sums;

      int findMaxLength(vector<int>& nums)
      {
            unordered_map<int, int> map;
            int maxLength = 0;

            int size = nums.size();
            if (nums.empty())
                  return 0;

            sums.resize(size);

            // think of base case for edges can help
            //  Base case: prefix sum 0 at index -1
            map[0] = -1;
            sums[0] = nums[0] == 1 ? 1 : -1;

            for (int i = 1; i < size; i++)
                  sums[i] = sums[i - 1] + (nums[i] == 1 ? 1 : -1);

            for (int i = 0; i < size; i++)
            {

                  if (map.find(sums[i]) == map.end())
                        map[sums[i]] = i;
                  else
                  {
                        maxLength = max(maxLength, i - map[sums[i]]);
                  }
            }

            return maxLength;
      }

      int subarraySum(vector<int>& nums, int k)
      {

            int size = nums.size();
            vector<int> sums(size + 1, 0);
            int count = 0;
            sums[0] = 0;

            // build prefix sum.
            for (int i = 1; i <= size; i++)
            {
                  sums[i] = sums[i - 1] + nums[i - 1];
            }
            // looping and using the prefix sum to calculate sub array the brute force
            //  is to to calculate for the range
            for (int i = 0; i < size; i++)
                  for (int j = i + 1; j <= size; j++)
                  {
                        if (k == sums[j] - sums[i])
                              count++;
                  }
            return count;
      }

      int subarraySumBrueteForce(vector<int>& nums, int k)
      {
            // calculating  sum on the fly

            int size = nums.size();
            int count = 0;
            for (int start = 0; start < size; start++)
            {
                  int sum = 0;
                  for (int end = start; end < size; end++)
                  {
                        sum += nums[end]; // savinf the extra loop
                        if (sum == k)
                              count++;
                  }
            }
            return count;
      }

      int subarraySumhasMap(vector<int>& nums, int k)
      {

            int size = nums.size();
            int count = 0;
            unordered_map<int, int> sumMap;
            int sum = 0;
            sumMap[0] = 1;

            for (int i = 0; i < size; i++)
            {
                  sum += nums[i];

                  if (sumMap.find(sum - k) != sumMap.end())
                        count += sumMap[sum - k]; // we are adding all frequencies every time we find a match.

                  sumMap[sum]++;
            }
            return count;
      }

      // 2. Two pointers Pattern
      
      //167. Two Sum II - Input Array Is Sorted
      vector<int> twoSum(vector<int>& numbers, int target)
      {

            int size = numbers.size();
            vector<int> res(2, 0);

            for (int i = 0, j = size - 1; i < j;)
            {
                  int sum = numbers[i] + numbers[j];
                  if (sum == target)
                  {
                        res[0] = i + 1;
                        res[1] = j + 1;
                        return res;
                  }
                  else if (sum > target)
                        --j;
                  else
                        i++;
            }

            return res;
      }

      vector<int> twoSumBinarySearch(vector<int>& numbers, int target)
      {

            int size = numbers.size();

            for (int i = 0; i < size; i++)
            {

                  int complementary = target - numbers[i];

                  // binary search complementary. /lgn performance
                  int start = i + 1;
                  int end = size - 1;

                  while (start <= end)
                  {

                        int mid = start + (end - start) / 2;
                        if (numbers[mid] == complementary)
                              return { i + 1, mid + 1 };
                        else if (numbers[mid] < complementary)
                              start = mid + 1;
                        else
                              end = mid - 1;
                  }
            }

            return {};
      }

      vector<vector<int>> threeSum(vector<int>& nums)
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
      vector<vector<int>> threeSumBinarySearch(vector<int>& nums)
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
      vector<vector<int>> threeSumHashSet(vector<int>& nums)
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

      // Brute Force
      int maxAreaBruteForce(vector<int>& height)
      {

            int max = 0;
            int size = height.size();

            for (int i = 0; i < size - 1; i++)
            {
                  for (int j = i + 1; j < size; j++)
                  {
                        max = std::max(max, (j - i) * (std::min(height[i], height[j])));
                  }
            }
            return max;
      }

      int maxAreaTwoPointers(vector<int>& height)
      {

            // moving the shorter line inward
            // by doing that we skip all permutuations that are not relevant since the shorter line always
            // dominates the size of the container.
            int max = 0;
            int size = height.size();

            int i = 0, j = size - 1;
            while (i < j)
            {

                  max = std::max(max, (j - i) * (std::min(height[i], height[j])));
                  if (height[i] > height[j])
                        j--;
                  else
                        i++;
            }
            return max;
      }

      // Sliding window

      double findMaxAverage(vector<int>& nums, int k)
      {

            int size = nums.size();

            double windowSum = 0.0;

            for (int i = 0; i < k; i++)
                  windowSum += nums[i];
            double max = windowSum;

            for (int i = 1; i <= size - k; i++)
            {

                  windowSum += (-(nums[i - 1]) + nums[i + k - 1]);

                  max = std::max(windowSum, max);
            }
            return max / k;
      }

      // BruteForce
      int lengthOfLongestSubstringBF(string s)
      {

            if (s.empty())
                  return 0;

            int size = s.size();

            if (size == 1)
                  return 1;

            int maxLen = 0;
            // maximum length until a duplicate is found
            for (int i = 0; i < size; i++)
            {
                  unordered_set<char> seen; // reset for every i
                  int currLen = 0;

                  for (int j = i; j < size; j++)
                  {

                        if (seen.find(s[j]) != seen.end())
                              break; // duplicate found;
                        currLen++;
                        seen.insert(s[j]);
                  }

                  maxLen = std::max(maxLen, currLen);
            }
            return maxLen;
      }
      // sliding window approach saves from recounting the length
      int lengthOfLongestSubstring(string s)
      {

            if (s.empty())
                  return 0;

            int size = s.size();

            if (size == 1)
                  return 1;

            int maxLen = 0;
            // maximum length until a duplicate is found

            unordered_map<char, int> prevLoc; // Map to store the last seen index of each character
            int currLen = 0;
            int start = 0;
            // the end is progressing the start jums between duplicates (windows)
            for (int end = 0; end < size; end++)
            {
                  char currCh = s[end];

                  // If the character is already seen and within the current window
                  if (prevLoc.find(currCh) != prevLoc.end() && prevLoc[currCh] >= start)
                  {
                        start = prevLoc[currCh] + 1; // Move the start to the right of the duplicate
                  }

                  // Update the last seen index of the current character
                  prevLoc[currCh] = end;

                  maxLen = std::max(maxLen, end - start + 1);
            }
            return maxLen;
      }

      // 4. Fast and slow pointers array/Linked List

      bool hasCycle(ListNode* head)
      {

            ListNode* slow = head;
            ListNode* fast = head;

            while (fast && fast->next)
            { // if null we know not a cycle also this is the step .

                  slow = slow->next;
                  fast = fast->next->next;

                  if (fast == slow)
                        return true;
            }

            return false;
      }

      // hash set
      bool hasCycleSet(ListNode* head)
      {
            unordered_set<ListNode*> set;

            ListNode* curr = head;

            while (curr != NULL)
            {
                  if (set.find(curr) != set.end())
                        return true;

                  set.insert(curr);
                  curr = curr->next;
            }

            return false;
      }

      bool isHappy(int n)
      {
            unordered_set<int> set;

            while (1)
            {
                  int sum;
                  sum = 0;

                  while (n)
                  {
                        int digit = n % 10;
                        n /= 10;
                        sum += (pow(digit, 2));
                  }

                  n = sum;
                  if (sum == 1)
                        return true;
                  else if (set.find(sum) != set.end())
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

      int removeDuplicates(vector<int>& nums)
      {

            //{0,0,1,1,1,2,2,3,3,4};
            int insertIndex = 1;

            for (int i = 1; i < nums.size(); i++)
            {
                  if (nums[i] != nums[i - 1])
                  {
                        nums[insertIndex] = nums[i];
                        insertIndex++;
                  }
            }
            return insertIndex; // This directly gives the new length
      }

      ListNode* reverseBetween(ListNode* head, int left, int right)
      {

            return head;
      }

      ListNode* swapPairs(ListNode* head)
      {

            return head;
      }

      // 6. Monotonic(increasing /decreasing)  Stack
      vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2)
      {

            return {};
      }

      // 7. K largest  /k smallest /most frequent
      int findKthLargest(vector<int>& nums, int k)
      {

            int size = nums.size();

            priority_queue<int, vector<int>, greater<int>> minHeap;

            for (int num : nums)
            {
                  minHeap.push(num); // Add the current number to the heap
                  if (minHeap.size() > k)
                        minHeap.pop(); // Remove the smallest element if size exceeds k
            }
            return minHeap.top();
      }

      vector<int> topKFrequent(vector<int>& nums, int k)
      {

            unordered_map<int, int> freqMap;
            vector<int> result;
            for (int num : nums)
                  freqMap[num]++;

            priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> minHeap;

            for (const auto& pair : freqMap)
            {

                  int num = pair.first;
                  int freq = pair.second;

                  minHeap.push({ freq, num });
                  if (minHeap.size() > k)
                        minHeap.pop();
            }

            while (!minHeap.empty())
            {

                  result.push_back(minHeap.top().second);
                  minHeap.pop();
            }

            return result;
      }

      vector<int> topKFrequentBucketSort(vector<int>& nums, int k)
      {

            vector<int> result;
            unordered_map<int, int> freqMap;
            int size = nums.size();

            for (int num : nums)
                  freqMap[num]++;

            vector<vector<int>> buckets(size + 1); //  that is the maximum freq if all were the same number

            for (const auto& pair : freqMap)
            {
                  int num = pair.first;
                  int freq = pair.second;
                  buckets[freq].push_back(num);
            }

            for (int i = size; size > 0 && k > result.size(); i--)
                  for (int num : buckets[i])
                  {

                        result.push_back(num);
                        if (result.size() == k)
                              break;
                  }
            return result;
      }

      vector<vector<int>> kSmallestPairs(vector<int>& nums1, vector<int>& nums2, int k)
      {

            vector<vector<int>> result;

            struct heapStruct
            {
                  int sum; // Task priority (lower value = higher priority)
                  int x;
                  int y;
                  heapStruct(int x, int y) : x(x), y(y), sum(x + y) {}
            };

            struct compareHeap
            {
                  bool operator()(const heapStruct& h1, const heapStruct& h2)
                  {
                        return h1.sum < h2.sum;
                  }
            };

            priority_queue<heapStruct, vector<heapStruct>, compareHeap> maxHeap;

            for (int x : nums1)
                  for (int y : nums2)
                  {
                        heapStruct h(x, y);
                        maxHeap.push(h);

                        if (maxHeap.size() > k)
                              maxHeap.pop();
                  }

            while (!maxHeap.empty())
            {
                  heapStruct h = maxHeap.top();
                  result.push_back({ h.x, h.y });
                  maxHeap.pop();
            }
            reverse(result.begin(), result.end());
            return result;
      }

      vector<vector<int>> kSmallestPairsEfficent(vector<int>& nums1, vector<int>& nums2, int k)
      {

            vector<vector<int>> result;

            int m = nums1.size();
            int n = nums2.size();

            set<pair<int, int>> visited;

            priority_queue<pair<int, pair<int, int>>, vector<pair<int, pair<int, int>>>, greater<pair<int, pair<int, int>>>> minHeap;

            minHeap.push({ nums1[0] + nums2[0], {0, 0} });
            visited.insert({ 0, 0 });

            while (k-- > 0 && !minHeap.empty())
            {

                  auto top = minHeap.top();
                  minHeap.pop();

                  int i = top.second.first;
                  int j = top.second.second;
                  result.push_back({ nums1[i], nums2[j] });

                  if (i + 1 < m && visited.find({ i + 1, j }) == visited.end())
                  {
                        minHeap.push({ nums1[i + 1] + nums2[j], {i + 1, j} });
                        visited.insert({ i + 1, j });
                  }

                  if (j + 1 < n && visited.find({ i, j + 1 }) == visited.end())
                  {
                        minHeap.push({ nums1[i] + nums2[j + 1], {i, j + 1} });
                        visited.insert({ i, j + 1 });
                  }
            }
            return result;
      }

      static bool compareIntervals(const vector<int>& a, const vector<int>& b)
      {
            return a[0] < b[0];
      }

      // Overlapping Intervals
      vector<vector<int>> merge(vector<vector<int>>& intervals)
      {

            // Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
            // Output: [[1,6],[8,10],[15,18]]

            vector<vector<int>> res;
            sort(intervals.begin(), intervals.end(), compareIntervals);

            res.push_back(intervals[0]);

            for (int i = 1; i < intervals.size(); i++)
            {

                  auto& last = res.back();
                  if (intervals[i][0] <= last[1])
                        last[1] = max(intervals[i][1], last[1]);
                  else
                        res.push_back(intervals[i]);
            }

            return res;
      }

      vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval)
      {

            // Input: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
            // Output: [[1,2],[3,10],[12,16]]
            vector<vector<int>> merged;

            if (newInterval.empty())
                  return merged;

            if (intervals.empty())
            {
                  merged.push_back(newInterval);
                  return merged;
            }

            int left = 0;
            int right = intervals.size() - 1;
            int newStart = newInterval[0];

            while (left <= right)
            {
                  int mid = left + (right - left) / 2;

                  auto& interval = intervals[mid];
            }

            return merged;
      }

      // 10. Binary Tree Traversal

      vector<string> binaryTreePaths(TreeNode* root)
      {

            // Input: root = [1,2,3,null,5]
            // Output: ["1->2->5","1->3"]

            return {};
      }


};