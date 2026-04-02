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
            //Prefix Sum + Hash Map
            unordered_map<int, int> map;
            int maxLength = 0;

            int size = nums.size();
            if (nums.empty())
                  return 0;

            sums.resize(size); 

            // think of base case for edges can help
            //  Base case: prefix sum 0 at index - 1
            map[0] = -1;  // we assume that the prefix sum 0 appeared at index -1. d
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

       //amazon - "Hacker Rank Demo" 
      
      //prefix sum pattern
      vector<int> minimalHeaviestSetA(vector<int> arr) {
            
            
            int size = arr.size();
            vector<int> result;
            if(size == 0 )
                  return result;
            sort(arr.begin(), arr.end());  // sort ascending
        
            vector<long long> sumArr(size, 0);
            sumArr[0] = arr[0];
            for (int i = 1; i < size; i++)
                sumArr[i] = arr[i] + sumArr[i - 1];
        
            int i = size - 1;
            for (;  i >= 0; i--) {
                long long a = sumArr[size - 1] - (i > 0 ? sumArr[i - 1] : 0);
                long long b = (i > 0 ? sumArr[i - 1] : 0);
                if (a > b)
                    break;
            }
        
            for (int j = i; j < size; j++)
                result.push_back(arr[j]);
        
            return result;
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
            // looping and using the pr   efix sum to calculate sub array the brute force
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
            // calculating  sum on the fly saving space 

            int size = nums.size();
            int count = 0;
            for (int start = 0; start < size; start++)
            {
                  int sum = 0; //we eliminate 1 cell at a time
                  for (int end = start; end < size; end++)
                  {
                        sum += nums[end]; // saving the extra loop
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
            //Input: numbers = [2,7,11,15], target = 9
            //Output: [1,2]

            // 2 sum sorted arra o(n) 2 pointers we move the one that get us closer to the sum
            int size = numbers.size();
            vector<int> res(2, 0);

            int i = 0 , j = size -1;
            while (i < j)
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

      ///result unsorted requiremnet 
      vector<int> twoSumHashOnePass(vector<int>& nums, int target) {
        
            unordered_map<int,int> map;
            
            for (int i = 0; i < nums.size(); i++){
            
                int comp = target - nums[i]; // hold the previous information
            
                if (map.find(comp) != map.end())
                    return {i,map[comp]};
                else
                    map[nums[i]] = i;
                
            }
            
            return {};
      }

      /*
      
            vector<int> twoSum(vector<int>& numbers, int target) {
      
        int size = numbers.size();

      for (int i = 0; i < size; i++){

            int complementary = target - numbers[i];

            //binary search complementary
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
      
      */



      vector<vector<int>> threeSum(vector<int>& nums)
      {
            //basically 2 sum solution with outer loop 2 pointers since retrun order doesn't matter
            vector<vector<int>> res;
            // sort vector
            int size = nums.size();
            if (size < 3) return res;
            sort(nums.begin(), nums.end()); //sorting is important 
            // since the aray sorted we can only search the nums right to i we avoid duplicates combination by that
            //creates order 
            
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

      /*// binary search
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
      */

      // Hash Set
      //like the 2 set solution no benefit over the pointers
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
                  unordered_set<int> set;  // the set is new for every i . we used it jsut for the computation 
                  // set and not map since we only care for the value 
                  if (i > 0 && nums[i] == nums[i - 1])
                        continue; // fix a number and skip duplicated after

                  for (int j = i + 1; j < size; j++)
                  { // k could be before or after j

                        int complement = -(nums[i] + nums[j]); //complement is what k should be I J K

                        if (set.find(complement) != set.end())
                        {
                              res.push_back({ nums[i], nums[j], complement });
                              while (j + 1 < size && nums[j] == nums[j + 1])
                                    j++; // j+1 means k ,
                        }

                        set.insert(nums[j]); // *** add j previous while we loop we populate it . new set per i // we  
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

           // Move the pointer at the shorter line.
            // The shorter line limits the container height.
            // Keeping it while reducing width cannot improve the area,
            // so we skip all those pairs and look for a taller line instead.

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


      int maxSumSubarray(vector<int>& nums, int k) {
      
            int maxSum =0;
            int windowSum =0;

            for (int i = 0; i < k; i++)
                  windowSum += nums[i];

            maxSum = windowSum;
            for (int i = 0 ; i <= nums.size() - k; i++  )
            {
                  windowSum +=  - nums[i] + nums[i+k];
                  maxSum = max(maxSum,windowSum); 
            }
            
            return maxSum;
      }

      int minSubArrayLen(int S, vector<int>& nums) {
            int left = 0, sum = 0;
            int minLen =   numeric_limits<int>::max();
      

            for (int right = 0; right < nums.size(); right ++){ //expanding

                  sum += nums[right];

                  while(sum >= S){

                        minLen = min(minLen, right - left + 1 );
                        sum-=nums[left];  
                        left++; //shrinking 
                  }
            }
      
            return minLen ==  (numeric_limits<int>::max()) ? 0: minLen;
      }


      // BruteForce
      // we find a duplicate we move to the next char.
      int lengthOfLongestSubstringBF(string s)
      {
            //Input: s = "abcabcbb"
            //Output: 3
            //Explanation: The answer is "abc", with the length of 3.

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
      //duplicates create windows from the begining like in BF
      //every time we found duplicate we start new window.
      int lengthOfLongestSubstring(string s)
      {
           //Input: s = "abcabcbb"
            //Output: 3
            //Explanation: The answer is "abc", with the length of 3.
          
            if (s.empty())
                  return 0;

            int size = s.size();

            if (size == 1)
                  return 1;

            int maxLen = 0;
            // maximum length until a duplicate is found

            unordered_map<char, int> prevLoc; // Map to store the last seen index of each character
            
            int start = 0;
            // the end is progressing the start jumps between duplicates (windows)
            for (int end = 0; end < size; end++)  // expanding
            {
                  char currCh = s[end];

                  // If the character is already seen and within the current window
                  if (prevLoc.find(currCh) != prevLoc.end() && prevLoc[currCh] >= start)
                  {
                        start = prevLoc[currCh] + 1; // Shrinking - Move the start to the right of the duplicate 
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
                  fast = fast->next->next; //we build it based on 1/2 steps and the while condition accordingly

                  if (fast == slow)
                        return true;
            }

            return false;
      }

      // hash set
      // not fast slow pointer
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

      // follow the procdure use hash set to detect cycle 
      bool isHappyHash(int n)
      {
            //Input: n = 19
            //Output: true
            //Explanation:
            //12 + 92 = 82
            //82 + 22 = 68
            //62 + 82 = 100
            //12 + 02 + 02 = 1
            
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

      // 5. LinkedList in-place reversal
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
      //496. Next Greater Element I - for each element find the next greater
      vector<int> nextGreaterElement(vector<int>& nums) {
            //[2, 1, 2, 4, 3] input
            //[4, 2, 4, -1, -1] result

            //3,2,1,5
            //5,5,5,-1

            vector<int> res(nums.size(),-1);
            stack<int> s;

            for (int i = 0; i < nums.size(); i++) //looping only once on array
            {  //compare the current num as long as it breaks the order          
                  while ( !s.empty() && nums[i] > nums[s.top()]){       
                        int idx = s.top(); //index of the the element that we found greater element
                        s.pop(); // removing the index that we found the greater
                        res[idx] = nums[i]; // current element is the next greater element 
                        
                  }
                  s.push(i); // store index in stack till we find greater number
            }
            return res;
      }
     
      vector<int> nextSmallerElement(vector<int>& nums) {

            vector<int> res(nums.size(),-1);

            stack<int> s;
            
            for (int i = 0; i < nums.size(); i ++){
                  //increaing stack. 
                  while (!s.empty() && nums[i] < nums[s.top()] ){
                        res[s.top()] = nums[i];
                        s.pop();
                  }
                  s.push(i);
            }
            return res;
      }
     
      vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2)
      {
            // Input: nums1 = [4,1,2], nums2 = [1,3,4,2]
            // Output: [-1,3,-1]
            // Explanation: The next greater element for each value of nums1 is as 
              
            vector<int> res(nums1.size(),-1);
            stack<int> s;
            unordered_map<int,int> m;

            for (int i = 0; i < nums2.size(); i ++ ){
                  
                  while(!s.empty() && nums2[i] > nums2[s.top()]  ){
                        m[nums2[s.top()]] = nums2[i];  // value and next greater element
                        s.pop();
                  }
                  s.push(i);
            }
            //stack can have numbers

            while (!s.empty())
            {
                  m[nums2[s.top()]] = -1;
                  s.pop();
            }
            

            for (int i =0; i < nums1.size(); i++)
                  res[i] = m[nums1[i]];

            return res;
      }

      vector<int> nextGreaterElements2(vector<int>& nums) {
      
            //Input: nums = [1,2,1]
            //Output: [2,-1,2]

            int size = nums.size();
            vector<int> res(size,-1);
            stack<int> s;

            for (int i = 0; i < size; i++){

                  while (!s.empty() && nums[i] > nums[s.top()]){
                        res[s.top()] = nums[i]; 
                        s.pop();
                  }
                  s.push(i);
            }

            for (int i = 0; i < size  ; i++){
                  while (!s.empty() && nums[i] > nums[s.top()]){
                        res[s.top()] = nums[i]; 
                        s.pop();
                  }
            }
            return res;
      }

      //next greater element
      vector<int> dailyTemperatures(vector<int>& temperatures) {
      
            //Input: temperatures = [73,74,75,71,69,72,76,73]
            //Output: [1,1,4,2,1,1,0,0]

            int size = temperatures.size();
            vector<int> res(size,0);

            stack<int> s;

            for (int i = 0; i < size; i++ ){

                  while (!s.empty() && temperatures[i] > temperatures[s.top()]    ){
                        res[s.top()] = i - s.top();
                        s.pop();
                  }
                  s.push(i);
            }

            return res;
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

            // freq map the heap sort by first element in pair
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

      //https://leetcode.com/problems/interval-list-intersections/
      //986. Interval List Intersections

    vector<vector<int>> intervalIntersection(vector<vector<int>>& firstList, vector<vector<int>>& secondList) {
      
      vector<vector<int>> res;

      if (firstList.empty() || secondList.empty())
            return res;   

      for (int i = 0, j =0;  i < firstList.size() && j < secondList.size()  ; ){

            int firstA = firstList[i][0];
            int firstB = firstList[i][1];
            
            int secondA = secondList[j][0];
            int secondB = secondList[j][1];

            int start = max(firstA,secondA);
            int end = min(firstB,secondB);
            
            if (start <= end){ // start < end no intersection
                  res.push_back({start,end});
            }
            
            firstB  < secondB? i++: j++;  //Move the interval that ends first:
                  
      }

      return res;
      
    }

    //https://leetcode.com/problems/insert-interval/description/
    //57. Insert Interval
      vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval)
      {

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

      // 10. Binary Tree Traversal

      vector<string> binaryTreePaths(TreeNode* root)

      {
            // Input: root = [1,2,3,null,5]
            // Output: ["1->2->5","1->3"]

            return {};
      }

      // 11. DFS

     

      //amazon - "Hacker Rank Demo" 
      void countGroupsDFS(vector<string> & related,int i,vector<int> & visited) {
      
            visited[i] = 1;

            for (int j = 0; j < related.size(); j++ )
                  if (related[i][j] == '1' && visited[j] == 0)
                     countGroupsDFS(related, j,visited);
      }                 
      
      int countGroups(vector<string> & related) {

            int size = related.size();

            vector<int> visited(size,0);
            int groupCount = 0;
            for (int i =0 ; i < size; i ++ ){
                  if (visited[i] == 0){
                        countGroupsDFS(related, i,visited);
                        groupCount++;
                  }
            }
            return groupCount;
      }

      class CountGroupsDisJointSet 
      {
         public:
            vector<int> parent;
            vector<int> rank;
      
            int find(int x){
                  if (x != parent[x])
                        parent[x] = find(parent[x]);
                  
                  return parent[x];
            }
      
            bool unite(int x , int y){

                  int rootX = find(x);
                  int rootY = find(y);

                  if (rootX != rootY){

                        if (rank[rootX] > rank[rootY])
                              parent[rootY]  =rootX ;
                        else if (rank[rootX] <  rank[rootY])
                              parent[rootX]  =rootY ;
                        else {
                              parent[rootY]  =rootX ;
                              rank[rootX]++;
                        }
                        return true;
                  }
                  return false;
            }

            int countGroupsDisJointSetF(vector<string> related) {

                  int size = related.size();
                  
                  for (int i = 0; i < size; i ++){
                        parent.push_back(i);
                        rank.push_back(0);
                  }

                  for (int i =0;  i < size; i ++)
                        for (int j =0;  j < size; j ++)
                              if (related[i][j] =='1')
                                    unite(i,j);

                  unordered_set<int> groups;
                  for (int i =0;  i < size; i ++)
                        groups.insert(find(parent[i]));
                  
                  return groups.size();
            }
      };
      
      // 14. Backtracking
      void permuteDFS(vector<int>& nums,vector<int> & visited,vector<int> & curr,vector<vector<int>> & results) {

            if (curr.size() == nums.size())
                  results.push_back(curr);

            for (int i = 0; i < nums.size(); i++){

                  if (visited[i] == 0){

                        visited[i] = 1;
                        curr.push_back(nums[i]);
                        permuteDFS(nums, visited,curr, results);
                        //backtrack 
                        curr.pop_back();
                        visited[i] = 0;
                  }
            }
            return;
      }

      vector<vector<int>> permute(vector<int>& nums) {
      
            vector<vector<int>> results;
            vector<int> curr;
            curr.reserve(nums.size());

            vector<int> visited(nums.size(),0);

            permuteDFS(nums, visited,curr, results);

            return results;
      }


      void subsetsDFS(vector<int>& nums,int index, vector<int> & current,  vector<vector<int>> & result) {

            //beyond tree
            if (index == nums.size() ){
                  result.push_back(current);
                  return;
            }

            //exclude
            subsetsDFS(nums,index +1,current,result);
            
            //include
            current.push_back(nums[index]);
            // we add dfs call and remove 
            subsetsDFS(nums,index +1,current,result); 
            
            //backtrack remove item 
            current.pop_back();

      }
      
      vector<vector<int>> subsets(vector<int>& nums) {

            vector<vector<int>> result;

            vector<int> current;
            current.reserve(nums.size());

            subsetsDFS(nums,0,current,result);

            return result;


      }

      // 15. DP - Dynamic Programming 
      
      //compare the 2 strings from left to right 
      // if s1[i] == s2[i]
            // dp[i][j] = 1 + dp[i+1][j+1]; // +1 and calculate the next charcter DP 
      //else //case when  not equal excluding each character 
            //dp[i][j] = max(dp[i][j+1] , dp[i+1][j]) // exculding 1 from each string

      //dp[i][j]. matrix representation dp[len] base 
      //base cases:
            //base cases are needed for starting point for the formuala
            // when i / j = length(s1/s2) empty strings 
            //dp[len(s1)][j] 0 for all j in the matrix
            //dp[len(i)][s2] 0 for all i in the matrix
            //we populate the dp matrix from smallest subproblem  dp[len(s1)-1][len(s2)-1]  
      int longestCommonSubsequence(string text1, string text2) {
      
            int size1 = text1.size();
            int size2 = text2.size();

            vector<vector<int>> lcs(size1+1,vector<int>(size2+1,0));

            for (int i = size1-1; i >=0; i--)
                  for (int j = size2-1; j >=0;  j--){
                        if (text1[i] == text2[j])
                              lcs[i][j] = 1 + lcs[i+1][j+1];
                        else  
                              lcs[i][j] = max(lcs[i][j+1] , lcs[i+1][j]);
                  }

            return lcs[0][0];
      }

      int lcsRec(string & text1, string & text2,int i , int j,vector<vector<int>> & dp) {
            
            if (i < 0 || j < 0)
                  return 0;

            if (dp[i][j] != -1 )
                  return dp[i][j];

            if (text1[i] == text2[j])
                  return dp[i][j] = 1 +  lcsRec(text1, text2 ,i-1 , j-1, dp);
            else  
                  return dp[i][j] = max(lcsRec(text1, text2 ,i , j-1, dp) , lcsRec(text1, text2 ,i-1 , j, dp));
      
      }

      //recursive solution which is basically a mirror of DP
      int longestCommonSubsequenceRec(string text1, string text2) {

            int size1 = text1.size();
            int size2 = text2.size();

      
            vector<vector<int>> dp(size1,vector<int>(size2,-1));
            
            return  lcsRec(text1, text2 ,size1-1 , size2-1, dp);
      
      
      }

      int lengthOfLIS(vector<int>& nums) {

      
            return 0;
      }
};


class Solution {
public:
    int numSubarraysWithSum(vector<int>& nums, int goal) {
        
      unordered_map<int,int> freq;
      int sum = 0;
      int result =0; 
      freq[0] = 1;

      for (int num:nums){
            sum+= num;

            if (freq.count(sum - goal))
                  result += freq[sum -goal];
            
            freq[sum ]++;
      }
      return result;

    }
};

class Solution1 {
public:
    int findMaxLength(vector<int>& nums) {
        
      int size = nums.size();
      unordered_map<int,int> freq;
      freq[0] = 0;
      int maxLen = 0;
      int sum =0;

      /*
      vector<int> prefixSum(size + 1,0);

      for (int i = 1; i <= size; i++ )
            prefixSum[i] = prefixSum[i-1] + (nums[i-1] == 0 )?-1:1;     
*/

      for (int i = 0; i < size; i++){
            sum += (nums[i] == 0 )?-1:1;
            
            if (freq.count(sum)  )
                  maxLen = max(maxLen, i-freq[sum] ); 
            else  
                  freq[sum] = i;
            

      }
      return maxLen;

        
    }
};

class Solution {
public:
    bool checkSubarraySum(vector<int>& nums, int k) {
        
      unordered_map<int,int> freq;
      freq[0] = -1;

      int sum = 0;
        
      for (int i =0; i <  nums.size(); i ++){
            sum += nums[i];
            int mod = sum % k;

            if (mod < 0) 
                  mod +=k;  
      
            if (freq.count(mod))
                  if ((i - freq[mod] >=2))
                        return true;
            else
                  freq[mod] = i;
      }

      return false;


    }
};


class Solution {
public:
    vector<int> twoSum(vector<int>& numbers, int target) {
        
      //numbers = [2,7,11,15], target = 9

      int size = numbers.size();
      
      int i = 0, j = size-1;
      
      while (i < j){
            int sum = numbers[i] +  numbers[j];
            
            if (sum == target){
                  return {i +1, j+1};
            }
            else if (sum > target )
                        j--;
                  else 
                        i++;
      }

      
      return {};

    }
    //maxArea
};

class Solution {
public:
    int maxArea(vector<int>& height) {
      
            int size = height.size();
            int maxRes = 0 ;
            int i =0, j= size -1;

            while (i<j){

                  maxRes = max(maxRes,(j-i) * min(height[i],height[j]));
                  if (height[i] > height[j])
                        j--;
                  else 
                        i++;

            }            
      return maxRes;

    }
};


class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) { //3 numbers  sum = 0 
      
            int size = nums.size();
            //sort so we can do 2 sum with 2 pointers 
            sort(nums.begin(),nums.end());

            vector<vector<int>> result;

            //fix 1 number and  it becomes 2 sum problem. we can since sorted. 
            for (int i = 0; i < size -2 ; i++  ){
                  if ( i > 0  &&  nums[i] == nums[i-1] ) //we avoid duplicates to avoid duplicate results. sorted  
                        continue; //skip we already calculated

                  int j = i+1;
                  int k = size -1;
                  
                  
                  while (j<k){
                        
                        int sum =  nums[j] + nums[k] + nums[i];

                        if (sum == 0  ){
                              result.push_back({nums[i],nums[j],nums[k]});

                              //skip duplicates
                              while (j<k && nums[j] == nums[j+1] ) j++;
                              while (j<k && nums[k] == nums[k-1] ) k--;

                              //next number
                              j++; 
                              k--;
                        }
                        else if (sum > 0  )
                                   k--;
                              else
                                    j++; 
                  }
            }     
            return result; 
    }
};

class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
     
      int next = 1;
      for (int i = 1; i < nums.size(); i++    ){
            
            if( nums[i] != nums[i-1] ){
                  nums[next] = nums[i];
                  next++;
            }
      }

      return next;


    }
};

class Solution {
public:
    void moveZeroes(vector<int>& nums) {
  
      int zeros = 0;
      int size = nums.size();
      
      int next = 0;
      
      for (int i = 0 ; i < size ; i++){

            if (nums[i] == 0){
                  zeros++;
            }
            else{ 
                  nums[next] = nums[i];
                  next++;
            }
      }
      for (int i =size - zeros ; i < size; i++    )
            nums[i] = 0;
    }
};

class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) { //whose sum is greater than or equal to target. I
      
      int minLen = numeric_limits<int>::max();

      int size = nums.size();
      int sum = 0;
      
      for (int right = 0, left =0; right < size; right ++ ){
      
            sum += nums[right]; //expand first

            while(sum >= target){
                  minLen = min(minLen,right - left +1);
                  sum -=nums[left];
                  left ++;      
            }
      }
      
      return minLen == numeric_limits<int>::max() ? 0: minLen;  


    }
};

class Solution {
public:
    int lengthOfLongestSubstring(string s) {

      unordered_set<char> seen;
      int longest = 0;

      for (int right = 0, left=0 ; right < s.length() ; right++ ){

            while (seen.count(s[right])){
                  seen.erase(s[left]);
                  left++;
            }
            
            seen.insert(s[right]);

            longest = max(longest, right-left + 1 ); 

      }
      return longest;
    }
};

class Solution {
public:
    bool checkInclusion(string s1, string s2) {
        
    }
};


//Input: s = "AABABBA", k = 1
//Output: 4

class Solution {
public:
    int characterReplacement(string s, int k) {
      
      array<int,26> freq= {};
      int result = 0;

      int right =0;
      int left = 0;
      int maxFreqChar =0;

      // window size  - maxFreqChar  <= k     - valid  k can replace 
      while(  right < s.size() ){ //expand window 
            freq[s[right] - 'A']++;
            maxFreqChar = max(maxFreqChar,freq[s[right] - 'A'] );

            while (( right - left +1) - maxFreqChar > k ){ //invalid ->  shrink 

                  freq[s[left] - 'A']--;
                  left++;
            }

            //update result, current window size
            result = max(result,right -left +1 );
            right++;
            
      }
      
      return result;
    }
};

/*
Input: nums = [1,1,1,0,0,0,1,1,1,1,0], k = 2
Output: 6
Explanation: [1,1,1,0,0,1,1,1,1,1,1]
*/
class Solution {
public:
    int longestOnes(vector<int>& nums, int k) {

      int size = nums.size();

      int longest = 0;
      int numOfOnes = 0;
      for (int left = 0, right = 0  ; right < size; ){
            
            numOfOnes += nums[right] == 1 ? 1:0;
            
            int windowSize = right - left +1;
            while (windowSize - numOfOnes > k  ){ //invalid condition  
                  windowSize--;
                  numOfOnes -= nums[left] == 1 ? 1:0;
                  left++;

            }
            longest = max(longest, windowSize);
            
            right++;
      }

      return longest;

    }
};

/*

Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
*/
class Solution {
public:
    string minWindow(string s, string t) {
  
      int sizeT = t.length();

      unordered_map<char,int> charFreq;
      for (char ch:t)
            charFreq[ch]++;

      int minBegin = 0, minEnd = s.length() +1;
      int totalFound = 0;
      
      for (int right = 0,left =0; right < s.length() ; right++ ){
            
            charFreq[s[right]]--;
            if(charFreq[s[right]] >= 0){
                  
                  totalFound++;
            } 
            
            while (totalFound == sizeT ){

                 charFreq[s[left]]++;
                  if (charFreq[s[left]] > 0) {
                        totalFound--;
            }

                  if (minEnd - minBegin +1 > right - left +1   ){
                        minBegin = left; 
                        minEnd = right;
                  }
                   
                  left++;
            }
      }
      
      return minEnd - minBegin == s.length() +1 ? "":s.substr(minBegin,minEnd - minBegin +1); 
    }
};


class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
      
      if (!head || !head->next)
            return nullptr;
      
      //unordered_set<ListNode *> seen;

      ListNode * slow = head , * fast = head , * meetingPt =nullptr ;  

      while (fast && fast->next){
            

            slow = slow->next;
            fast = fast->next->next;

            if (slow == fast){
                  meetingPt = slow;
                  break;
            }
      } 

      if (!meetingPt)
            return nullptr;
      else{
            while (head != meetingPt){
                  head= head->next;
                  meetingPt = meetingPt->next;
            }

      }

      return meetingPt;
    }
};

// 1-2-2-1
class Solution {
public:
    bool isPalindrome(ListNode* head) {
         
      if (!head || !head->next)
            return true;
      
      ListNode* slow = head;
      ListNode* fast = head;

      while (fast && fast ->next){

            slow = slow->next;
            fast = fast->next->next;
      }     
      
      /// fast == nullptr even mid e
      if (fast != nullptr)
            slow = slow->next;

      ListNode* mid = slow;

      ListNode* prev = nullptr;
      ListNode* curr = mid;
      
      while (curr  ){
            
            ListNode* next = curr->next;
            curr->next = prev;

            prev= curr;
            curr= next;
      }

      ListNode * p1 = head;
      ListNode * p2 = prev;

      while (p2){

            if (p1->val != p2->val)
                  return false;
            p1 = p1->next;
            p2 = p2->next;
      }

      return true;


      


      

    }
};


//Input: head = [1,2,3,4,5], n = 2
//Output: [1,2,3,5]

class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
     
            ListNode dummy(0);
            dummy.next = head;

        ListNode* fast = &dummy;
        ListNode* slow = &dummy;
        
        // give fast n+1 head start
        for (int i =0; i <= n; i++)
            fast= fast->next;

      while(fast){
            fast = fast->next;
            slow = slow->next;
      }

      slow->next =slow->next->next; 
     
      
     return dummy.next; 

    }
};

class Solution {
public:
    bool isHappy(int n) {
        
      int curr = n;
      
      unordered_set<int> seen;

      while (1){

            int sum = 0;
            
            while (curr){
                  int mod = curr % 10;      
                  sum += pow(mod,2);
                  curr = curr / 10;
            }
            curr = sum;
            if (sum == 1)     
                  return true;

            if (seen.count(sum))
                  return false;

            seen.insert(sum);
      }



    }
};


class Solution {
public:
    ListNode* reverseList(ListNode* head) {
  
      if (!head || !head->next )     
            return head;

      ListNode* prev = nullptr , * curr = head ;


      while (curr){

            ListNode* nextNode = curr ->next ;
            curr->next = prev;

            prev = curr;
            curr = nextNode;

      }

      return prev;


    }
};


//Input: head = [1,2,3,4,5], left = 2, right = 4
//Output: [1,4,3,2,5]

class Solution {
public:
    ListNode* reverseBetween(ListNode* head, int left, int right) {
  
      ListNode * prev = nullptr, * curr = head;

      int i = 0;
      while ( i < left  ){
            ListNode* nextNode = curr ->next ;
            prev = curr;
            curr = nextNode;
            i++;
      }

      while ( i < right ){
      
      
      }
      
      

      
      



    }
};