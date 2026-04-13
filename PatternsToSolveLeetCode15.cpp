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
  
      
      
      if (!head || !head->next || left == right )
            return head;

      ListNode dummy(0);
      ListNode * prev = &dummy;

      dummy.next = head;


      // move prev to before left
      for (int i = 1;  i < left; i ++)
            prev= prev->next;

      ListNode * curr =  prev->next;//2
      ListNode * leftTail = prev;
      
      ListNode* leftHead =curr ;  
      
      
      prev = nullptr;   

      //reversing
      for (int i = 0 ; i < right - left +1 ; i++){
            ListNode* temp = curr -> next ;
            curr -> next = prev;
            prev = curr;
            curr = temp;
      }
      
      leftHead -> next = curr;
      leftTail ->next = prev; 

      return dummy.next; //might be new head

    }
};


//Input: head = [1,2,3,4,5], k = 3
//Output: [3,2,1,4,5]

class Solution {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
      ListNode dummy(0);

      dummy.next = head;

      ListNode * curr = head;

      ListNode * tailPrev = &dummy; // tail of the previous sublist  
      while (curr){

            ListNode * headCurr = curr; // head in the sub list that we reverse

            ListNode * tempCurr = curr; 
            //check if to reverse
            for (int i = 0; i < k; i++  ){
                  if ( !tempCurr ){
                        return dummy.next;
                  }
                  tempCurr = tempCurr->next;
            }
            
            ListNode * prev = nullptr;    
                    

            for (int i = 0; i < k; i++){
                  ListNode * temp = curr->next;
                  curr->next = prev;
                  prev = curr;
                  curr = temp;
            }

            headCurr ->next = curr;
            tailPrev->next = prev;
            tailPrev = headCurr;
      }

        return dummy.next;
    }
};

class Solution {
public:
    void reorderList(ListNode* head) {
      
      if (!head || !head->next)
            return;
      
      stack<ListNode*> st;
      ListNode* slow = head, * fast = head;
      ListNode* mid;
      
      while (fast->next && fast->next->next){ // find end of first half //works for 
             
            fast = fast->next->next;
            slow = slow->next;
      }

      ListNode* p2 = slow->next;
      
      slow->next = nullptr;
      
      //push to stack second list 
      while (p2  ){ 
            ListNode* temp = p2->next;
            p2->next = nullptr;
            st.push(p2);     
            p2 = temp;
      }
      
      //weave
      ListNode* curr = head;
      while (curr  && !st.empty()){
            
            ListNode* temp = curr->next;
            ListNode* top = st.top();
            st.pop();

            curr->next =top;
            top->next = temp;
            curr = temp;
      }

    }
};

//Input: nums1 = [4,1,2], nums2 = [1,3,4,2]     3,4,-1,-1
//Output: [-1,3,-1]
class Solution {
public:
    vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2) {
      
      unordered_map<int,int> map;
      stack<int> st;
      vector<int> res(nums1.size());

      for ( int i = 0 ; i < nums2.size(); i++ ){
            
            while(!st.empty() && st.top() < nums2[i]  ){ // decreasing stack - remove when greater found 
                  int top = st.top();
                  st.pop();
                  map[top] = nums2[i];
            }
            st.push(nums2[i]);
      }
      
      while(!st.empty()){
              int top = st.top();
                  st.pop();
                  map[top] = -1;

      }
      
      for (int i =0; i < nums1.size(); i++)
            res[i] = map[nums1[i]];

        return res;
    }
};

//Input: temperatures = [73,74,75,71,69,72,76,73]
//Output: [1,1,4,2,1,1,0,0]
class Solution {
public:
    vector<int> dailyTemperatures(vector<int>& temperatures) {
      
      vector<int> res(temperatures.size(),0);
      stack<int> st;

      for(int i = 0; i < temperatures.size(); i++){

            int currNum = temperatures[i];
            while (!st.empty() && temperatures[st.top()] < currNum  ){
                  int top = st.top();
                  st.pop();
                  res[top] = i -  top;
            }
            st.push(i);
      }

      return res;

      
    }
};

//Input: nums = [1,2,1]
//Output: [2,-1,2]
class Solution {
public:
    vector<int> nextGreaterElements(vector<int>& nums) {
        
      stack<int> st;
      int size = nums.size();

      vector<int> res (size,-1);

      for (int i = 0; i < size * 2 ; i++  ){
            
            while(!st.empty() && nums[i%size] > nums[st.top()]  ){
                  int top = st.top();
                  st.pop();
                  res[top] = nums[i%size];
            }

            if (i<size)
                  st.push(i);

      }

      return res;
    }
};


//Input: heights = [2,1,5,6,2,3]
//Output: 10

class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        
      stack<int> st;
      int size = heights.size();

      //smallest increasing stack bot min top 

      //virtual boubdry  - index
      vector<int> nextSmallest(size,size); //means can be extended all the way  to the right  
      
      vector<int> prevSmallest(size,-1); // 

      for (int i = 0; i < size; i++ ){

            while ( !st.empty() && heights[st.top()] >= heights[i] ){ //found smaller breaking condition 
                  int top = st.top();
                  st.pop();
                  nextSmallest[top] =  i;
            }
            st.push(i);
      }
      
      while (!st.empty())
            st.pop();
      
      
      for (int i = size -1; i >= 0; i-- ){

            while ( !st.empty() && heights[st.top()] >= heights[i] ){ //found smaller breaking condition 
                  int top = st.top();
                  st.pop();
                  prevSmallest[top] =  i;
            }
            st.push(i);
      }
      
      while (!st.empty())
            st.pop();

      int maxRect = 0;

      for (int i = 0; i < size; i++ ){

            int right =  nextSmallest[i];
            int left =  prevSmallest[i];

            // for each height we want to see how far can be extended
            
            int area = (right -left-1) * heights[i];  // it can extend left until just before the previous smaller bar -1 -1
            maxRect = max(maxRect,area);  
      }

      return maxRect;


    }
};


class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        
      stack<int> st;
      int n = heights.size();
      int maxArea = 0;

      for (int i =      0; i<=n ; i++){ //expanding right

            int currHeight = (i == n ) ? 0 : heights[i]; // right boundry

            while (!st.empty() &&  heights[st.top()] >  currHeight){ //expanding left  since we pop left 
                  int h =  heights[st.top()];
                  st.pop();

                  int leftBoundary = st.empty() ? -1 : st.top(); //left boundry
                  int width = i - leftBoundary - 1;  // left is changin with every pop

                  maxArea = max(maxArea , width * h);
            }
            st.push(i);
      }

      return maxArea;

    }
};


//sum of all continiouns subarray ranges = sum of all subarray maximums - sum of all subarray minimums
//find for every num   
//Input: nums = [4,-2,-3,4,1]
//Output: 59
class Solution {
public:
    long long subArrayRanges(vector<int>& nums) {
        




    }
};


class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
  
      priority_queue<int,vector<int> ,greater<int>> minHeap; //min is root so the rest are greater


      for (int num:nums){
            minHeap.push(num);

             if (minHeap.size() > k   ){
                  minHeap.pop();
             }

      }
      return minHeap.top();

      
    }
};



class Solution {
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
      
      vector<int> res;
      unordered_map<int,int> freq;
      priority_queue < pair<int,int> , vector< pair<int,int> >, greater<pair<int,int>>> minHeap; 

      for (int num:nums)
            freq[num]++;
      
      for (auto &[num,count]:freq){
            minHeap.push({count,num});

            if (minHeap.size() > k)
                  minHeap.pop();
      }    
      
      while (!minHeap.empty())
      {
            res.push_back(minHeap.top().second);
            minHeap.pop();
      }
      return res;

    }
};

//Input: points = [[3,3],[5,-1],[-2,4]], k = 2
//Output: [[3,3],[-2,4]]
class Solution {
public:
    vector<vector<int>> kClosest(vector<vector<int>>& points, int k) {
     // minimum -> max heap  default

     priority_queue<pair<int,int>> maxHeap;
     vector<vector<int>> res;

     int i =0;
     for (auto point:points){
            int distance = pow(point[0],2) + pow(point[1],2);
            maxHeap.push({distance,i});

            if (maxHeap.size() > k)
                  maxHeap.pop();

            i++;
     }

     while (!maxHeap.empty())
     {
            
            res.push_back(points[maxHeap.top().second]);
             maxHeap.pop();
     }
     return res;


      
    }
};


//KthLargest kthLargest = new KthLargest(3, [4, 5, 8, 2]);
//kthLargest.add(3); // return 4
//kthLargest.add(5); // return 5

class KthLargest {
public:
    KthLargest(int k, vector<int>& nums) {
      //          this->k = k;  

      for (int num:nums){
            minHeap.push(num);
            if (minHeap.size() > k)
                  minHeap.pop();
            
      }     
}
    
    int add(int val) {
         minHeap.push(val);
            minHeap.pop();
            minHeap.top();
    }

private:
    priority_queue<int,vector<int>,greater<int>> minHeap;
};


//MedianFinder medianFinder = new MedianFinder();
//medianFinder.addNum(1);    // arr = [1]
//medianFinder.addNum(2);    // arr = [1, 2]
//medianFinder.findMedian(); // return 1.5 (i.e., (1 + 2) / 2)

class MedianFinder {
public:
    MedianFinder() {
        
        nextQueue = true;

    }
    
    void addNum(int num) {
      
      if (nextQueue ){
            
            if (maxHeap.empty())
                  minHeap.push(num);
            else{
            

                  int topLeft = maxHeap.top(); 
                  if (topLeft >= num ){ //left is greater . we swap.
                        maxHeap.pop(); 
                        minHeap.push(topLeft);
                        maxHeap.push(num);
                  }
                  else  
                        minHeap.push(num);
                  }
      }
      else{
            int topRight = minHeap.top();
            if (topRight <= num ){
                  minHeap.pop();
                  maxHeap.push(topRight);
                  minHeap.push(num);
            }
            else  
                  maxHeap.push(num);
      }

      nextQueue = !nextQueue;

    }
    
    double findMedian() {
        
      return !nextQueue  ? minHeap.top() : minHeap.top() + (maxHeap.top() - minHeap.top())/2.0 ;
}

private:
    
    priority_queue<int> maxHeap; //left false
    priority_queue<int,vector<int>,greater<int>> minHeap; //right  true
    bool nextQueue; 
};

/**
 * Your MedianFinder object will be instantiated and called as such:
 * MedianFinder* obj = new MedianFinder();
 * obj->addNum(num);
 * double param_2 = obj->findMedian();
 */



 class Solution {
public:

      struct Compare {
        bool operator()(ListNode* a, ListNode* b) {
            return a->val > b->val;
        }
    };

    ListNode* mergeKLists(vector<ListNode*>& lists) {
      
      if (lists.size() == 0)
            return nullptr;

      priority_queue <ListNode * ,vector<ListNode *>, Compare > minHeap;

      ListNode * head = nullptr;;

      for (ListNode * list:lists)
            if (list)
                  minHeap.push(list);

        if (minHeap.empty()) {
            return nullptr;
        }


      head = minHeap.top();
      minHeap.pop();
      
      if (head->next)
                  minHeap.push(head->next);
      
      ListNode * curr = head;

      while (!minHeap.empty()){

            ListNode * top = minHeap.top(); 
            minHeap.pop();
            
            curr->next = top;
            curr = curr->next;
            if (top->next)
                  minHeap.push(top->next);
            
      }
      curr->next = nullptr;
      return head;
    }
};

//return the kth smallest element in the matrix.
//Input: matrix = [[1,5,9],[10,11,13],[12,13,15]], k = 8
//Output: 13
//The elements in the matrix are [1,5,9,10,11,12,13,13,15], and the 8th smallest number is 13
//you're ordering all numbers from smallest to largest, then taking position k.

//using heap to sort the array 

class Solution {
public:
    int kthSmallest(vector<vector<int>>& matrix, int k) {
      
      int size = matrix.size();

      priority_queue< pair<int,int>, vector<pair<int,int>> ,greater<pair<int,int>> > minHeap; 
      //priority_queue<int> maxHeap;  // top is max last
      
      //A[i][j] is equivalent to A[i * N + j]
      //since each row is sorted the min of all this will be the minimum   
      for (int i = 0; i < size ; i ++  ){
            minHeap.push({matrix[i][0], i*size }  );
      }

      //extract k times
      for (int i = 0; i < k-1 ; i ++  ){
            
            auto top = minHeap.top();
            minHeap.pop();

            int row =  top.second /size;
            int col = top.second % size; 

            //push next
            if ( col  + 1 < size)
                  minHeap.push( {matrix[row][col  + 1] ,row* size + col +1 } );
            
      }

      return minHeap.top().first;

    }
};


//Input: k = 2, w = 0, profits = [1,2,3], capital = [0,1,1]
//Output: 4

class Solution {
public:
    int findMaximizedCapital(int k, int w, vector<int>& profits, vector<int>& capital) {
        
      int size = profits.size();

      vector<pair<int,int>> pAc;

      priority_queue<int> maxHeap;

      //pair the two
      for (int i = 0; i < size; i++)
            pAc.push_back({capital[i],profits[i]});

      sort(pAc.begin(),pAc.end());

      int canWork = 0;

      for (int finishedProject = 0; finishedProject < k; finishedProject++ ){

            while ( canWork < size && pAc[canWork].first <= w  ){
                  maxHeap.push(pAc[canWork].second);
                  canWork++;
            }

            if (maxHeap.empty())
                  break;

            w+= maxHeap.top();
            maxHeap.pop();
      }

      return w;

    }
};

//Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
//Output: [[1,6],[8,10],[15,18]]
//Explanation: Since intervals [1,3] and [2,6] overlap, merge them into [1,6].
class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
  
      sort(intervals.begin(),intervals.end());

      int size = intervals.size();
      vector<vector<int>> res; 

      if (size){
            res.push_back(intervals[0]);
      }

      for (int i = 1; i < size; i++){

            if (res.back()[1] < intervals[i][0]  )
                  res.push_back(intervals[i]);
            else
               res.back()[1] = max (intervals[i][1],res.back()[1]);   

      } 

return res;

    }
};


//Input: intervals = [[0,30],[5,10],[15,20]]
//Output: false


class Solution {
public:
    bool canAttendMeetings(vector<vector<int>>& intervals) {

      sort(intervals.begin(),intervals.end());

      for (int i = 0; i + 1 < intervals.size() ; i++){
            auto current = intervals[i];
            auto next = intervals[i+1];
            
            if (current[1] > next[0]   )
                  return false;


      }     

        return true;
    }
};

//Input: intervals = [[0,30],[5,10],[15,20]]
//Output: 2

class Solution {
public:
    int minMeetingRooms(vector<vector<int>>& intervals) {
       
       sort(intervals.begin(),intervals.end());
    
       //How many meetings are still running when this one starts?
      priority_queue<int, vector<int>, greater<int>> minHeap; 

      int rooms = 0;
      int size = intervals.size();

      for (int i = 0 ; i < size; i++){
            // if the current time greater then earliest min - we can grab the room   
            // we only clear 1 slot since  other  rooms were needed at the time  
            if (!minHeap.empty() && minHeap.top() <= intervals[i][0]   ) /// intervals[i][0] the current time
                  minHeap.pop();

            
            minHeap.push(intervals[i][1]); //room added 
           
      }
      
      return minHeap.size(); 

    }
};

//Input: intervals = [[1,2],[2,3],[3,4],[1,3]]
//Output: 1
class Solution {
public:
    int eraseOverlapIntervals(vector<vector<int>>& intervals) {
      
      if (intervals.empty())
            return 0;

      int size = intervals.size();
      sort(intervals.begin(),intervals.end());
      int res = 0;

      int prevEnd = intervals[0][1];
      
      for (int i = 1; i < size; i++ ){
            
            auto & curr = intervals[i]; 
            
            if ( curr[0] >= prevEnd) //no overlap 
                  prevEnd = curr[1];
            else { // overlap
                  res++;
                  prevEnd = min (prevEnd, curr[1]);
            }

      }

      return res;

    }
};


//Input: points = [[10,16],[2,8],[1,6],[7,12]]
//Output: 2
// [1,6][2,8][7,12][10,16]  sorted

class Solution {
public:
    int findMinArrowShots(vector<vector<int>>& points) {
        
      if (points.empty())
            return 0;

      sort(points.begin(), points.end());
      int size = points.size();
      auto & prevEnd = points[0][1];
      int arrows =1;

      for (int i =1; i < size; i++){
            auto & curr =  points[i];

            if (curr[0] <= prevEnd   ) //overlap 
                  prevEnd = min(prevEnd,curr[1]);
            else{ //no overlap new arrow
                  arrows++;
                  prevEnd = curr[1];
            }
      }
    return arrows;
    }
};



//Input: trips = [[2,1,5],[3,3,7]], capacity = 4
//Output: false

class Solution {
public:
    bool carPooling(vector<vector<int>>& trips, int capacity) {

       sort(trips.begin(), trips.end(),
             [](const vector<int>& a, const vector<int>& b) {
                 return a[1] < b[1];   // sort by start
             });

      priority_queue < pair<int,int> , vector<pair<int,int>> , greater<pair<int,int> > > minHeap;
      //end - passangers 

      int size = trips.size();

      int prevEnd = trips[0][2];
      
      int empty = capacity ; 

      for (int i = 0; i < size; i ++){

            auto & curr = trips[i];
            
            while( !minHeap.empty() &&  minHeap.top().first <= curr[1] ){ //clear seats 
                  empty += minHeap.top().second;
                   minHeap.pop();
            }
            
            empty -= curr[0];

            if (empty < 0)
                  return false;

            minHeap.push({curr[2],curr[0]});
            
      }

      return true;
    }
};


class Solution {
public:
    int search(vector<int>& nums, int target) {
        
      int size = nums.size();

      int l = 0;
      int r = size -1;

      while (l <= r){
      
            int mid = l + (r-l) / 2;


            if  ( target  <   nums[mid] )
                  r = mid - 1;
            else if   ( target  >   nums[mid] )
                  l = mid + 1;
                  else
                        return mid; 

      }
      return -1; 

    }
};



class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        
      int size = nums.size();

      int l = 0;
      int r = size -1;

      while (l <= r){
      
            int mid = l + (r-l) / 2;


            if  ( target  <   nums[mid] )
                  r = mid - 1;
            else if   ( target  >   nums[mid] )
                  l = mid + 1;
                  else
                        return mid; 

      }
      return l-1; 


    }
};


//Input: nums = [5,7,7,8,8,10], target = 8
//Output: [3,4]
//   0 + 5-0 /2
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
     
      int size = nums.size();      
      int l = 0;
      int r = size -1;

      int right = -1;
      int left = -1;
      

    while (l <= r){
      
            int mid = l + (r-l) / 2;

            if  ( target  ==  nums[mid] ){ //  
                  right = mid ; //canidate
                  l = mid + 1; //keep searching right
            }else 
                  if  ( target  <   nums[mid] ) 
                        r = mid - 1;
                  else 
                        l = mid + 1;
            
      }      

      l = 0;
      r = size -1;

       while (l <= r){
      
            int mid = l + (r-l) / 2;

            if  ( target  ==  nums[mid] ){  
                  left = mid ; //canidate
                  r = mid - 1; //keep searching left
            } else
                  if  ( target  <   nums[mid] ) 
                        r = mid - 1;
                  else 
                        l = mid + 1;
      
      }      


      return {left,right};

    }
};


bool isBadVersion(int);

class Solution {
public:
    int firstBadVersion(int n) {
     
      int l = 1;
      int r = n;

      int res = 0;
      while (l <= r){

            int mid = l + (r-l) /2;

            if (isBadVersion(mid)){
                  res = mid; //candiate
                  r = mid -1;
            }
            else 
                  l = mid +1;


      }

      return res;


    }
};

//Input: nums = [1,2,3,1]
//Output: 2
class Solution {
public:
    int findPeakElement(vector<int>& nums) {
  
      int size = nums.size();

      int l = 0;
      int r = size - 1;

      // nums[i] vs nums[i+1]  if greater increasing so the peak on the rightr continue.
      while (l < r){
            
            int mid = l + (r - l) /2;

            if (nums[mid] < nums[mid+1]){ // increasing    peak on the right
                  l = mid + 1;  //mid cannot be peak
                  
            }
            else{ // nums[mid] > nums[mid+1] peak on the left
                  r = mid ; // decreasing mid could be peak
            }
      }

      return l;

    }
};

//Input: nums = [4,5,6,7,0,1,2], target = 0
//Output: 4

class Solution {
public:
    int search(vector<int>& nums, int target) {
        
      int size = nums.size();

      int l = 0;
      int r = size - 1;

      while (l <= r){

            int mid = l + (r-l) /2;

            if (nums[mid] == target)
                  return mid;

            if ( nums[l] <= nums[mid]  )// left is sorted we can check if target in range

                  if ( nums[l] <= target &&  target < nums[mid] ) // found in the sorted 
                        r = mid - 1;
                  else 
                        l = mid + 1; // not found smaller vector
            else
                   if ( nums[mid] < target &&  target <= nums[r] ) // found in the sorted  right
                        l = mid + 1;
                  else 
                        r = mid - 1; // not found smaller vector

      }
      return -1 ;
    }
};



//Input: nums = [3,4,5,1,2]
//Output: 1
class Solution {
public:
    int findMin(vector<int>& nums) {
        
      int size = nums.size();

      int l = 0;
      int r = size - 1;

      int minVal = nums[0]; //a valid canidate 

      
      while (l <= r){ // 
      
            int mid = l + (r-l) /2;

            if ( nums[l] <= nums[mid]  ){ // left is sorted we can check if target in range
                  minVal = min( nums[l] , minVal);
                  l = mid + 1;
            }
            else{   // right is sorted 
                  minVal = min( nums[mid] , minVal);
                  r  = mid -1;
            }       
      }

      return minVal;

    }
};

//Input: piles = [3,6,7,11], h = 8
//Output: 4

class Solution {
public:
    
      bool canFinish(vector<int>& piles, int h, int mid){

            //(p + k - 1) / k. 
            int hours =0;
            for (int i =0;  i < piles.size() ; i ++){
                  hours += (piles[i] + mid - 1 )/mid;    
                  if (hours > h)
                        return false;
            }
            return true;
      }

      int minEatingSpeed(vector<int>& piles, int h) {
        
            int l = 1;
            
            int r = piles[0];
            for (int pile:piles)
                  r = max(r,pile);

            int res = r;

            while (l <= r){
                  int mid = l + (r-l )/2;

                  if  (canFinish(piles,h ,mid)){
                        r = mid - 1;
                        res = mid;
                  }
                  else
                        l = mid + 1 ;
            }

            return res;

    }
};

//A[i][j] is equivalent to A[i * N + j] 
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
      
      int m = matrix.size();
      int n = matrix[0].size();

      int l  = 0;
      int r = n*m -1;

      while (l <= r){
            int mid = l + (r-l) / 2;

            int midI = mid / n; 
            int midJ = mid % n;

            if ( matrix[midI][midJ] == target )
                  return true;
            else if (matrix[midI][midJ] < target)
                  l = mid +1;
            else 
                  r = mid -1;

      }

      return false;
    } 
};

class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
      
      int m = matrix.size();
      int n = matrix[0].size();

      int i = 0;
      int j = n - 1;
      
      while (i < m && j > -1   ){

            if (matrix[i][j] == target)
                  return true;
            else if (matrix[i][j] > target)
                  j--;
            else
                  i++;
      } 

      return false;

    }
};

/*
Input: grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
Output: 1
*/

class Solution {

private:
      void dfs(vector<vector<char>>& grid, int i, int j ){

            int m = grid.size();
            int n = grid[0].size();

            if (i<0 || j <0 || i == m || j ==n  || grid[i][j] == '0' || visited.count(i*n + j) )
                  return;
            
            visited.insert(i*n + j);

            
            dfs(grid,i,j+1);
            dfs(grid,i,j-1);
            dfs(grid,i+1,j);
            dfs(grid,i-1,j);
      



      }

      unordered_set<int> visited;

public:
    int numIslands(vector<vector<char>>& grid) {
      
      if (grid.empty() || grid[0].empty() )
            return 0;

      int m = grid.size();
      int n = grid[0].size();

      int count = 0;
      for (int i = 0 ; i < m; i++)
            for (int j = 0 ; j < n; j++){

                  if (!visited.count(i*n + j) && grid[i][j] == '1' ){
                        count++;
                        dfs(grid,i,j);
                  }
            }
      return count;
    }
};


class Solution {

private:
      int dfs(vector<vector<int>>& grid, int i, int j ){

            int m = grid.size();
            int n = grid[0].size();

            if (i < 0 || j < 0 || i == m || j ==n  || grid[i][j] == 0 )
                  return;
            
            grid[i][j] = 0;

            dfs(grid,i,j+1);
            dfs(grid,i,j-1);
            dfs(grid,i+1,j);
            dfs(grid,i-1,j);
      
      }



public:
    int maxAreaOfIsland(vector<vector<int>>& grid) {
        
        if (grid.empty() || grid[0].empty() )
            return 0;

      int m = grid.size();
      int n = grid[0].size();

      int maxIslands = 0;

       for (int i = 0 ; i < m; i++)
            for (int j = 0 ; j < n; j++)
                  maxIslands = max(maxIslands, dfs(grid , i , j));      
            


      return maxIslands;

    }
};