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


void rotate(vector<int>& nums, int k)
{

      // 1 <= nums.length <= 105
      //-231 <= nums[i] <= 231 - 1
      // 0 <= k <= 105

      // 123 2
      int size = nums.size();
      if (k == 0 || k % size == 0)
            return;
      k %= size;
      // using reverse
      reverse(nums.begin(), nums.end());
      reverse(nums.begin(), nums.begin() + k);
      reverse(nums.begin() + k, nums.end());

      // using cyclic replacements
      // 123 2
      /*
      int size = nums.size();

      if (k == 0 || k%size == 0 )
            return;

      int start = 0;
      int curr = 0 ;
      int num = nums[start];

      for (int i =0; i < size ;i ++){

            int step = (curr+k) % size ;

            int temp = nums[step];
            nums[step ] = num;
            num = temp;
            if (step == start){

                  start++;
                  curr = start;
                  num = nums[curr];
            }
            else
                  curr = step;

      }

      */

      /*

      //extra vector
      int size = nums.size();

      vector<int> temp(size,0);

      if (k == 0 || k%size == 0 )
            return;

      for (int i = 0; i < size ; i++){

           temp[(i+k)%size] = nums[i];
      }

      nums.assign(temp.begin(),temp.end());
      */

      /*
      for (int i =0 ; i < k; i++){

            int last = nums[nums.size() -1];

            nums.pop_back();

            nums.insert(nums.begin(),last);

      }

      */

      /*
      int size = nums.size();
      if (k == 0 || k%size == 0 )
            return;

      for (int j =0 ; j < k ; j++){

            int num = nums[0];

            for (int i = 0; i < size ; i++   ){

                  int temp = nums[((i+1) % size) ];

                  nums[(i+1) % size] = num;

                  num = temp;
            }

      }
      */
}



int minSubArrayLen(int target, vector<int>& nums)
{

      int len = numeric_limits<int>::max();

      // 2 pointer

      int left = 0;

      int sum = 0;

      for (int i = 0; i < nums.size(); i++)
      {

            sum += nums[i];

            while (sum >= target)
            {

                  len = min(len, (i - left + 1));
                  sum -= nums[left++];
            }
      }

      // using sum array

      /*
      vector<int> sums(nums.size());

      sums[0] = nums[0];


      for (int i =1; i < nums.size(); i++)
            sums[i] =sums[i-1] + nums[i];


       for ( int i = 0 ; i< nums.size(); i ++){
            for ( int j = i; j < nums.size(); j ++){

                  int sum = sums[j] - sums[i] +nums[i] ;

                  if (sum >= target){
                        len = min(len,(j-i+1));
                        break;
                  }
            }
       }

      */

      // Brute force
      /*
       for ( int i = 0 ; i< nums.size(); i ++){
            for ( int j = i; j < nums.size(); j ++){

                  int sum = 0 ;
                  for (int k = i; k <= j; k++)
                        sum += nums[k];

                  if (sum >= target){
                        len = min(len,(j-i+1));
                        break;
                  }
            }
       }


       */

      return len == numeric_limits<int>::max() ? 0 : len;
}


int binarySearch(vector<int>& nums, int target)
{

      int low = 0;
      int high = nums.size() - 1;

      while (low <= high)
      {

            int mid = low + (high - low) / 2;

            if (nums[mid] == target)
                  return mid;
            else if (nums[mid] > target)
                  high = mid - 1;
            else
                  low = mid + 1;
      }

      return -1;
}


vector<int> getRow(int rowIndex)
{

      // pascal
      // rowIndex == 0 return 1

      // writing to same row

      vector<int> ans(rowIndex + 1, 1);

      for (int i = 1; i < rowIndex; i++)
      {

            for (int j = i; j > 0; j--)
            {
                  ans[j] = ans[j] + ans[j - 1];
            }
      }

      return ans;

      // save one row instaed of caching everything

      /*
      vector<int> curr,prev= {1};


      for (int i=1; i <= rowIndex; i++){

            curr.assign(i+1,1);

            for (int j =1; j <i; j++ ){

                curr[j] = prev[j-1] + prev[j];
            }

            prev = move(curr);
      }

      return prev;
      */

      // recursive
      /*
      static unordered_map<int,int> cache;
      cache.clear();

      vector<int> ret(rowIndex);
      for (int i =0 ; i <= rowIndex; i++){

            ret.push_back(binomialCoefficient(rowIndex,i));
      }

      return ret;
      */

      /* build the triangle

      vector<vector<int>> tri;

      for (int i =0 ; i < rowIndex +1; i++  ){

            vector<int> row(i + 1,1);

            for (int j =1; j <i; j++ ){

                row[j] = tri[i-1][j-1]  +tri[i-1][j];
            }

            tri.push_back(row);
      }

      return tri[rowIndex];
      */
}



vector<vector<int>> generate(int numRows)
{

      vector<vector<int>> triangle;

      // Pascal
      for (int i = 0; i < numRows; i++)
      {
            vector<int> row(i + 1, 1);

            for (int j = 1; j < i; j++)
            {
                  row[j] = triangle[i - 1][j - 1] + triangle[i - 1][j];
            }

            triangle.push_back(row);
      }

      return triangle;
}

vector<int> spiralOrder(vector<vector<int>>& mat)
{
      // 1,2,3,6,9,8,7,4,5

      const int VISITED = 101;

      vector<int> ret;

      int rows = mat.size();
      int cols = mat[0].size();

      int directions[4][2] = { {0, 1}, {1, 0}, {0, -1}, {-1, 0} };
      int currDirection = 0;

      int i = 0, j = 0;

      ret.push_back(mat[0][0]);
      mat[0][0] = VISITED;

      for (int k = 1; k < rows * cols;)
      {

            while (i + directions[currDirection][0] >= 0 && i + directions[currDirection][0] < rows &&
                  j + directions[currDirection][1] >= 0 && j + directions[currDirection][1] < cols &&
                  mat[i + directions[currDirection][0]][j + directions[currDirection][1]] != VISITED)
            {

                  k++;
                  i += directions[currDirection][0];
                  j += directions[currDirection][1];
                  ret.push_back(mat[i][j]);
                  mat[i][j] = VISITED;
            }

            currDirection = (currDirection + 1) % 4;
      }

      return ret;
}

vector<int> findDiagonalOrder(vector<vector<int>>& mat)
{

      vector<int> ret;
      // 1,2,4,7,5,3,6,8,9]
      int rows = mat.size();
      int cols = mat[0].size();

      for (int k = 0; k < rows + cols - 1; k++)
      {

            int i = k < cols ? 0 : k - cols + 1;
            int j = k < cols ? k : cols - 1;

            vector<int> reverse;

            while (i < rows && j > -1)
            {
                  if (k % 2 == 0)
                        reverse.insert(reverse.begin(), mat[i][j]);
                  else
                        ret.push_back(mat[i][j]);
                  j--;
                  i++;
            }

            if (k % 2 == 0)
            {

                  for (int a : reverse)
                        ret.push_back(a);
                  reverse.clear();
            }
      }
      return ret;
}

vector<int> plusOne(vector<int>& digits)
{

      int size = digits.size();
      int i = size - 1;

      for (; i >= 0; i--)
      {

            if (digits[i] == 9)
                  digits[i] = 0;
            else
            {
                  digits[i]++;
                  break;
            }
      }

      if (digits[0] == 0)
            digits.insert(digits.begin(), 1);

      return digits;

      // while(digits.rbegin() != digits.rend() )
}

int dominantIndex(vector<int>& nums)
{

      //[3,6,1,0]

      int index = -1;

      int max = 0;

      int next = 0;

      for (int i = 0; i < nums.size(); i++)
      {

            int num = nums[i];

            if (num > max)
            {
                  next = max;
                  max = num;
                  index = i;
            }
            else if (num > next)
                  next = num;
      }

      if (max >= 2 * next)
            return index;
      else
            return -1;
}

int pivotIndex(vector<int>& nums)
{

      int size = nums.size();
      int pivot = -1;

      int sum = 0, leftSum = 0;

      for (int i : nums)
            sum += i;

      for (int i = 0; i < size; i++)
      {

            if (leftSum == (sum - leftSum - nums[i]))
            {
                  pivot = i;
                  break;
            }

            leftSum += nums[i];
      }

      /*

      vector<int> leftSum(size,nums[0]);
      vector<int> rightSum(size,nums[size-1]);



      for (int i = 1; i < size ; i++){

            leftSum[i] = nums[i] + leftSum[i-1];

            rightSum[size -i -1] = rightSum[size - i] + nums[size - i -1]  ;
      }


      for (int i =  0;i < size-1; i++ ){

            if (leftSum[i] == rightSum[i])
            {
                  pivot = i;
                  break;
            }

      }

      if (pivot == -1)
            if (size > 1){

                  if (leftSum[size-2] == 0)
                        pivot = size -1;
                  else if ((rightSum[1] == 0))
                        pivot = 1;
            }else
                  pivot = 0;


      */

      return pivot;
}

vector<int> findDisappearedNumbers(vector<int>& nums)
{

      // 4,3,2,7,8,2,3,1

      set<int> mySet;

      vector<int> ret;

      int size = nums.size();

      for (int i = 0; i < size; i++)
      {

            mySet.insert(nums[i]);
      }

      for (int i = 1; i <= size; i++)
      {

            if (mySet.find(i) == mySet.end())
                  ret.push_back(i);
      }

      return ret;
}

int thirdMax(vector<int>& nums)
{

      // 1,2,2,5,3,5
      int max = 0;
      set<int> mySet;
      long l;
      pair<int, int> a;

      l = numeric_limits<int>::min();

      for (int i = 0; i < nums.size(); i++)
      {

            int num = nums[i];

            if (mySet.size() < 3)
            {

                  mySet.insert(num);
                  continue;
            }
            else
            {
                  if (num > *mySet.begin())
                  {
                        if (mySet.find(num) == mySet.end())
                        {

                              mySet.erase(mySet.begin());
                              mySet.insert(num);
                        }
                  }
            }
      }

      if (mySet.size() == 3)
            max = *mySet.begin();
      else
            max = *mySet.rbegin();

      /*
      priority_queue<int, vector<int>, greater<int>> minHeap;
      unordered_set<int> mySet;
      //3,2,1

            int max = 0 ;

            for (int i = 0; i <nums.size(); i++){

                  int num = nums[i];

                  if(mySet.find(num) != mySet.end())
                        continue;

                  if(minHeap.size() != 3){
                        minHeap.push(num);
                        mySet.insert(num);
                  }
                  else{
                        if (minHeap.top() < num){
                              minHeap.pop();
                              minHeap.push(num);
                              mySet.insert(num);
                        }
                  }

            }

            if(minHeap.size() != 3){
                  while(!minHeap.empty() ){
                        max = minHeap.top();
                        minHeap.pop();
                  }
            }
            else
                  max = minHeap.top();
      */

      return max;
}

int findMaxConsecutiveOnes(vector<int>& nums)
{

      // 1,0,1,1,0,1,1,1,0,0,0

      // 1,0,1,1,0

      bool bFlag = false;
      int max = 0;
      int count = 0;

      int minCount = 0;

      for (int i = 0; i < nums.size(); i++)
      {

            if (nums[i] == 1)
            {
                  count++;
                  continue;
            }

            if (nums[i] == 0)
            {

                  if (bFlag == false)
                  {
                        bFlag = true;
                        count++;
                        minCount = count;
                        continue;
                  }
                  else
                  {
                        bFlag = false;
                        i--;
                        if (count > max)
                              max = count;
                        count = count - minCount;
                  }
            }
      }

      if (count > max)
            max = count;

      /*
            int max =0;

              for (int i = 0; i< nums.size(); i++){

                  int iZeros = 0;

                  for (int j = i; j< nums.size(); j++){

                      if(nums[j] == 0)
                          iZeros++;

                      if (iZeros <= 1){
                          max = std::max(max,j - i + 1 );
                      }



                  }
              }

      */
      return max;
}

int heightChecker(vector<int>& heights)
{
      // 1,1,4,2,1,3

      vector<int> expected(heights.begin(), heights.end());

      sort(expected.begin(), expected.end());

      int num = 0;

      for (int i = 0; i < heights.size(); i++)
      {

            if (heights[i] != expected[i])
                  num++;
      }

      return num;
}

void moveZeroes(vector<int>& nums)
{
      // 0,1,0,3,12
      // 1,3,12,0,0

      int writePtr = 0;

      for (int readPtr = 0; readPtr < nums.size(); readPtr++)
      {

            if (nums[readPtr] != 0)
            {
                  nums[writePtr] = nums[readPtr];
                  writePtr++;
            }
      }

      for (; writePtr < nums.size(); writePtr++)
            nums[writePtr] = 0;
}

vector<int> replaceElements(vector<int>& arr)
{
      // 17,18,5,4,6,1
      // 18,6,6,6,1,-1

      int size = arr.size();
      int max = arr[size - 1];
      arr[size - 1] = -1;

      for (int i = size - 2; i >= 0; i--)
      {
            int temp;

            temp = arr[i];

            if (arr[i] > max)
            {
                  max += arr[i];
                  arr[i] = max - arr[i];
                  max -= arr[i];
            }
            else if (arr[i] < max)
                  arr[i] = max;
      }
      return arr;
}

bool validMountainArray(vector<int>& arr)
{

      int size = arr.size();

      if (size < 3)
            return false;

      // 3,1,1

      int i = 1;

      while (arr[i - 1] < arr[i])
      {
            i++;
      }

      if (i == 1 || i == size)
            return false;

      while (arr[i - 1] > arr[i] && i < size)
      {
            i++;
      }

      return size == i;
}

bool checkIfExist(vector<int>& arr)
{

      int size = arr.size();
      map<int, int> myMap;

      //[7,1,14,11]
      for (int i = 0; i < size; i++)
      {

            int num = arr[i];

            int iDouble = num * 2;

            if (myMap.find(num) == myMap.end())
                  myMap[num] = i;

            if (myMap.find(iDouble) != myMap.end() && myMap[iDouble] != i)
                  return true;

            if ((num % 2 == 0 && myMap.find(num / 2) != myMap.end()))
            {

                  if (myMap[num / 2] != i)
                        return true;
            }
      }

      return false;
}

int removeDuplicates(vector<int>& nums)
{

      //{0,0,1,1,1,2,2,3,3,4};
      int size = nums.size();
      int duplicates = 0;
      int insertIndex = 1;
      for (int i = 1; i < size; i++)
      {

            if (nums[i] != nums[i - 1])
            {
                  nums[insertIndex] = nums[i];
                  insertIndex++;
            }
      }
      return insertIndex;
}

int removeElement(vector<int>& nums, int val)
{
      int size = nums.size();

      int reversePtr = 0;
      for (int i = 0; i < size - reversePtr; i++)
      {

            if (nums[i] != val)
            {
                  continue;
            }
            else if (nums[size - 1 - reversePtr] == val)
            { // both equal
                  --i;
                  ++reversePtr;
            }
            else
            {
                  nums[i] = nums[size - 1 - reversePtr];
                  ++reversePtr;
            }
      }
      //[0,1,2,2,3,0,4,2] 2

      // 3,2,2,3 3

      nums.resize(size - reversePtr);
      return size - reversePtr;
}

void duplicateZeros(vector<int>& arr)
{

      int size = arr.size();
      int zeros = 0;

      for (int i = 0; i < size; i++)
      {
            if (arr[i] == 0)
                  zeros++;
      }

      vector<int> temp(size + zeros);
      //{1,0,2,3,0,4,5,0};
      int i = 1;
      for (; zeros > 0; i++)
      {

            if (arr[size - i] == 0)
            {
                  temp[size + zeros - i] = 0;
                  temp[size + zeros - i - 1] = 0;
                  zeros--;
            }
            else
                  temp[size + zeros - i] = arr[size - i];
      }

      for (int j = size - i + 1; j < size; j++)
            arr[j] = temp[j];
}

void mergeSort(vector<int>& nums1, int m, vector<int>& nums2, int n)
{

      for (int i = 0; i < n; i++)
      {
            nums1[m + i] = nums2[i];
      }

      sort(nums1.begin(), nums1.end());
}

void merge(vector<int>& nums1, int m, vector<int>& nums2, int n)
{

      /*  int r =  m + n;
        int i = m-1;
        int j = n-1;

        for ( r = r-1 ; r > 0, j >=0 ; r--    ){

              if ( nums2[j] >= nums1[i] ){
                    nums1[r] = nums2[j];
                    j--;
              }
              else{
                    nums1[r] = nums1[i];
                    nums1[i] = 0;
                    i--;
              }
        }
        */
}



vector<int> twoSumMap(vector<int>& nums, int target)
{

      map<int, int> myMap;

      int size = nums.size();

      vector<int> indices(2);
      //&&  myMap[compliment] != i
      for (int i = 0; i < size; i++)
      {

            int compliment = target - nums[i];

            if (myMap.find(compliment) != myMap.end() && myMap[compliment] != i)
            {

                  indices[0] = myMap[compliment];
                  indices[1] = i;
                  break;
            }

            myMap[nums[i]] = i;
      }

      return indices;
}

//sorted array
vector<int> twoSum(vector<int>& nums, int target)
{

      // numbers[low] > (1 << 31) - 1 - numbers[high]

      // 2147483647
      /*
      int a = (2<<30) -  1;
      int p = (1<<31) -  1;
      long b = pow(2,32) /2;

      unsigned int       c = pow(2,32) ;


      int d = numeric_limits<int>::max();
      long e = numeric_limits<long>::min();
      long long f = numeric_limits<long long>::min();
      double g = numeric_limits<double>::min();
      float h = numeric_limits<float>::min();
      size_t i = numeric_limits<size_t>::min();
      */
      // long d = ((1<<32) -1)>>1 ;

      // 2,7,11,15
      //  two pointer solution for sorted array

      int size = nums.size();

      vector<int> indices(2);

      for (int i = 0, j = size - 1; i < j;)
      {

            // check overflow
            if (nums[i] > (1L << 31) - 1 - nums[j])
            {
                  j--;
                  continue;
            }

            int sum = nums[i] + nums[j];

            if (sum == target)
            {
                  indices[0] = i;
                  indices[1] = j;
            }
            else if (sum < target)
                  i++;
            else
                  j--;
      }

      /*
            int size = nums.size();

            vector<int> indices(2);

            for (int i = 0 ; i < size-1; i++  ){
                  for (int j = i+1; j < size; j++){

                        if (nums[i] + nums[j] == target){
                              indices[0] = i;
                              indices[1] = j;
                              return indices;
                        }


                  }

            }

      */

      return indices;
}


void floodFillHelper(vector<vector<int>>& image, int sr, int sc, int originalColor, int newColor)
{

      int rows = image.size();
      int cols = image[0].size();

      if (sr < 0 || sr >= rows || sc < 0 || sc >= cols || image[sr][sc] != originalColor || image[sr][sc] == newColor)
            return;

      image[sr][sc] = newColor;

      floodFillHelper(image, sr - 1, sc, originalColor, newColor);
      floodFillHelper(image, sr + 1, sc, originalColor, newColor);
      floodFillHelper(image, sr, sc - 1, originalColor, newColor);
      floodFillHelper(image, sr, sc + 1, originalColor, newColor);
}

vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int color)
{

      if (image.empty())
            return image;

      int rows = image.size();
      int cols = image[0].size();

      if (sr < 0 || sr >= rows || cols < 0 || sc >= cols)
            return image;

      int originalColor = image[sr][sc];

      if (originalColor == color)
            return image;

      floodFillHelper(image, sr, sc, originalColor, color);

      return image;
}

//252. Meeting Rooms
bool canAttendMeetings(vector<vector<int>>& intervals)
{
      if (intervals.empty())
            return true;

      // Sort intervals based on their start times
      sort(intervals.begin(), intervals.end());

      for (int i = 0; i < intervals.size() - 1; i++)
      {
            if (intervals[i][1] > intervals[i + 1][0])
                  return false;
      }

      return true;
}
//Sₙ = n/2 × (a₁ + aₙ)

//Intuition create pairs forward and backward the sum / 2
//268. Missing Number71.1%Easy
int missingNumber(vector<int>& nums) {



int n = nums.size();

int sum = 0;
int seriesSum = (n* (n+1))/2;  //trancation last /2 multiply first 

for (int num:nums)
      sum += num;

      return seriesSum - sum;

}


 bool isPalindrome(int x) {

      if ((x<  0 ) || (x%10 == 0 && !x)) //e.g 40 never 
            return false;
      
      int num = x ;
      int reversed = 0;
      
      while(num){
            reversed = reversed * 10 + num%10; 
            
            num /= 10;
      }
      return (reversed == x) ? true:false;
            

 }

 //avoid overflow . Max int. = 2,147,483,647 if we reverse it is bigger

  bool isPalindromeOF(int x) {

      if ((x<  0 ) || (x%10 == 0 && x)) //e.g 40 never 
            return false;
      
      int reversed = 0;
      
      while (x > reversed){ //finding half since we are chopping x  
            reversed =  reversed*10 +  x%10;
            x/=10;
      }

      return x == reversed || x == (reversed/10) ;
 }