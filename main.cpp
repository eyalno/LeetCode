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
#include "lib/binarySearchTree.h"

using namespace std;





int romanToInt(string s)
{
      unordered_map<char, int> map;

      map['I'] = 1;
      map['V'] = 5;
      map['X'] = 10;
      map['L'] = 50;
      map['C'] = 100;
      map['D'] = 500;
      map['M'] = 1000;

      /*
      map["I"] =1;
      map["V"] =5;
      map["X"] =10;
      map["L"] =50;
      map["C"] =100;
      map["D"] =500;
      map["M"] =1000;
      map["IV"] =4;


      map["IX"] =9;
      map["XL"] =40;
      map["XC"] =90;
      map["CD"] =400;
      map["CM"] =900;
      */
      // right to left

      int i = s.size() - 1;
      int sum = 0;

      while (i >= 0)
      {
            int lastNum = 0;
            char ch = s[i];

            int num = map[ch];

            if (num < lastNum)
                  sum -= num;
            else
                  sum += num;

            lastNum = num;
            i--;
      }
      return sum;

      /*left to right


      int i =0;
      int sum =0;

      while ( i < s.size()){
            if (i < s.size() -1){
                  string doubleSymbol = s.substr(i,2);

                  if (map.count(doubleSymbol) > 0){
                        i+=2;
                        sum += map[doubleSymbol];
                        continue;
                  }
            }

            string singleSymbol = s.substr(i,1);
            i+=1;
            sum+= map[singleSymbol];
      }



      return sum;
      */
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


int majorityElement(vector<int>& nums)
{

      // bit manipulation
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


//278. First Bad Version

// Mock global variable representing the first bad version
int firstBad = 4; // Example: Assume version 4 is the first bad version

// Implementation of the isBadVersion API
bool isBadVersion(int version) {
      return version >= firstBad;
}

int firstBadVersion(int n) {

      int i = 1;
      int j = n;

      while (i < j) {

            int mid = i + (j - i) / 2;
            if (isBadVersion(mid)) {
                  j = mid;
            }
            else
                  i = mid + 1;
      }
      return i;
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





class MyLinkedList
{

private:
      Node* tail;

public:
      int size;

      Node* head;
      MyLinkedList() : size(0)
      {
            head = new Node();
            tail = new Node();
            head->next = tail;
            tail->prev = head;
      }

      int get(int index)
      {

            if (index < 0 || index >= size)
                  return -1;

            int mid = size / 2;
            Node* curr;
            if (index < mid)
            {

                  curr = head->next;

                  for (int i = 0; i < index; i++)
                  {
                        curr = curr->next;
                  }
            }
            else
            {
                  curr = tail->prev;
                  for (int i = 0; i < size - index - 1; i++)
                        curr = curr->prev;
            }

            return curr->val;
      }




      class MyHashMap
      {

      private:
#define primeBase 769
            array<LinkedList*, primeBase> arr;

      public:
            MyHashMap()
            {

                  for (int i = 0; i < primeBase; i++)
                        arr[i] = new LinkedList;
            }

            void put(int key, int value)
            {
                  int mappedVal = key % primeBase;

                  LinkedList* bucket = arr[mappedVal];
                  bucket->insertFront(key);
            }

            void remove(int key)
            {
                  int mappedVal = key % primeBase;
                  LinkedList* bucket = arr[mappedVal];
                  bucket->remove(key);
            }

            bool contains(int key)
            {
                  int mappedVal = key % primeBase;
                  LinkedList* bucket = arr[mappedVal];
                  if (bucket->search(key))
                        return true;

                  return false;
            }
      };

      class MyHashSet
      {

      private:
#define primeBase 769

            array<BST*, primeBase> arr;

      public:
            MyHashSet()
            {

                  for (int i = 0; i < primeBase; i++)
                        arr[i] = new BST;
            }

            void add(int key)
            {
                  int mappedVal = key % primeBase;

                  BST* bucket = arr[mappedVal];
                  bucket->insert(key);
            }

            void remove(int key)
            {
                  int mappedVal = key % primeBase;
                  BST* bucket = arr[mappedVal];
                  bucket->deleteNode(key);
            }

            bool contains(int key)
            {
                  int mappedVal = key % primeBase;
                  BST* bucket = arr[mappedVal];
                  if (bucket->search(key))
                        return true;

                  return false;
            }
      };


      

      ListNode* getIntersectionNode(ListNode* headA, ListNode* headB)
      {

            // two pointers
            ListNode* ptr1 = headA;
            ListNode* ptr2 = headB;

            while (ptr1 != ptr2)
            {
                  ptr1 = ptr1 == nullptr ? headB : ptr1->next;
                  ptr2 = ptr2 == nullptr ? headA : ptr2->next;
            }

            return ptr1;

            // hash set
            /*
            unordered_set<ListNode* > set;

            ListNode* curr = headA;

            while(curr != nullptr){

                  set.insert(curr);

                  curr = curr->next;
            }

            curr = headB;

            while (curr != nullptr){

                  if(set.find(curr)!= set.end())
                        return curr;

                  curr = curr->next;
            }
            return nullptr;

                */
      }

      ListNode* detectCycle(ListNode* head)
      {

            /*
            unordered_set<ListNode *> set;

            ListNode * curr = head;

            while (curr != NULL){

                  if (set.find(curr) != set.end())
                        return curr;

                  set.insert(curr);

                  curr = curr->next;
            }

            return NULL;
            */

            // Floyd's Tortoise and Hare

            ListNode* slow = head;
            ListNode* fast = head;

            while (fast != NULL && fast->next != NULL)
            {

                  slow = slow->next;
                  fast = fast->next->next;
                  if (slow == fast)
                        break;
            }

            if (fast == NULL || fast->next == NULL)
                  return NULL;

            slow = head;

            while (slow != fast)
            {
                  slow = slow->next;
                  fast = fast->next;
            }

            return slow;
      }

      string reverseWords(string s)
      {

            //  hello   world

            // in place with indices

            reverse(s.begin(), s.end());

            int n = s.size();
            int idx = 0; // where we copy slower than start due to zeros

            for (int start = 0; start < n; start++)
            {

                  if (s[start] != ' ')
                  {

                        // add space ignore begining
                        if (idx != 0)
                              s[idx++] = ' ';

                        // word
                        int end = start;
                        while (s[end] != ' ' && end < n)
                              s[idx++] = s[end++];

                        reverse(s.begin() + idx - (end - start), s.begin() + idx);

                        // end
                        start = end;
                  }
            }

            s.erase(s.begin() + idx, s.end());
            return s;

            // in place using built functions

            /*
            size_t start = 0;
            size_t end =0 ;

            start =  s.find_first_not_of(" \t",start);

            s.erase(s.begin(),s.begin() + start);
            reverse(s.begin(),s.end());
            start =0;
            start =  s.find_first_not_of(" \t",start);
            s.erase(s.begin(),s.begin() + start);

            start =0;
            end =0 ;

            while (end != string::npos){

                  start =  s.find_first_not_of(" \t",start);
                      if (start == string::npos)
                        break;
                  end = s.find_first_of(" \t",start);
                  if (end == string::npos){

                        end = s.length();
                        reverse(s.begin()+start, s.begin() +end);
                        break;
                  }
                  reverse(s.begin()+start, s.begin() +end);



                  start = end+1;
            }

            start =0;
            end =0 ;

            while (end != string::npos){
                  start = s.find_first_of(" \t",start);
                    if (start == string::npos)
                        break;
                  end = s.find_first_not_of(" \t",start);
                  start++;
                  s.erase(start, end-start  );
            }

            return s;

            */
            // not in place
            /*
            size_t start = 0;
            size_t end =0 ;
            string ret;
            vector<string> vec;

            while (end != string::npos){

                  start =  s.find_first_not_of(" \t",start);

                  if (start == string::npos)
                        break;

                  end = s.find_first_of(" \t",start);

                  string word = s.substr(start, end - start );
                  start = end+1;

                  vec.push_back(word);
            }

            reverse(vec.begin(),vec.end());

            for (string w: vec){

                  ret +=( w + ' ');
            }

            ret.pop_back();

            return ret;

            */
      }

      int key(int i, int j)
      {
            size_t hash_i = hash<int>{}(i), hash_j = hash<int>{}(j);
            int hashed = (int)(hash_i ^ (hash_i >> 32));
            return (hashed << 5) - 1 + (int)(hash_j ^ (hash_j >> 32));
      }

      //(n choose k)
      int binomialCoefficient(int n, int k)
      {

            int rowCol = key(n, k);
            static unordered_map<int, int> cache;

            if (cache.count(rowCol) > 0)
                  return cache[rowCol];

            if (k == 0 || n == 0 || n == k)
                  return cache[rowCol] = 1;

            return cache[rowCol] = (binomialCoefficient(n - 1, k - 1) + binomialCoefficient(n - 1, k));
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


      void addAtIndex(int index, int val)
      {

            if (index < 0 || index > size)
                  return;

            Node* n = new Node(val);

            Node* pred;

            int mid = size / 2;
            Node* curr;
            if (index < mid)
            {
                  pred = head;
                  for (int i = 0; i < index; i++)
                        pred = pred->next;

                  n->next = pred->next;
                  n->next->prev = n;
                  pred->next = n;
                  n->prev = pred;
            }
            else
            {
                  pred = tail;
                  for (int i = 0; i < size - index; i++)
                        pred = pred->prev;

                  n->prev = pred->prev;
                  n->prev->next = n;
                  pred->prev = n;
                  n->next = pred;
            }
            size++;
      }


};

string longestCommonPrefix(vector<string>& strs, int l, int r)
{

      // divide and conquer
      if (l == r)
            return strs[l];

      int mid = l + (r - l) / 2;

      string leftStr = longestCommonPrefix(strs, l, mid);
      string rightStr = longestCommonPrefix(strs, mid + 1, r);

      int minLen = min(leftStr.length(), rightStr.length());

      for (int i = 0; i < minLen; i++)
      {

            if (leftStr[i] != rightStr[i])
                  return leftStr.substr(0, i);
      }

      return leftStr.substr(0, minLen);
}

bool isCommonPrefix(vector<string>& strs, int len)
{

      string prefix = strs[0].substr(0, len + 1);

      for (int j = 1; j < strs.size(); j++)
      {

            if (prefix != strs[j].substr(0, len + 1))
                  return false;
      }

      return true;
}
string longestCommonPrefix(vector<string>& strs)
{

      // using binary search.

      if (strs.empty())
            return "";

      int minLen = INT_MAX;
      for (string str : strs)
            minLen = min((int)str.length(), minLen);

      int low = 0;
      int high = strs.size() - 1;

      int mid = 0;

      while (low <= high)
      {

            int mid = low + (high - low) / 2;

            if (isCommonPrefix(strs, mid))
                  low = mid + 1;
            else
                  high = mid - 1;
      }

      return strs[0].substr(0, mid + 1);

      string ret;
      // flower","flow","flight"

      // 1 <= strs.length <= 200
      // 0 <= strs[i].length <= 200

      // Using Trie

      Trie t;

      for (string str : strs)
            t.insert(str);

      TrieNode* current = t.root;

      while (current->children.size() == 1)
      {
            ret += current->children.begin()->first;
            current = current->children.begin()->second;
      }

      return ret;

      /*
      //Binary search
      size_t minLen = numeric_limits<size_t>::max() ;

      for (int i =0; i < strs.size(); i++)
            minLen = min(minLen,strs[i].length());


      int low = 01;
      int high = minLen;

      while (low <= high){
            int mid = (low + high) / 2;

            if (isCommonPrefix(strs,mid))
                  low = mid + 1 ;
            else
                  high = mid - 1;
      }

      ret = strs[0].substr(0,(low+high)/2);


      //Recursion Divide n Conquer

      /// ret = longestCommonPrefix(strs,0,strs.size()-1);


      */

      // Horizantal scanning compare 1 string to all and remove characters
      /*
      string lcp = strs[0];

      if (lcp.empty())
            return lcp;

      // flower","flow","flight"
      for (int i = 1; i < strs.size(); i++){

            string nStr = strs[i];


            while ( nStr.find(lcp) != 0   ){
                  lcp.erase(lcp.end() - 1);

                  if (lcp.empty())
                        return lcp;
            }
      }

      return lcp;
      */

      // Vertical scanning  - compare charters instead of ull strings
      /*
      string ret = "";

      for (int i =0; i <strs[0].length(); i++){

            for (int j =1; j < strs.size(); j++  ){

                  if (strs[0][i] != strs[j][i] || i == strs[j].length())
                        return strs[0].substr(0, i);


            }
      }

      return strs[0];
      */

      /*. Brute force
      int minLen = 201;

      string ret = "";

      for (int i = 0; i < strs.size(); i ++){

            if (strs[i].length() < minLen)
                  minLen = strs[i].length();
      }

      if (minLen == 201 || minLen == 0   )
            return ret;

      for (int j = 0; j<minLen ; j++ ){

            char ch = strs[0][j];
            int i = 0;
            for (   ;i < strs.size(); i++ ){

                  if (ch != strs[i][j])
                        break;

            }

            if (i != strs.size() )
                  break;

            ret.push_back(ch);



      }

      return ret;
      */
}

string addBinary(string a, string b)
{

      // add bit bit and push to string
      int maxL = max(a.length(), b.length());
      // padding zeros

      a = string(maxL - a.length(), '0') + a;
      b = string(maxL - b.length(), '0') + b;
      string ret;

      int carry = 0;

      for (int i = 0; i < maxL; i++)
      {

            int A = static_cast<int>(a[maxL - i - 1] - '0');
            int B = static_cast<int>(b[maxL - i - 1] - '0');

            int result = A + B + carry;

            if (result == 3)
            {
                  carry = 1;
                  ret.push_back('1');
            }
            else if (result == 2)
            {
                  ret.push_back('0');
                  carry = 1;
            }
            else if (result == 1)
            {
                  ret.push_back('1');
                  carry = 0;
            }
            else if (result == 0)
            {
                  ret.push_back('0');
                  carry = 0;
            }
      }

      if (carry)
            ret.push_back('1');
      reverse(ret.begin(), ret.end());
      return ret;

      // 1111
      // 0010
      /*
      int maxL = max(a.length(),b.length());

      a = string(maxL - a.length() ,'0') + a;
      b = string(maxL - b.length() ,'0') + b;

      int A = 0;
      int B = 0;

      for ( int i = 0;  i < maxL; i++){

            A = (A << 1) + a[i] - '0';
            B = (B << 1) + b[i] - '0';
      }

      int carry = A & B;

      A^=B;

      while (carry != 0 ){

            B = carry << 1;
            carry = A & B;
            A^=B;
      }


            string ret;
      while (A) {

            ret = static_cast<char>(A%2 + '0') + ret;

            A/=2;
      }

      string zero("0");
      return ret.empty()? zero : ret;

      */

      /*
      bitset<32> binaryNumber1(stoi(a));
      bitset<32> binaryNumber2(stoi(b));

      bitset<32> sum = binaryNumber1. binaryNumber2;
      return sum.to_string();

      */
      /*
      int maxLength = max(a.length(),b.length());


      string paddedA(maxLength-a.length(),'0');
      string paddedB(maxLength-b.length(),'0');

      a = paddedA + a;
      b = paddedB + b;

      string ret;

      int carry = 0;

      for (int i = maxLength-1 ; i >= 0; i--){


            int bitA = a[i] - '0';
            int bitB = b[i] - '0';

            int sum = carry + bitA + bitB;

            ret.push_back(sum%2 + '0');
            carry =sum/2;

      }

      if (carry)
            ret.push_back('1');

      reverse(ret.begin(),ret.end());
      return ret;
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

int maxProfit(vector<int>& prices)
{

      int size = prices.size();
      int min = pow(10, 4);

      int profit = 0;

      for (int i = 0; i < size; i++)
      {

            if (prices[i] < min)
                  min = prices[i];
            else if ((prices[i] - min > profit))
            {
                  profit = prices[i] - min;
            }
      }

      return profit;
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



int main()
{
      return 0;
}
