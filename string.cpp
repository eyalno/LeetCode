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


//409. Longest Palindrome
int longestPalindrome(string s)
{
      switch (1) {

      case 1: {

            vector<int> lettersCount(52, 0);

            for (char ch : s)
            {
                  if (isupper(ch))
                        lettersCount[26 + ch - 'A']++;
                  else
                        lettersCount[ch - 'a']++;
            }

            int mid = 0;
            int length = 0;
            for (int sum : lettersCount)
            {

                  if ((sum % 2) == 0)
                        length += sum;
                  else
                  {
                        mid = 1;
                        length += sum - 1;
                  }
            }

            return length + mid;
      }
      case 2: {
            //hash map
            unordered_map<char, int> map;

            int even = 0, odd = 0;

            for (char ch : s) {
                  map[ch]++;
            }


            for (auto it = map.begin(); it != map.end(); it++) {

                  if ((it->second % 2) == 0)
                        even += it->second;
                  else {
                        even += it->second - 1;
                        odd++;
                  }

            }

            return even + (odd > 0 ? 1 : 0);

      }
      }
}


string processString(string& s)
{

      string result;

      for (int i = 0; i < s.size(); i++)
      {
            if (s[i] == '#' && (!result.empty()))
                  result.pop_back();
            else
                  result.push_back(s[i]);
      }
      return result;
}

//844. Backspace String Compare
bool backspaceCompare(string s, string t)
{

      return processString(s) == processString(t);
}



//28. Find the Index of the First Occurrence in a String
int strStr(string haystack, string needle)
{

      int hL = haystack.length();
      int nL = needle.length();

      int window = 0;
      for (; window < hL - nL + 1; window++)
      {

            int j = 0;

            while (haystack[window + j] == needle[j] && j < nL)
            {
                  j++;
            }

            if (j == nL)
                  break;
            j = 0;
      }
      if (window == hL - nL + 1 || hL < nL)
            window = -1;

      return window;
}


//387. First Unique Character in a String
int firstUniqChar(string s)
{

      unordered_map<char, int> map;
      // s ="loveleetcode";
      int ret = -1;

      for (char ch : s)
      {
            map[ch]++;
      }

      for (int i = 0; i < s.size(); i++)
      {

            if (map[s[i]] == 1)
                  return i;
      }

      return ret;
}

//383. Ransom Note
bool canConstruct(string ransomNote, string magazine)
{
      if (ransomNote.length() > magazine.length())
            return false;

      unordered_map<char, int> map;

      for (char ch : magazine)
      {
            map[ch]++;
      }

      for (char ch : ransomNote)
      {

            if (map[ch] > 0)
                  map[ch]--;
            else
                  return false;
      }
      return true;
}


//242. Valid Anagram
bool isAnagram(string s, string t)
{
switch (1){ //map

case 1:{
      unordered_map<char, int> counter;

      if (s.length() != t.length())
            return false;

      for (char ch : s)
            counter[ch]++;

      for (char ch : t)
      {
            if (counter.find(ch) == counter.end() || counter[ch] == 0)
                  return false;
            counter[ch]--;
      }

      for (auto& pair : counter)
            if (pair.second != 0)
                  return false;

      return true;
}
case 2:{// sort
      
            if (s.length() != t.length())
            return false;

            sort(s.begin(),s.end());
            sort(t.begin(),t.end());

            return s == t;
}
}
}


bool isPalindrome(string s)
{
      // s ="A man, a plan, a canal: Panama";

      string filteredString;

      for (char ch : s)
            if (isalnum(ch))
                  filteredString.push_back(tolower(ch));

      string reversedString(filteredString);

      reverse(reversedString.begin(), reversedString.end());

      return filteredString.compare(reversedString);
}

// paranthesis
bool isValid(string s)
{

      if (s.length() % 2 == 1)
            return false;

      stack<char> stack;

      map<char, char> map;
      map['('] = ')';
      map['{'] = '}';
      map['['] = ']';

      for (char ch : s)
      {
            if (ch == '(' || ch == '[' || ch == '{')
                  stack.push(ch);

            if (ch == ')' || ch == ']' || ch == '}')
            {
                  if (stack.empty())
                        return false;

                  if (map[stack.top()] != ch)
                        return false;
                  stack.pop();
            }
      }

      if (!stack.empty())
            return false;

      return true;
}


vector<string> findRestaurant(vector<string>& list1, vector<string>& list2)
{

      map<int, vector<string>> map;

      vector<string>& smallerList = list1.size() > list2.size() ? list2 : list1;

      vector<string>& greaterList = list1.size() > list2.size() ? list1 : list2;

      // vector<string> * smasfsdlerList = list1.size() > list2.size() ? &list2:&list1;

      for (int i = 0; i < smallerList.size(); i++)
      {

            auto it = find(greaterList.begin(), greaterList.end(), smallerList[i]);
            if (it != greaterList.end())
            {
                  int j = distance(greaterList.begin(), it);

                  map[i + j].push_back(smallerList[i]);
            }
      }

      return (map.begin()->second);
}

bool isIsomorphic(string s, string t)
{

      unordered_map<char, char> mapStoT;
      unordered_map<char, char> mapTtoS;
      // badc
      // baba
      // paper
      // title
      if (s.length() != t.length())
            return false;

      for (int i = 0; i < s.length(); i++)
      {

            char chS = s[i];
            char chT = t[i];

            int countS = mapStoT.count(chS);
            int countT = mapTtoS.count(chT);

            if (countS == 0)
                  mapStoT[chS] = t[i];

            if (countT == 0)
                  mapTtoS[chT] = s[i];

            if (mapStoT[chS] != chT || mapTtoS[chT] != chS)
                  return false;
      }

      return true;
}


vector<vector<string>> groupAnagrams(vector<string>& strs)
{

      vector<vector<string>> ret;
      // strs = {"eat","tea","tan","ate","nat","bat"};
      array<int, 26> arr{};
      unordered_map<string, vector<string>> map;

      for (string word : strs)
      {

            for (char ch : word)
            {
                  arr[ch - 'a']++;
            }
            string key;
            for (int i : arr)
            {
                  key.push_back(i + '0');
                  key.push_back('#');
            }

            map[key].push_back(word);
            fill(arr.begin(), arr.end(), 0);
            key.clear();
      }

      for (auto it = map.begin(); it != map.end(); it++)
      {
            ret.push_back(it->second);
      }

      return ret;

      /*sorting
      vector<vector<string>>  ret;
      unordered_map<string,vector<string>> map;
      strs = {"eat","tea","tan","ate","nat","bat"};
      for (string word: strs){

            string sortedWord = word;
            sort(sortedWord.begin(),sortedWord.end());
            map[sortedWord].push_back(word);
      }

       for ( auto  it =  map.begin() ; it != map.end(); it++  ){
            ret.push_back(it->second);
       }


      return ret;
      */
}