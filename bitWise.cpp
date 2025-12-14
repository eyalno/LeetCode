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

using namespace std;

int hammingWeight(int n) {

      switch (1) {

      case 1: {  // use running mask

            int mask = 1;
            int count = 0;

            for (int i = 0; i < 32; i++) {

                  if ((n & mask) == mask)
                        count++;

                  mask <<= 1;

            }
            return count;

      }
      case 2: { //removeing lsb
            int bitCount = 0;

            while (n != 0) {
                  n &= n - 1;
                  bitCount++;

            }
            return bitCount;
      }
      }

      return 0;
}

vector<int> countBits(int n) {

      vector<int> result;
      switch (1) {

      case 1: { //popcount

            for (int i = 0; i <= n; i++) {

                  int bitCount = 0;
                  int num = i;

                  while (num != 0) {
                        num &= num - 1;
                        bitCount++;
                  }
                  result.push_back(bitCount);
            }
      }
      case 2: { //brute force

            for (int i = 0; i <= n; i++) {

                  int bitCount = 0;
                  int num = i;

                  while (num != 0) {
                        bitCount += (num & 1);
                        num = num >> 1;
                  }
                  result.push_back(bitCount);
            }
      }
            return result;
      }
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


int reverseBits(int n) {
 
      
      for (int i =0 ; i< 16; i ++){
            


      }



}