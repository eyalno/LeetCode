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

switch (1){

case 1:{  // use running mask

  int mask = 1;
  int count = 0;

  for (int i =0; i < 32; i++ ){
        
        if ((n & mask) == mask)
              count++;

        mask <<= 1;  

  } 
  return count ;

}
case 2:{ //removeing lsb
    int bitCount = 0;
  
  while (n !=0){    
    n&= n-1; 
    bitCount++;
        
  }  
  return bitCount;
}
}

return 0;
}




vector<int> countBits(int n) {

vector<int> result;
switch (1){

case 1:{ //popcount

  for (int i= 0; i<= n ; i++){

        int bitCount = 0;
        int num = i;

        while (num !=0){    
              num&= num-1;
              bitCount++;            
        }  
        result.push_back(bitCount);
  }
}
case 2:{ //brute force

  for (int i= 0; i<= n ; i++){

        int bitCount = 0;
        int num = i;

        while (num !=0){    
              bitCount+=( num&1 );
              num = num >> 1;
        }  
        result.push_back(bitCount);
  }
  }
  return result;
}
}