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
#include <iostream>
#include <vector>

#include "lib/TreeNode.h"
#include "lib/ListNode.h"
#include "lib/Trie.h"
#include "lib/DisJointSet.h"
#include "lib/LinkedList.h"


///Sieve of Eratosthenes (per prime)
//non-primes like i = 4, 6, 8.
//non-primes like i = 9, 27.
vector<bool> findPrimeNumbers(int n){

    vector<bool> isPrime(n + 1, true); //mark all primes n+1 to include n

    isPrime[0] = isPrime[1] = false;
     
    for (int i = 2; i * i <= n; i ++  ){
        if (isPrime[i])
            for (int j = i*i ; j <=n; j+=i) //increments by i
                isPrime[j] = false;
    }

    return isPrime;
}




