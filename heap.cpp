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


vector<vector<int>> kClosest(vector<vector<int>>& points, int k) {
    
    vector<vector<int>> results;

    priority_queue<pair<long long,int>> maxHeap; // the compare is by first and second item

    for (int i =0; i < points.size() ; i++){
        long long x = points[i][0];
        long long y = points[i][1];

        maxHeap.push({x*x + y*y, i});  // push and if we have extra pop
        if (maxHeap.size() > k ) 
            maxHeap.pop();         
    }

    while (!maxHeap.empty()){

        results.push_back(points[maxHeap.top().second]);
        maxHeap.pop(); 
    }
    
    return results;
}

// complexity nlog(k).  n points size 

 