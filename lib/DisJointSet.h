#ifndef DISJOINTSET_H
#define DISJOINTSET_H

#include <vector>
using namespace std;

class DisJointSet {
private:
    vector<int> parent;
    vector<int> root;
    vector<int> rank; //rank is 0 in the begining  

public:
    // Constructor initializes parent and root with values from 0 to n-1
    DisJointSet(int n) : parent(n), root(n), rank(n, 0) {
        for (int i = 0; i < n; i++) {
            parent[i] = i;
            root[i] = i;
        }
    }

    // Finds the root of the element x (O(n) in worst case)
    int find(int x) {
        if (parent[x] == x)
            return x;
        return find(parent[x]);
    }

    // Unites the sets containing a and b (O(n) in worst case)
    void unionSets(int a, int b) {
        int rootA = find(a);
        int rootB = find(b);

        if (rootA != rootB)
            parent[rootB] = rootA;
    }

    // Checks if a and b are in the same set
    bool isConnected(int a, int b) {
        return find(a) == find(b);
    }

    // Quick Find: returns the root of x (O(1) time)
    int quickFind(int x) {
        return root[x];
    }

    // Quick Find Union: unites the sets containing a and b (O(n) time)
    void quickFindUnionSets(int a, int b) {
        int rootA = quickFind(a);
        int rootB = quickFind(b);

        if (rootA != rootB) {
            for (int i = 0; i < root.size(); ++i) {
                if (root[i] == rootB)
                    root[i] = rootA;
            }
        }
    }

    // Checks if a and b are connected using Quick Find
    bool isQuickConnected(int a, int b) {
        return quickFind(a) == quickFind(b);
    }

    // Quick Union: finds the root of x (O(1) time)
    int quickUnionFind(int x) {
        while (root[x] != x)
            x = root[x];
        return x;
    }

    // Quick Union: unites the sets containing a and b (O(1) time)
    void quickUnionSets(int a, int b) {
        int rootA = find(a); // can be O(n)
        int rootB = find(b);

        if (rootA != rootB)
            parent[rootB] = rootA;
    }

    // Union by Rank: unites the sets containing a and b (O(1) time)
    bool unionbyRank(int a, int b) {
        int rootA = findPathCompression(a); // can be O(log n)
        int rootB = findPathCompression(b);

        if (rootA == rootB)
            return false;

        if (rootA != rootB) {
            if (rank[rootA] > rank[rootB])
                root[rootB] = rootA;
            else if (rank[rootB] > rank[rootA])
                root[rootA] = rootB;
            else {
                rank[rootA]++;
                root[rootB] = rootA;
            }
        }
        return true;
    }

    // Path compression: optimizes the find operation by making nodes point directly to the root
    int findPathCompression(int x) {
        if (x == root[x])
            return x;

        return root[x] = findPathCompression(root[x]);
    }
};

#endif // DISJOINTSET_H