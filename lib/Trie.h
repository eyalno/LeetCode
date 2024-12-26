#ifndef TRIE_H
#define TRIE_H

#include <string>
#include <unordered_map>

using namespace std;

// TrieNode class definition
class TrieNode {
public:
    bool isEndOfWord;
    unordered_map<char, TrieNode*> children;

    // Constructor for TrieNode
    TrieNode() : isEndOfWord(false) {}
};

// Trie class definition
class Trie {

public:
    TrieNode* root;
    // Constructor for Trie
    Trie() : root(new TrieNode()) {}

    // Inserts a word into the trie
    void insert(string word) {
        TrieNode* current = root;
        for (char ch : word) {
            if (current->children.find(ch) == current->children.end()) {
                current->children[ch] = new TrieNode();
            }
            current = current->children[ch];
        }
        current->isEndOfWord = true;
    }

    // Searches for a word in the trie
    bool search(string word) {
        TrieNode* current = root;
        for (char ch : word) {
            if (current->children.find(ch) == current->children.end()) {
                return false;
            }
            current = current->children[ch];
        }
        return current->isEndOfWord;
    }

    // Checks if any word in the trie starts with the given prefix
    bool startsWith(string prefix) {
        TrieNode* current = root;
        for (char ch : prefix) {
            if (current->children.find(ch) == current->children.end()) {
                return false;
            }
            current = current->children[ch];
        }
        return true;
    }
};

#endif // TRIE_H