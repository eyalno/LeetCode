#ifndef TRIE_H
#define TRIE_H

#include <string>
#include <unordered_map>

using namespace std;

// TrieNode class definition
class TrieNode {
public:
    bool isEndOfWord; //represents end of search
    unordered_map<char, TrieNode*> children; //represents the graph node with 
    //char as key ever char is a child with many children 

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






class TrieNode {
public:
    bool isWord;
    TrieNode* children[26];

    TrieNode() {
        isWord = false; // at every node indicates if its a word.
        memset(children, 0, sizeof(children));  //setting all pointers to 0 
    }
};

class Trie {
private:
    TrieNode* root;
public:
    Trie() {
        root = new TrieNode();
    }
    
    void insert(string word) { //insert new word
        TrieNode* node = root;

        for (char c : word) { // looping on word.
            int index = c - 'a';

            if (!node->children[index]) {
                node->children[index] = new TrieNode(); // creates the new character 
            }
            node = node->children[index]; //setting the new node to be current
        }
        node->isWord = true; // setting the word on last node
    }

    bool search(string word) {// Search full word
        TrieNode* node = root;

        for (char c : word) { //same looping logic now just comparing 
            int index = c - 'a';

            if (!node->children[index]) {
                return false;
            }
            node = node->children[index];
        }
        return node->isWord;
    }
    bool startsWith(string prefix) {// Prefix search same loop as search without checking word
        TrieNode* node = root;

        for (char c : prefix) { 
            int index = c - 'a';

            if (!node->children[index]) {
                return false;
            }
            node = node->children[index];
        }
        return true;
    }
};