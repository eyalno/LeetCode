#ifndef LINKEDLIST_H
#define LINKEDLIST_H

#include <iostream>

using namespace std;

// Node class definition
class Node {
public:
    Node() : val(0), next(NULL), prev(NULL), child(NULL), random(NULL), left(NULL), right(NULL) {}

    Node(int num) : val(num), next(NULL), prev(NULL), child(NULL), random(NULL), left(NULL), right(NULL) {}

    int val;
    int data;
    Node* left;
    Node* right;
    Node* next;
    Node* prev;
    Node* child;
    Node* random;
};

// LinkedList class definition
class LinkedList {
private:
    Node* head;

public:
    // Constructor for LinkedList
    LinkedList() : head(nullptr) {}

    // Destructor for LinkedList
    ~LinkedList() {
        while (head) {
            Node* temp = head;
            head = head->next;
            delete temp;
        }
    }

    // Inserts a new node with data at the front of the list
    void insertFront(int data) {
        Node* newNode = new Node(data);
        newNode->next = head;
        head = newNode;
    }

    // Removes the first occurrence of the node with the given data
    void remove(int data) {
        if (!head)
            return;

        // If the node to be removed is the head
        if (head->data == data) {
            Node* temp = head;
            head = head->next;
            delete temp;
            return;
        }

        Node* prev = head;
        Node* current = head->next;

        while (current) {
            if (current->data == data) {
                prev->next = current->next;
                delete current;
                return;
            }
            prev = current;
            current = current->next;
        }
    }

    // Searches for a node with the given data in the list
    bool search(int data) const {
        Node* current = head;
        while (current) {
            if (current->data == data)
                return true;
            current = current->next;
        }
        return false;
    }
};

#endif // LINKEDLIST_H