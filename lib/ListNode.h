#ifndef LISTNODE_H
#define LISTNODE_H

// ListNode class definition
class ListNode {
public:
    int val;
    ListNode* next;

    // Default constructor for ListNode
    ListNode() : val(0), next(nullptr) {}

    // Constructor for ListNode with a value
    ListNode(int x) : val(x), next(nullptr) {}

    // Constructor for ListNode with a value and a next pointer
    ListNode(int x, ListNode* next) : val(x), next(next) {}
};

#endif // LISTNODE_H