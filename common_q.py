# reverse a linked list iteratively: O(n)
def reverse_linked_list(head):
    # A -> B -> C -> D -> None
    if head == None or head.next == None:
        return head

    # head points to A
    # tail points to D
    tail = head
    while tail.next != None:
        tail = tail.next

    while head.next != None:
        cur = head
        # C -> D -> None
        while cur.next.next != None:
            cur = cur.next
        # D points to C
        cur.next.next = cur
        # C to points to None
        cur.next = None
    return tail
    
# reverse a linked list recursively: O(N)
def reverse_linked_list_rec(head):
    if head == None or head.next == None:
        return head
    new_head = reverse_linked_list_rec(head.next)
    head.next = None
    return (new_head, new_tail.next)

# invert a btree
def invert_btree(node):
    pass
