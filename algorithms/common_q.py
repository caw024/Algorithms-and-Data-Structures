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


# invert a btree
def invert_btree(root):
    if root is None:
        return
    root.left, root.right = invert_btree(root.right), invert_btree(root.left)
    return root

from collections import deque

# get level order traversal of tree
def level_order_traversal(root):
    ans = []
    queue = deque()
    queue.append((root, 0))
    while queue:
        prev, depth = queue.popleft()
        if prev is None:
            continue
        if len(ans) <= depth:
            ans.append([])
        ans[depth].append(prev.val)
        queue.append((prev.left, depth+1))
        queue.append((prev.right, depth+1))
    return ans
