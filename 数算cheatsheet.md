# 数算cheatsheet

## 1 二分

```python
lo, hi = 0, L+1
ans = -1
while lo < hi:
    mid = (lo + hi) // 2
    if check(mid):
        hi = mid
    else:               # 返回False，有可能是num==m
        ans = mid       # 如果num==m, mid就是答案
        lo = mid + 1
print(ans)
```

## 2 排序

### 1 冒泡排序

```python
def bubbleSort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if (swapped == False):
            break
```

### 2 选择排序

先排小的

```python
for i in range(len(A)):
    min_idx = i
    for j in range(i + 1, len(A)):
        if A[min_idx] > A[j]:
            min_idx = j
    A[i], A[min_idx] = A[min_idx], A[i]
print(' '.join(map(str, A)))
```

先排大的

```python
def selectionSort(alist):
    for fillslot in range(len(alist)-1, 0, -1):
        positionOfMax = 0
        for location in range(1, fillslot+1):
            if alist[location] > alist[positionOfMax]:
                positionOfMax = location
        if positionOfMax != fillslot:
            alist[fillslot], alist[positionOfMax] = alist[positionOfMax], alist[fillslot]
```

### 3 快速排序

```python
def quicksort(arr, left, right):
    if left < right:
        partition_pos = partition(arr, left, right)
        quicksort(arr, left, partition_pos - 1)
        quicksort(arr, partition_pos + 1, right)
def partition(arr, left, right):
    i = left
    j = right - 1
    pivot = arr[right]#这里选择数组的最后一个元素 `arr[right]` 作为基准值；指针 `i` 从左往右移动，指针 `j` 从右往左移动。
    while i <= j:
        while i <= right and arr[i] < pivot:
            i += 1
        while j >= left and arr[j] >= pivot:
            j -= 1
        if i < j:
            arr[i], arr[j] = arr[j], arr[i]
    '''这个循环的含义：
		while i <= j: 是主循环，直到两个指针相遇。
		arr[i] < pivot: 找到一个不小于 pivot 的数，停下来。
		arr[j] >= pivot: 找到一个小于 pivot 的数，停下来。
		如果此时 i < j，说明这两个数在错误的分区中，所以交换它们。'''
    if arr[i] > pivot:
        arr[i], arr[right] = arr[right], arr[i]#主循环结束后，i 指向第一个不小于 pivot 的元素，我们把 pivot 放到这个位置（这样 pivot 左边的都比它小，右边都不小于它）。
    return i
```

`partition` 函数是快速排序（QuickSort）算法中的核心部分。它的作用是**以一个基准值（pivot）为中心，把数组划分成两个部分**：

- 左边部分的元素都小于基准值；
- 右边部分的元素都大于等于基准值（或者根据比较方式变化）。

### 4 归并排序（merge sort)

```python
def mergeSort(arr):
	if len(arr) > 1:
		mid = len(arr)//2
		L = arr[:mid]	# Dividing the array elements
		R = arr[mid:] # Into 2 halves
		mergeSort(L) # Sorting the first half
		mergeSort(R) # Sorting the second half
		i = j = k = 0
		# Copy data to temp arrays L[] and R[]
		while i < len(L) and j < len(R):
			if L[i] <= R[j]:
				arr[k] = L[i]
				i += 1
			else:
				arr[k] = R[j]
				j += 1
			k += 1
		# Checking if any element was left
		while i < len(L):
			arr[k] = L[i]
			i += 1
			k += 1
		while j < len(R):
			arr[k] = R[j]
			j += 1
			k += 1
```

### 5 希尔排序

```python
def shellSort(arr, n):
    gap = n // 2
    while gap > 0:
        j = gap
        while j < n:
            i = j - gap 
            while i >= 0:
                if arr[i + gap] > arr[i]:
                    break
                else:
                    arr[i + gap], arr[i] = arr[i], arr[i + gap]
                i = i - gap
            j += 1
        gap = gap // 2
```

## 3 链表

### 1 单向链表

```python
class Node:
    def __init__(self, value):
        self.value = value
        self.next = None
class LinkedList:
    def __init__(self):
        self.head = None
    def insert(self, value):
        new_node = Node(value)
        if self.head is None:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
    def delete(self, value):
        if self.head is None:
            return
        if self.head.value == value:
            self.head = self.head.next
        else:
            current = self.head
            while current.next:
                if current.next.value == value:
                    current.next = current.next.next
                    break
                current = current.next
    def display(self):
        current = self.head
        while current:
            print(current.value, end=" ")
            current = current.next
        print()
```

```python
class LinkList:
    class Node:
        def __init__(self, data, next=None):
            self.data = data  # Store data
            self.next = next  # Point to the next node
    def __init__(self):
        self.head = None  # Initialize head as None
        self.tail = None  # Initialize tail as None
        self.size = 0  # Initialize size to 0
    def print(self):
        ptr = self.head
        while ptr is not None:
            if ptr != self.head:  # Avoid printing a comma before the first element
                print(',', end='')
            print(ptr.data, end='')
            ptr = ptr.next
        print()  # Move to the next line after printing all elements
    def insert_after(self, p, data):  
        nd = LinkList.Node(data)
        if p is None:  # If p is None, insert at the beginning
            self.pushFront(data)
        else:
            nd.next = p.next
            p.next = nd
            if p == self.tail:  # Update tail if necessary
                self.tail = nd
            self.size += 1
    def delete_after(self, p):  
        if p is None or p.next is None:
            return  # Nothing to delete
        if self.tail is p.next:  # Update tail if necessary
            self.tail = p
        p.next = p.next.next
        self.size -= 1
    def popFront(self):
        if self.head is None:
            raise Exception("Popping front from empty link list.")
        else:
            data = self.head.data
            self.head = self.head.next
            self.size -= 1
            if self.size == 0:
                self.tail = None
            return data
    def pushFront(self, data):
        nd = LinkList.Node(data, self.head)
        self.head = nd
        if self.size == 0:
            self.tail = nd
        self.size += 1
    def pushBack(self, data):
        if self.size == 0:
            self.pushFront(data)
        else:
            self.insert_after(self.tail, data)
    def clear(self):
        self.head = None
        self.tail = None
        self.size = 0
    def __iter__(self):
        self.ptr = self.head
        return self
    def __next__(self):
        if self.ptr is None:
            raise StopIteration()
        else:
            data = self.ptr.data
            self.ptr = self.ptr.next
            return data
```

### 2 双向链表

```python
class Node:
    def __init__(self, data):
        self.data = data  # 节点数据
        self.next = None  # 指向下一个节点
        self.prev = None  # 指向前一个节点
class DoublyLinkedList:
    def __init__(self):
        self.head = None  # 链表头部
        self.tail = None  # 链表尾部
    # 在链表尾部添加节点
    def append(self, data):
        new_node = Node(data)
        if not self.head:  # 如果链表为空
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node
    # 在链表头部添加节点
    def prepend(self, data):
        new_node = Node(data)
        if not self.head:  # 如果链表为空
            self.head = new_node
            self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
    # 删除链表中的指定节点
    def delete(self, node):
        if not self.head:  # 链表为空
            return
        if node == self.head:  # 删除头部节点
            self.head = node.next
            if self.head:  # 如果链表非空
                self.head.prev = None
        elif node == self.tail:  # 删除尾部节点
            self.tail = node.prev
            if self.tail:  # 如果链表非空
                self.tail.next = None
        else:  # 删除中间节点
            node.prev.next = node.next
            node.next.prev = node.prev
        node = None  # 删除节点
    # 打印链表中的所有元素，从头到尾
    def print_list(self):
        current = self.head
        while current:
            print(current.data, end=" <-> ")
            current = current.next
        print("None")
    # 打印链表中的所有元素，从尾到头
    def print_reverse(self):
        current = self.tail
        while current:
            print(current.data, end=" <-> ")
            current = current.prev
        print("None")
```

### 3 循环链表

```python
class CircleLinkList:
    class Node:
        def __init__(self, data, next=None):
            self.data = data
            self.next = next
    def __init__(self):
        self.tail = None  # 尾指针，指向最后一个节点
        self.size = 0  # 链表大小
    def is_empty(self):
        """检查链表是否为空"""
        return self.size == 0
    def pushFront(self, data):
        """在链表头部插入元素"""
        nd = CircleLinkList.Node(data)
        if self.is_empty():
            self.tail = nd
            nd.next = self.tail  # 自己指向自己形成环
        else:
            nd.next = self.tail.next  # 新节点指向当前头节点
            self.tail.next = nd  # 当前尾节点指向新节点
        self.size += 1
    def pushBack(self, data):
        """在链表尾部插入元素"""
        nd = CircleLinkList.Node(data)
        if self.is_empty():
            self.tail = nd
            nd.next = self.tail  # 自己指向自己形成环
        else:
            nd.next = self.tail.next  # 新节点指向当前头节点
            self.tail.next = nd  # 当前尾节点指向新节点
            self.tail = nd  # 更新尾指针
        self.size += 1
    def popFront(self):
        """移除并返回链表头部元素"""
        if self.is_empty():
            return None
        else:
            old_head = self.tail.next
            if self.size == 1:
                self.tail = None  # 如果只有一个元素，更新尾指针为None
            else:
                self.tail.next = old_head.next  # 跳过旧头节点
            self.size -= 1
            return old_head.data
    def popBack(self):
        """移除并返回链表尾部元素"""
        if self.is_empty():
            return None
        elif self.size == 1:
            data = self.tail.data
            self.tail = None
            self.size -= 1
            return data
        else:
            prev = self.tail
            while prev.next != self.tail:  # 找到倒数第二个节点
                prev = prev.next
            data = self.tail.data
            prev.next = self.tail.next  # 跳过尾节点
            self.tail = prev  # 更新尾指针
            self.size -= 1
            return data
    def printList(self):
        """打印链表中的所有元素"""
        if self.is_empty():
            print('Empty!')
        else:
            ptr = self.tail.next
            while True:
                print(ptr.data, end=', ' if ptr != self.tail else '\n')
                if ptr == self.tail:
                    break
                ptr = ptr.next
```

### 4 单链表反转

```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
def reverse_linked_list(head: ListNode) -> ListNode:
    prev = None
    curr = head
    while curr is not None:
        next_node = curr.next  # 暂存当前节点的下一个节点
        curr.next = prev       # 将当前节点的下一个节点指向前一个节点
        prev = curr            # 前一个节点变为当前节点
        curr = next_node       # 当前节点变更为原先的下一个节点
    return prev
```

## 4 栈

### 1 中缀表达式转后缀（调度场算法）

```python
def infixToPostfix(infixexpr):
    prec = {}
    prec["*"] = 3
    prec["/"] = 3
    prec["+"] = 2
    prec["-"] = 2
    prec["("] = 1
    opStack = [] # Stack()
    postfixList = []
    tokenList = infixexpr.split()
    for token in tokenList:
        if token in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" or token in "0123456789":
            postfixList.append(token)
        elif token == '(':
            opStack.append(token)
        elif token == ')':
            topToken = opStack.pop()
            while topToken != '(':
                postfixList.append(topToken)
                topToken = opStack.pop()
        else:
            while opStack and (prec[opStack[-1]] >= prec[token]):
                postfixList.append(opStack.pop())
            opStack.append(token)
    while opStack:
        postfixList.append(opStack.pop())
    return " ".join(postfixList)
```

以下是 Shunting Yard 算法的基本步骤：

1. 初始化运算符栈和输出栈为空。
2. 从左到右遍历中缀表达式的每个符号。
   - 如果是操作数（数字），则将其添加到输出栈。
   - 如果是左括号，则将其推入运算符栈。
   - 如果是运算符：
     - 如果运算符的优先级大于运算符栈顶的运算符，或者运算符栈顶是左括号，则将当前运算符推入运算符栈。
     - 否则，将运算符栈顶的运算符弹出并添加到输出栈中，直到满足上述条件（或者运算符栈为空）。
     - 将当前运算符推入运算符栈。
   - 如果是右括号，则将运算符栈顶的运算符弹出并添加到输出栈中，直到遇到左括号。将左括号弹出但不添加到输出栈中。
3. 如果还有剩余的运算符在运算符栈中，将它们依次弹出并添加到输出栈中。
4. 输出栈中的元素就是转换后的后缀表达式。

## 5 队列

好像没啥可写的

## 6 树

### 1 X序遍历

前序：根节点->左->右

中序：左->根节点->右

后序：左->右->根节点

### 2 二叉树分类

- **满二叉树（Full Binary Tree）**：所有非叶子节点都有两个子节点。
- **完全二叉树（Complete Binary Tree）**：只有最后一层可以不满，并且节点从左到右排列。
- **平衡二叉树（Balanced Binary Tree）**：左右子树的高度差不超过 1，如 AVL 树。
- **二叉搜索树（Binary Search Tree，BST）**：对于任意节点，左子树的所有节点值小于该节点值，右子树的所有节点值大于该节点值。

### 3 一些概念

**高度 Height**：树中所有节点的最大层级称为树的高度，如图1所示树的高度为 2。

对于只有一个节点的树来说，高度为0，深度为0。如果是空树，高度、深度都是 -1.

这是合理的定义方式，但需要⚠️：

高度：通常定义为从根节点到最远叶子节点的边数。对于空树，高度为 -1 是一种常见的约定（但也有人定义为空树的高度为 0）。 深度：通常是指从根节点到某个节点的边数。对于空树，深度没有意义，也可以定义为 -1。

![image-20250601221024054](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250601221024054.png)

### 4 判断平衡二叉树

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
def is_balanced(root):
    def check_height(node):
        if not node:
            return 0
        left_height = check_height(node.left)
        if left_height == -1:
            return -1  # Left subtree is unbalanced
        right_height = check_height(node.right)
        if right_height == -1:
            return -1  # Right subtree is unbalanced
        if abs(left_height - right_height) > 1:
            return -1  # Current node is unbalanced
        return max(left_height, right_height) + 1
    return check_height(root) != -1
```

### 5 霍夫曼编码

```python
import heapq
class Node:
    def __init__(self, weight, char=None):
        self.weight = weight
        self.char = char
        self.left = None
        self.right = None
    def __lt__(self, other):
        if self.weight == other.weight:
            return self.char < other.char
        return self.weight < other.weight
def huffman_encoding(char_freq):
    heap = [Node(freq, char) for char, freq in char_freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(left.weight + right.weight, min(left.char, right.char))
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)
    return heap[0]
def external_path_length(node, depth=0):
    if node is None:
        return 0
    if node.left is None and node.right is None:
        return depth * node.weight
    return (external_path_length(node.left, depth + 1) +
            external_path_length(node.right, depth + 1))
```

### 6 二叉堆

```python
class BinaryHeap:
    def __init__(self):
        self._heap = []
    def _perc_up(self, i):
        while (i - 1) // 2 >= 0:
            parent_idx = (i - 1) // 2
            if self._heap[i] < self._heap[parent_idx]:
                self._heap[i], self._heap[parent_idx] = (
                    self._heap[parent_idx],
                    self._heap[i],
                )
            i = parent_idx
    def insert(self, item):
        self._heap.append(item)
        self._perc_up(len(self._heap) - 1)
    def _perc_down(self, i):
        while 2 * i + 1 < len(self._heap):
            sm_child = self._get_min_child(i)
            if self._heap[i] > self._heap[sm_child]:
                self._heap[i], self._heap[sm_child] = (
                    self._heap[sm_child],
                    self._heap[i],
                )
            else:
                break
            i = sm_child
    def _get_min_child(self, i):
        if 2 * i + 2 > len(self._heap) - 1:
            return 2 * i + 1
        if self._heap[2 * i + 1] < self._heap[2 * i + 2]:
            return 2 * i + 1
        return 2 * i + 2
    def delete(self):
        self._heap[0], self._heap[-1] = self._heap[-1], self._heap[0]
        result = self._heap.pop()
        self._perc_down(0)
        return result
    def heapify(self, not_a_heap):
        self._heap = not_a_heap[:]
        i = len(self._heap) // 2 - 1    # 超过中点的节点都是叶子节点
        while i >= 0:
            print(f'i = {i}, {self._heap}')
            self._perc_down(i)
            i = i - 1
```

### 7 二叉搜索树

基于BST的快速排序

```python
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
def insert(root, val):
    if root is None:
        return TreeNode(val)
    if val < root.val:
        root.left = insert(root.left, val)
    else:
        root.right = insert(root.right, val)
    return root
def inorder_traversal(root, result):
    if root:
        inorder_traversal(root.left, result)
        result.append(root.val)
        inorder_traversal(root.right, result)
def quicksort(nums):
    if not nums:
        return []
    root = TreeNode(nums[0])
    for num in nums[1:]:
        insert(root, num)
    result = []
    inorder_traversal(root, result)
    return result
```

### 8 平衡二叉搜索树

```python
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.height = 1
class AVL:
    def __init__(self):
        self.root = None
    def insert(self, value):
        if not self.root:
            self.root = Node(value)
        else:
            self.root = self._insert(value, self.root)
    def _insert(self, value, node):
        if not node:
            return Node(value)
        elif value < node.value:
            node.left = self._insert(value, node.left)
        else:
            node.right = self._insert(value, node.right)
        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))
        balance = self._get_balance(node)
        if balance > 1:
            if value < node.left.value:	# 树形是 LL
                return self._rotate_right(node)
            else:	# 树形是 LR
                node.left = self._rotate_left(node.left)
                return self._rotate_right(node)
        if balance < -1:
            if value > node.right.value:	# 树形是 RR
                return self._rotate_left(node)
            else:	# 树形是 RL
                node.right = self._rotate_right(node.right)
                return self._rotate_left(node)
        return node
    def _get_height(self, node):
        if not node:
            return 0
        return node.height
    def _get_balance(self, node):
        if not node:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)
    def _rotate_left(self, z):
        y = z.right
        T2 = y.left
        y.left = z
        z.right = T2
        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        return y
    def _rotate_right(self, y):
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        x.height = 1 + max(self._get_height(x.left), self._get_height(x.right))
        return x
    def preorder(self):
        return self._preorder(self.root)
    def _preorder(self, node):
        if not node:
            return []
        return [node.value] + self._preorder(node.left) + self._preorder(node.right)
```

AVL树删除节点：

```python
class AVL:
    # Existing code...
    def delete(self, value):
        self.root = self._delete(value, self.root)
    def _delete(self, value, node):
        if not node:
            return node
        if value < node.value:
            node.left = self._delete(value, node.left)
        elif value > node.value:
            node.right = self._delete(value, node.right)
        else:
            if not node.left:
                temp = node.right
                node = None
                return temp
            elif not node.right:
                temp = node.left
                node = None
                return temp
            temp = self._min_value_node(node.right)
            node.value = temp.value
            node.right = self._delete(temp.value, node.right)
        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))
        balance = self._get_balance(node)
        if balance > 1:
            if self._get_balance(node.left) >= 0:
                return self._rotate_right(node)
            else:
                node.left = self._rotate_left(node.left)
                return self._rotate_right(node)
        if balance < -1:
            if self._get_balance(node.right) <= 0:
                return self._rotate_left(node)
            else:
                node.right = self._rotate_right(node.right)
                return self._rotate_left(node)
        return node
    def _min_value_node(self, node):
        current = node
        while current.left:
            current = current.left
        return current
    # Existing code...
```

## 7 并查集

```python
class DisjSet:
	def __init__(self, n):
		self.rank = [1] * n
		self.parent = [i for i in range(n)]
	def find(self, x):
		if (self.parent[x] != x):
			self.parent[x] = self.find(self.parent[x])
		return self.parent[x]
	def Union(self, x, y):
		xset = self.find(x)
		yset = self.find(y)
		if xset == yset:
			return
		if self.rank[xset] < self.rank[yset]:
			self.parent[xset] = yset
		elif self.rank[xset] > self.rank[yset]:
			self.parent[yset] = xset
		else:
			self.parent[yset] = xset
			self.rank[xset] = self.rank[xset] + 1
```

通过size来union：

与rank不同，这里的size是指代表该集合的树中元素个数

```python
class UnionFind:
	def __init__(self, n):
		self.Parent = list(range(n))
		self.Size = [1] * n
	def unionBySize(self, i, j):
		irep = self.find(i)
		jrep = self.find(j)
		if irep == jrep:
			return
		isize = self.Size[irep]
		jsize = self.Size[jrep]
		if isize < jsize:
			self.Parent[irep] = jrep
			self.Size[jrep] += self.Size[irep]
		else:
			self.Parent[jrep] = irep
			self.Size[irep] += self.Size[jrep]
```

## 8 图论

### 1 字典树（Trie）

```python
class TrieNode:
    def __init__(self):
        self.child={}
class Trie:
    def __init__(self):
        self.root = TrieNode()
    def insert(self, nums):
        curnode = self.root
        for x in nums:
            if x not in curnode.child:
                curnode.child[x] = TrieNode()
            curnode=curnode.child[x]
    def search(self, num):
        curnode = self.root
        for x in num:
            if x not in curnode.child:
                return 0
            curnode = curnode.child[x]
        return 1
```

### 2 图的表示

邻接矩阵，邻接表，关联矩阵（行代表顶点，列代表边，相连则对应位置为1，否则为0）

OOP:

```python
class Vertex:
    def __init__(self, key):
        self.key = key
        self.neighbors = {}
    def get_neighbor(self, other):
        return self.neighbors.get(other, None)
    def set_neighbor(self, other, weight=0):
        self.neighbors[other] = weight
    def __repr__(self):  # 为开发者提供调试信息
        return f"Vertex({self.key})"
    def __str__(self):  # 面向用户的输出
        return (
                str(self.key)
                + " connected to: "
                + str([x.key for x in self.neighbors])
        )

    def get_neighbors(self):
        return self.neighbors.keys()
    def get_key(self):
        return self.key
class Graph:
    def __init__(self):
        self.vertices = {}
    def set_vertex(self, key):
        self.vertices[key] = Vertex(key)
    def get_vertex(self, key):
        return self.vertices.get(key, None)
    def __contains__(self, key):
        return key in self.vertices
    def add_edge(self, from_vert, to_vert, weight=0):
        if from_vert not in self.vertices:
            self.set_vertex(from_vert)
        if to_vert not in self.vertices:
            self.set_vertex(to_vert)
        self.vertices[from_vert].set_neighbor(self.vertices[to_vert], weight)
    def get_vertices(self):
        return self.vertices.keys()
    def __iter__(self):
        return iter(self.vertices.values())
```

词梯问题：建立桶来建图，然后BFS

骑士周游问题：warnsdorff算法，染色法

```python
def knight_tour(n, path, u, limit):
    u.color = "gray"
    path.append(u)              #当前顶点涂色并加入路径
    if n < limit:
        neighbors = ordered_by_avail(u) #对所有的合法移动依次深入
        #neighbors = sorted(list(u.get_neighbors()))
        i = 0
        for nbr in neighbors:
            if nbr.color == "white" and \               
                knight_tour(n + 1, path, nbr, limit):   #选择“白色”未经深入的点，层次加一，递归深入
                return True
        else:                       #所有的“下一步”都试了走不通
            path.pop()              #回溯，从路径中删除当前顶点
            u.color = "white"       #当前顶点改回白色
            return False
    else:
        return True
```

使用染色法和dfs判断有向图的环

```python
def has_cycle(n, edges):
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)
    color = [0] * n
    def dfs(node):
        if color[node] == 1:
            return True
        if color[node] == 2:
            return False
        color[node] = 1
        for neighbor in graph[node]:
            if dfs(neighbor):
                return True
        color[node] = 2
        return False
    for i in range(n):
        if dfs(i):
            return "Yes"
    return "No"
```

### 3 拓扑排序

#### 1 dfs染色法实现

```python
class Graph:
    def __init__(self):
        self.vertices = {}
        self.num_vertices = 0
    def add_vertex(self, key):
        self.num_vertices = self.num_vertices + 1
        new_ertex = Vertex(key)
        self.vertices[key] = new_ertex
        return new_ertex
    def get_vertex(self, n):
        if n in self.vertices:
            return self.vertices[n]
        else:
            return None
    def __len__(self):
        return self.num_vertices
    def __contains__(self, n):
        return n in self.vertices
    def add_edge(self, f, t, cost=0):
        if f not in self.vertices:
            nv = self.add_vertex(f)
        if t not in self.vertices:
            nv = self.add_vertex(t)
        self.vertices[f].add_neighbor(self.vertices[t], cost)
        #self.vertices[t].add_neighbor(self.vertices[f], cost)
    def getVertices(self):
        return list(self.vertices.keys())
    def __iter__(self):
        return iter(self.vertices.values())
class Vertex:
    def __init__(self, num):
        self.key = num
        self.connectedTo = {}
        self.color = 'white'
        self.distance = sys.maxsize
        self.previous = None
        self.discovery = 0
        self.finish = None
    def __lt__(self, o):
        return self.key < o.key
    def add_neighbor(self, nbr, weight=0):
        self.connectedTo[nbr] = weight
    def setDiscovery(self, dtime):
        self.discovery = dtime
    def setFinish(self, ftime):
        self.finish = ftime
    def getFinish(self):
        return self.finish
    def getDiscovery(self):
        return self.discovery
    def get_neighbors(self):
        return self.connectedTo.keys()
    def __str__(self):
        return str(self.key) + ":color " + self.color + ":disc " + str(self.discovery) + ":fin " + str(
            self.finish) + ":dist " + str(self.distance) + ":pred \n\t[" + str(self.previous) + "]\n"
class DFSGraph(Graph):
    def __init__(self):
        super().__init__()
        self.time = 0
        self.topologicalList = []
    def dfs(self):
        for aVertex in self:
            aVertex.color = "white"
            aVertex.predecessor = -1
        for aVertex in self:
            if aVertex.color == "white":
                self.dfsvisit(aVertex)
    def dfsvisit(self, startVertex):
        startVertex.color = "gray"
        self.time += 1
        startVertex.setDiscovery(self.time)
        for nextVertex in startVertex.get_neighbors():
            if nextVertex.color == "white":
                nextVertex.previous = startVertex
                self.dfsvisit(nextVertex)
        startVertex.color = "black"
        self.time += 1
        startVertex.setFinish(self.time)
    def topologicalSort(self):
        self.dfs()
        temp = list(self.vertices.values())
        temp.sort(key = lambda x: x.getFinish(), reverse = True)
        print([(x.key,x.finish) for x in temp])
        self.topologicalList = [x.key for x in temp]
        return self.topologicalList
```

#### 2 Kahn算法（bfs）

```python
from collections import deque, defaultdict
def topological_sort(graph):
    indegree = defaultdict(int)
    result = []
    queue = deque()
    # 计算每个顶点的入度
    for u in graph:
        for v in graph[u]:
            indegree[v] += 1
    # 将入度为 0 的顶点加入队列
    for u in graph:
        if indegree[u] == 0:
            queue.append(u)
    # 执行拓扑排序
    while queue:
        u = queue.popleft()
        result.append(u)
        for v in graph[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)
    # 检查是否存在环
    if len(result) == len(graph):
        return result
    else:
        return None
```

也可用于判断有向图是否有环

#### 3 无向图判环

##### 1 dfs+visited+parent

```python
def has_cycle_undirected(graph):
    visited = set()
    def dfs(node, parent):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:#注意这里要换行再dfs判断，否则会进入elif的判定导致出错
                if dfs(neighbor, node):
                    return True
            elif neighbor != parent:
                return True
        return False
    for node in graph:
        if node not in visited:
            if dfs(node, -1):
                return True
    return False
```

##### 2 并查集

````python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 路径压缩
        return self.parent[x]
    def union(self, x, y):
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return False  # 同一集合，成环
        self.parent[root_y] = root_x
        return True
def has_cycle_union_find(n, edges):
    uf = UnionFind(n)
    for u, v in edges:
        if not uf.union(u, v):
            return True
    return False
````

#### 4 有向图判环

##### 1 dfs+递归栈

```python
def has_cycle_directed(graph):
    visited = set()
    rec_stack = set()
    def dfs(node):
        visited.add(node)
        rec_stack.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True
        rec_stack.remove(node)
        return False
    for node in graph:
        if node not in visited:
            if dfs(node):
                return True
    return False
```

##### 2 Kahn 见上面

### 4 强连通单元

#### 1 Kosaraju（2dfs）

```python
def dfs1(graph, node, visited, stack):
    visited[node] = True
    for neighbor in graph[node]:
        if not visited[neighbor]:
            dfs1(graph, neighbor, visited, stack)
    stack.append(node)#使用栈来模拟按照结束时间递减顺序访问节点
def dfs2(graph, node, visited, component):
    visited[node] = True
    component.append(node)
    for neighbor in graph[node]:
        if not visited[neighbor]:
            dfs2(graph, neighbor, visited, component)
def kosaraju(graph):
    # Step 1: Perform first DFS to get finishing times
    stack = []
    visited = [False] * len(graph)
    for node in range(len(graph)):
        if not visited[node]:
            dfs1(graph, node, visited, stack)
    # Step 2: Transpose the graph
    transposed_graph = [[] for _ in range(len(graph))]
    for node in range(len(graph)):
        for neighbor in graph[node]:
            transposed_graph[neighbor].append(node)
    # Step 3: Perform second DFS on the transposed graph to find SCCs
    visited = [False] * len(graph)
    sccs = []
    while stack:
        node = stack.pop()
        if not visited[node]:
            scc = []
            dfs2(transposed_graph, node, visited, scc)
            sccs.append(scc)
    return sccs
```

#### 2 Tarjan算法

Tarjan算法使用了一种称为深度优先搜索（DFS）的技术来遍历图，并在遍历的过程中标记和识别强连通分量。算法的基本思想是，通过在深度优先搜索的过程中维护一个栈来记录已经访问过的顶点，并为每个顶点分配一个"搜索次序"（DFS编号）和一个"最低链接值"。搜索次序表示顶点被首次访问的次序，最低链接值表示从当前顶点出发经过一系列边能到达的最早的顶点的搜索次序。

```python
def tarjan(graph):
    """
    Tarjan算法用于查找有向图中的所有强连通分量（Strongly Connected Components, SCCs）。
    参数:
        graph: 邻接表形式表示的图。graph[i] 是节点i指向的所有邻居节点的列表。
    返回:
        一个包含所有SCC的列表，每个SCC是一个节点列表。
    """
    def dfs(node):
        """
        深度优先搜索函数，递归处理每一个节点。
        
        使用nonlocal关键字访问外部变量。
        """
        nonlocal index, stack, indices, low_link, on_stack, sccs
        # 初始化当前节点的时间戳index，并记录到indices和low_link中
        index += 1
        indices[node] = index
        low_link[node] = index
        # 将当前节点压入栈中，表示该节点在当前SCC的候选路径上
        stack.append(node)
        on_stack[node] = True  # 标记该节点在栈中
        # 遍历当前节点的所有邻居
        for neighbor in graph[node]:
            if indices[neighbor] == 0:  # 如果邻居未被访问过
                dfs(neighbor)  # 递归进行DFS
                # 回溯时更新当前节点的low_link值（从子节点继承）
                low_link[node] = min(low_link[node], low_link[neighbor])
            elif on_stack[neighbor]:  # 如果邻居已经被访问且还在栈中（即属于当前SCC路径）
                # 更新当前节点的low_link为邻居的index（回边或横叉边）
                low_link[node] = min(low_link[node], indices[neighbor])
        # 如果当前节点的index等于low_link，说明发现了一个SCC
        if indices[node] == low_link[node]:
            scc = []
            while True:
                top = stack.pop()       # 弹出栈顶元素
                on_stack[top] = False   # 标记不在栈中
                scc.append(top)         # 加入当前SCC集合
                if top == node:         # 直到弹出当前节点为止
                    break
            sccs.append(scc)            # 将找到的SCC加入结果列表
    # 初始化全局变量
    index = 0               # 时间戳索引
    stack = []              # 用于维护DFS过程中节点的栈
    indices = [0] * len(graph)   # 每个节点的访问时间戳（index）
    low_link = [0] * len(graph)  # 每个节点的low值（最早能追溯到的节点）
    on_stack = [False] * len(graph)  # 标记节点是否在栈中
    sccs = []               # 存储所有SCC的结果
    # 对图中每个未访问的节点进行DFS
    for node in range(len(graph)):
        if indices[node] == 0:  # 如果该节点未被访问过
            dfs(node)
    return sccs
```

### 5 dijkstra

```python
def dijkstra(graph,start):
    q = ((0,start))
    heapq.heapify(q)
    dist = [float('inf')] * n
    dist[start] = 0
    visited = set()
    while q:
        distance,node = heapq.heappop(q)
        if node in visited:
            continue
        visited.add(node)
        for nbr in node.neighbours:
            if distance + node.neighbours[nbr] < dist[nbr]:
                dist[nbr] = distance + node.neightbours[nbr]
                heapq.heappush(q,(dist[nbr],nbr))
```

Dijkstra不能用于负权图

### 6 Bellman-Ford

```python
def bellman_ford(graph, V, source):
    # 初始化距离
    dist = [float('inf')] * V
    dist[source] = 0
    # 松弛 V-1 次，每次对所有的边遍历(Dijkstra是对节点遍历)
    for _ in range(V - 1):
        for u, v, w in graph:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    # 检测负权环
    for u, v, w in graph:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            print("图中存在负权环")
            return None
    return dist
```

### 7 Floyd-Warshall算法（多源最短路径）

**思想**：动态规划 + 三重循环

```python
def floyd_warshall(graph):
    V = len(graph)
    dist = [row[:] for row in graph]  # 深拷贝初始图矩阵
    for k in range(V):        # 中间点
        for i in range(V):    # 起点
            for j in range(V):  # 终点
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist
```

### 8 最小生成树（MST）

#### 1 Prim

```python
def prim(graph,start):
    dist = [float('inf')] * n
    dist[start] = 0
    q = [(dist[start],start)]
    heapq.heapify(q)
    visited = set()
    while q:
        distance,node = heapq.heappop(q)
        if node in visited:
            continue
        visited.add(node)
        for nbr in node.neighbours:
            if dist[nbr] > node.neighbours[nbr] and nbr not in visited:
                dist[nbr] = node.neighbours[nbr]
                heapq.heappush(q,(dist[nbr],nbr))
    return dist
```

#### 2 Kruskal（常与并查集一起使用）

以下是Kruskal算法的基本步骤：

1. 将图中的所有边按照权重从小到大进行排序。
2. 初始化一个空的边集，用于存储最小生成树的边。
3. 重复以下步骤，直到边集中的边数等于顶点数减一或者所有边都已经考虑完毕：
   - 选择排序后的边集中权重最小的边。
   - 如果选择的边不会导致形成环路（即加入该边后，两个顶点不在同一个连通分量中），则将该边加入最小生成树的边集中。
4. 返回最小生成树的边集作为结果。

```python
class DisjointSet:
    def __init__(self, num_vertices):
        self.parent = list(range(num_vertices))
        self.rank = [0] * num_vertices
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_x] = root_y
                self.rank[root_y] += 1
def kruskal(graph):
    num_vertices = len(graph)
    edges = []
    # 构建边集
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if graph[i][j] != 0:
                edges.append((i, j, graph[i][j]))
    # 按照权重排序
    edges.sort(key=lambda x: x[2])
    # 初始化并查集
    disjoint_set = DisjointSet(num_vertices)
    # 构建最小生成树的边集
    minimum_spanning_tree = []
    for edge in edges:
        u, v, weight = edge
        if disjoint_set.find(u) != disjoint_set.find(v):
            disjoint_set.union(u, v)
            minimum_spanning_tree.append((u, v, weight))
    return minimum_spanning_tree
```

### 9 关键路径

#### 1. 构建图模型

首先，构建一个有向无环图（DAG），其中：

- **节点**代表事件或里程碑。
- **边**代表活动，并且每条边有一个权重，表示完成该活动所需的时间。

#### 2. 计算最早开始时间 (Earliest Start Time, EST)

使用拓扑排序遍历图，计算每个节点的最早开始时间（EST）。EST 表示从起点到达该节点的最长路径长度。具体步骤如下：

- 初始化所有节点的 EST 为 0。
- 对于图中的每一个节点 `u`，更新其所有邻接节点 `v` 的 EST 值：如果 `EST[u] + weight(u, v)` 大于 `EST[v]`，则更新 `EST[v] = EST[u] + weight(u, v)`。

#### 3. 计算最晚开始时间 (Latest Start Time, LST)

反向遍历拓扑排序后的图，计算每个节点的最晚开始时间（LST）。LST 表示为了不延迟整个项目的完成时间，节点 `u` 必须的最晚开始时间。具体步骤如下：

- 初始化终点的 LST 为其 EST 值。
- 对于图中的每一个节点 `u`，更新其所有前置节点 `v` 的 LST 值：如果 `LST[u] - weight(v, u)` 小于 `LST[v]`，则更新 `LST[v] = LST[u] - weight(v, u)`。

#### 4. 确定关键路径

- 关键活动是指那些最早开始时间和最晚开始时间相等的活动。即对于边 `(u, v)`，如果 `EST[u] + weight(u, v) == LST[v]`，则 `(u, v)` 是关键活动。
- 通过检查所有边来确定哪些是关键活动，并根据这些关键活动构建关键路径。

```python
from collections import defaultdict, deque
class Edge:
    def __init__(self, v, w):
        self.v = v#终点
        self.w = w#权重
def topo_sort(n, G, in_degree):
    q = deque([i for i in range(n) if in_degree[i] == 0])
    ve = [0] * n
    topo_order = []
    while q:
        u = q.popleft()
        topo_order.append(u)
        for edge in G[u]:
            v = edge.v
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)
            if ve[u] + edge.w > ve[v]:
                ve[v] = ve[u] + edge.w
    if len(topo_order) == n:
        return ve, topo_order
    else:
        return None, None
def get_critical_path(n, G, in_degree):
    ve, topo_order = topo_sort(n, G, in_degree.copy())
    if ve is None:
        return -1, []
    maxLength = max(ve)
    vl = [maxLength] * n
    for u in reversed(topo_order):
        for edge in G[u]:
            v = edge.v
            if vl[v] - edge.w < vl[u]:
                vl[u] = vl[v] - edge.w
    activity = defaultdict(list)
    for u in G:
        for edge in G[u]:
            v = edge.v
            e, l = ve[u], vl[v] - edge.w
            if e == l:
                activity[u].append(v)
    return maxLength, activity
# Main
n, m = map(int, input().split())
G = defaultdict(list)
in_degree = [0] * n
for _ in range(m):
    u, v, w = map(int, input().split())
    G[u].append(Edge(v, w))
    in_degree[v] += 1
maxLength, activity = get_critical_path(n, G, in_degree)
if maxLength == -1:
    print("No")
else:
    print("Yes")
    print(f"Critical Path Length: {maxLength}")
    # 打印所有关键路径
    def print_critical_path(u, activity, path=[]):
        path.append(u)
        if u not in activity or not activity[u]:
            print("->".join(map(str, path)))
        else:
            for v in sorted(activity[u]):
                print_critical_path(v, activity, path.copy())
        path.pop()
    for i in range(n):
        if in_degree[i] == 0:
            print_critical_path(i, activity)
```

## 9 散列表

```python
class HashTable:
    def __init__(self):
        self.size = 11
        self.slots = [None] * self.size
        self.data = [None] * self.size
    def put(self,key,data):
        hashvalue = self.hashfunction(key,len(self.slots))
        if self.slots[hashvalue] == None:
            self.slots[hashvalue] = key
            self.data[hashvalue] = data
        else:
            if self.slots[hashvalue] == key:#注意key相同的情况
                self.data[hashvalue] = data #replace
            else:
                nextslot = self.rehash(hashvalue,len(self.slots))
                while self.slots[nextslot] != None and self.slots[nextslot] != key:
                    nextslot = self.rehash(nextslot,len(self.slots))
                if self.slots[nextslot] == None:
                    self.slots[nextslot] = key
                    self.data[nextslot] = data
                else:
                    self.data[nextslot] = data #replace
    def hashfunction(self,key,size):
        return key%size
    def rehash(self,oldhash,size):
        return (oldhash+1)%size
    def get(self,key):
        startslot = self.hashfunction(key,len(self.slots))
        data = None
        stop = False
        found = False
        position = startslot
        while self.slots[position] != None and not found and not stop:
                if self.slots[position] == key:
                    found = True
                    data = self.data[position]
                else:
                    position=self.rehash(position,len(self.slots))
                    if position == startslot:
                        stop = True
        return data
    def __getitem__(self,key):
        return self.get(key)
    def __setitem__(self,key,data):
        self.put(key,data)
```

## 10 KMP

```python
""""
compute_lps 函数用于计算模式字符串的LPS表。LPS表是一个数组，
其中的每个元素表示模式字符串中当前位置之前的子串的最长前缀后缀的长度。
该函数使用了两个指针 length 和 i，从模式字符串的第二个字符开始遍历。
"""
def compute_lps(pattern):
    """
    计算pattern字符串的最长前缀后缀（Longest Proper Prefix which is also Suffix）表
    :param pattern: 模式字符串
    :return: lps表
    """
    m = len(pattern)
    lps = [0] * m  # 初始化lps数组
    length = 0  # 当前最长前后缀长度
    for i in range(1, m):  # 注意i从1开始，lps[0]永远是0
        while length > 0 and pattern[i] != pattern[length]:
            length = lps[length - 1]  # 回退到上一个有效前后缀长度
        if pattern[i] == pattern[length]:
            length += 1
        lps[i] = length
    return lps
def kmp_search(text, pattern):
    n = len(text)
    m = len(pattern)
    if m == 0:
        return 0
    lps = compute_lps(pattern)
    matches = []
    # 在 text 中查找 pattern
    j = 0  # 模式串指针
    for i in range(n):  # 主串指针
        while j > 0 and text[i] != pattern[j]:
            j = lps[j - 1]  # 模式串回退
        if text[i] == pattern[j]:
            j += 1
        if j == m:
            matches.append(i - j + 1)  # 匹配成功
            j = lps[j - 1]  # 查找下一个匹配
    return matches
```

周期性质

<img src="C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250603203302082.png" alt="image-20250603203302082" style="zoom: 60%;" />

```python
    def compute_lps(s):
        lps = [0] * len(s)
        length = 0
        for i in range(1,len(s)):
            while length > 0 and s[i] != s[length]:
                length = lps[length - 1]
            if s[i] == s[length]:
                length += 1
            lps[i] = length
        return lps
    def min_repeat(s):
        lps = compute_lps(s)
        res = []
        for i in range(1,len(s)):
            if (i + 1) % (i + 1 - lps[i]) == 0 and lps[i] != 0:
                res.append((i + 1,(i + 1) // (i + 1 - lps[i])))
        return res
```

## 11 其它

浮点数保留

```python
a = 3.1415926
print(f{a:.2f})
```

