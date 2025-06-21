# Assignment #6: 回溯、树、双向链表和哈希表

Updated 1526 GMT+8 Mar 22, 2025

2025 spring, Complied by <mark>蔡东辰、工学院</mark>



> **说明：**
>
> 1. **解题与记录：**
>
>    对于每一个题目，请提供其解题思路（可选），并附上使用Python或C++编写的源代码（确保已在OpenJudge， Codeforces，LeetCode等平台上获得Accepted）。请将这些信息连同显示“Accepted”的截图一起填写到下方的作业模板中。（推荐使用Typora https://typoraio.cn 进行编辑，当然你也可以选择Word。）无论题目是否已通过，请标明每个题目大致花费的时间。
>
> 2. **提交安排：**提交时，请首先上传PDF格式的文件，并将.md或.doc格式的文件作为附件上传至右侧的“作业评论”区。确保你的Canvas账户有一个清晰可见的头像，提交的文件为PDF格式，并且“作业评论”区包含上传的.md或.doc附件。
>
> 3. **延迟提交：**如果你预计无法在截止日期前提交作业，请提前告知具体原因。这有助于我们了解情况并可能为你提供适当的延期或其他帮助。 
>
> 请按照上述指导认真准备和提交作业，以保证顺利完成课程要求。



## 1. 题目

### LC46.全排列

backtracking, https://leetcode.cn/problems/permutations/

思路：

dfs+回溯

代码：

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        def dfs(nums,n,result,output,used):
            if len(result) == n:
                output.append(result.copy())
                return
            for i in range(n):
                if not used[i]:
                    result.append(nums[i])
                    used[i] = True
                    dfs(nums,n,result,output,used)
                    used[i] = False
                    result.pop()
        output = []
        dfs(nums,n,[],output,[False]*n)
        return output
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250331142147894](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250331142147894.png)



### LC79: 单词搜索

backtracking, https://leetcode.cn/problems/word-search/

思路：

dfs+回溯

代码：

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        m = len(board)
        n = len(board[0])
        def dfs(x,y,i,used):
            if i == len(word):
                return True
            dx = [1,-1,0,0]
            dy = [0,0,1,-1]
            for j in range(4):
                nx,ny = x + dx[j],y + dy[j]
                if 0 <= nx < m and 0 <= ny < n:
                    if board[nx][ny] == word[i] and not used[nx][ny]:
                        used[nx][ny] = True
                        if dfs(nx,ny,i + 1,used):
                            return True
                        used[nx][ny] = False
            return False
        for x in range(m):
            for y in range(n):
                if board[x][y] == word[0]:
                    used = [[False]*n for _ in range(m)]
                    used[x][y] = True
                    canfind = dfs(x,y,1,used)
                    if canfind:
                        return True
        return False
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250331230859006](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250331230859006.png)



### LC94.二叉树的中序遍历

dfs, https://leetcode.cn/problems/binary-tree-inorder-traversal/

思路：

递归，把整体的中序遍历拆成左边的中序遍历+root+右边的中序遍历

代码：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        def infix(tree):
            if not tree:
                return []
            if tree.left == tree.right == None:
                return [tree.val]
            if tree.left == None:
                return [tree.val] + infix(tree.right)
            if tree.right == None:
                return infix(tree.left) + [tree.val]
            return infix(tree.left) + [tree.val] + infix(tree.right)
        return infix(root)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250331232839678](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250331232839678.png)



### LC102.二叉树的层序遍历

bfs, https://leetcode.cn/problems/binary-tree-level-order-traversal/

思路：

bfs即可，需要注意最后要把最后一层放进结果里

代码：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        q = deque()
        result = []
        temp = []
        now_layer = 0
        q.append((root,0))
        while q:
            tree,layer = q.popleft()
            if layer > now_layer:
                now_layer += 1
                result.append(temp)
                temp = []
            if tree:
                temp.append(tree.val)
                if tree.left:
                    q.append((tree.left,layer + 1))
                if tree.right:
                    q.append((tree.right,layer + 1))
        if temp:
            result.append(temp)
        return result
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250401094734128](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250401094734128.png)



### LC131.分割回文串

dp, backtracking, https://leetcode.cn/problems/palindrome-partitioning/

思路：

教训：把每个分割结果放进最终结果时一定要拷贝，忘了这一点卡了好久

代码：

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        def huiwen(s):
            if len(s) == 1:
                return True
            i,j = 0,len(s) - 1
            while i < j:
                if s[i] != s[j]:
                    return False
                i += 1
                j -= 1
            return True
        res = []
        tmp = []
        def dfs(i,j,s,t):
            if t == len(s):
                res.append(tmp[:])
                return
            for k in range(i,j + 1):
                if huiwen(s[i:k + 1]):
                    tmp.append(s[i:k + 1])
                    dfs(k + 1,j,s,t + k + 1 - i)
                    tmp.pop()
        dfs(0,len(s) - 1,s,0)
        return res
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250401142502159](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250401142502159.png)



### LC146.LRU缓存

hash table, doubly-linked list, https://leetcode.cn/problems/lru-cache/

思路：

用链表来记录使用顺序太绝了！感觉这个方法在优化代码时会很有用

代码：

```python
class ListNode:
    def __init__(self,key=None,value=None):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None
class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = dict()
        self.now = 0
        self.head = ListNode()
        self.tail = ListNode()
        self.head.next = self.tail
        self.tail.prev = self.head
    def move_to_tail(self,key):
        node = self.cache[key]
        node.prev.next = node.next
        node.next.prev = node.prev
        self.tail.prev.next = node
        node.prev = self.tail.prev
        node.next = self.tail
        self.tail.prev = node
    def get(self, key: int) -> int:
        if key in self.cache:
            self.move_to_tail(key)
            return self.cache[key].value
        return -1
    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache[key].value = value
            self.move_to_tail(key)
        else:
            if self.now < self.capacity:
                self.now += 1
            else:
                self.cache.pop(self.head.next.key)
                self.head.next = self.head.next.next
                self.head.next.prev = self.head
            new = ListNode(key,value)
            self.cache[key] = new
            new.prev = self.tail.prev
            new.next = self.tail
            self.tail.prev.next = new
            self.tail.prev = new

# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250401231443234](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250401231443234.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

这周作业不算太难，但是仍然被细节卡了很久（比如忘记拷贝），快期中了没怎么投入时间给数算，明天月考去一下看看能做出几个









