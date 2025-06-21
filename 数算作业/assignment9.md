# Assignment #9: Huffman, BST & Heap

Updated 1834 GMT+8 Apr 15, 2025

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

### LC222.完全二叉树的节点个数

dfs, https://leetcode.cn/problems/count-complete-tree-nodes/

思路：

递归处理，总节点数为左子树节点数加右子树节点数加一

代码：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def countNodes(self, root: Optional[TreeNode]) -> int:
        def count(root):
            if not root:
                return 0
            return count(root.left) + count(root.right) + 1
        return count(root)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250421215520902](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250421215520902.png)



### LC103.二叉树的锯齿形层序遍历

bfs, https://leetcode.cn/problems/binary-tree-zigzag-level-order-traversal/

思路：

使用元组来同时记录节点以及遍历的方向，比较麻烦，看到题解之后才意识到deque不但能从左面出还能从左面进

代码：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        q = deque()
        q.append((root,'l'))
        res = []
        level = 'l'
        val_temp = []
        tree_temp = []
        while q:
            root,lr = q.popleft()
            if lr != level:
                level = lr
                if val_temp:
                    res.append(val_temp)
                    val_temp = []
            val_temp.append(root.val)
            if lr == 'l':
                if root.left:
                    tree_temp.append((root.left,'r'))
                if root.right:
                    tree_temp.append((root.right,'r'))
            elif lr == 'r':
                if root.right:
                    tree_temp.append((root.right,'l'))
                if root.left:
                    tree_temp.append((root.left,'l'))
            if tree_temp and not q:
                q.extend(tree_temp[::-1])
                tree_temp = []
        if val_temp:
            res.append(val_temp)
        return res
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250421222859319](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250421222859319.png)



### M04080:Huffman编码树

greedy, http://cs101.openjudge.cn/practice/04080/

思路：

利用heapq实现

代码：

```python
import heapq
n = int(input())
nums = list(map(int,input().split()))
heapq.heapify(nums)
res = 0
while len(nums) > 1:
    a = heapq.heappop(nums)
    b = heapq.heappop(nums)
    heapq.heappush(nums,a + b)
    res += (a + b)
print(res)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250421225707174](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250421225707174.png)



### M05455: 二叉搜索树的层次遍历

http://cs101.openjudge.cn/practice/05455/

思路：

重点在于insert的递归函数，除此之外的部分就都很好弄了

代码：

```python
from collections import deque
nums = list(map(int,input().split()))
nums1 = set()
nums2 = []
for i in nums:
    if i not in nums1:
        nums1.add(i)
        nums2.append(i)
class Tree:
    def __init__(self,val):
        self.val = val
        self.left = None
        self.right = None
def insert(root,val):
    if not root:
        return Tree(val)
    if val < root.val:
        root.left = insert(root.left,val)
    if val > root.val:
        root.right = insert(root.right,val)
    return root
def levelOrder(root):
    q = deque()
    q.append(root)
    res = []
    while q:
        root = q.popleft()
        res.append(root.val)
        if root.left:
            q.append(root.left)
        if root.right:
            q.append(root.right)
    return res
root = Tree(nums2[0])
for i in nums2[1:]:
    root = insert(root,i)
res = levelOrder(root)
print(*res)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250421231309806](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250421231309806.png)



### M04078: 实现堆结构

手搓实现，http://cs101.openjudge.cn/practice/04078/

类似的题目是 晴问9.7: 向下调整构建大顶堆，https://sunnywhy.com/sfbj/9/7

思路：

新加入的数和父节点比较，pop时将第一个数与最后一个交换，将交换后的第一个数与子节点比较

代码：

```python
class heap:
    def __init__(self):
        self.arr = []
        self.size = 0
    def append(self,new):
        self.arr.append(new)
        self.size += 1
        self.up(self.size - 1)
    def up(self,i):
        while i > 0:
            parent = (i - 1) // 2
            if self.arr[i] < self.arr[parent]:
                self.arr[i],self.arr[parent] = self.arr[parent],self.arr[i]
            i = parent
    def pop(self):
        self.arr[0],self.arr[-1] = self.arr[-1],self.arr[0]
        res = self.arr.pop()
        self.size -= 1
        self.down(0)
        return res
    def down(self,i):
        while 2 * i + 1 < self.size:
            j = self.smaller(i)
            if self.arr[i] > self.arr[j]:
                self.arr[i],self.arr[j] = self.arr[j],self.arr[i]
            i = j
    def smaller(self,i):
        if 2 * i + 1 < self.size:
            if 2 * i + 2 == self.size:
                return 2 * i + 1
            else:
                if self.arr[2 * i + 1] < self.arr[2 * i + 2]:
                    return 2 * i + 1
                else:
                    return 2 * i + 2
n = int(input())
nums = heap()
for i in range(n):
    a = list(map(int,input().split()))
    if a[0] == 1:
        u = a[1]
        nums.append(u)
    else:
        print(nums.pop())
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250422134447687](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250422134447687.png)



### T22161: 哈夫曼编码树

greedy, http://cs101.openjudge.cn/practice/22161/

思路：

先建树并定义大小比较，其中vals表示子树叶子节点对应的字符的集合，之后定义编码与解码的函数，需要注意的是解码函数在执行到最后一步（s为空）时需要将当前节点的值加进path里而编码函数不需要进行这一点（因为解码函数是“阶段性”地向path中添加东西而编码函数是“一步步”地添加）

代码：

```python
import heapq
class Tree:
    def __init__(self,val,weight,vals):
        self.val = val
        self.left = None
        self.right = None
        self.weight = weight
        self.vals = vals
    def __lt__(self, other):
        if self.weight != other.weight:
            return self.weight < other.weight
        else:
            return self.val < other.val
n = int(input())
tree = []
for i in range(n):
    val,weight = map(str,input().split())
    tree.append(Tree(val,int(weight),{val}))
heapq.heapify(tree)
while len(tree) > 1:
    a = heapq.heappop(tree)
    b = heapq.heappop(tree)
    if a.val < b.val:
        c = Tree(a.val,a.weight + b.weight,a.vals.union(b.vals))
    else:
        c = Tree(b.val,a.weight + b.weight,a.vals.union(b.vals))
    c.left = a
    c.right = b
    heapq.heappush(tree,c)
def code_to_str(root,s,path):
    if not s:
        return path + root.val
    if not root.left and not root.right:
        path += root.val
        return code_to_str(tree[0],s,path)
    if s[0] == '0':
        return code_to_str(root.left,s[1:],path)
    elif s[0] == '1':
        return code_to_str(root.right,s[1:],path)
def str_to_code(root,s,path):
    if not s:
        return path
    if not root.left and not root.right:
        return str_to_code(tree[0],s[1:],path)
    if s[0] in root.left.vals:
        return str_to_code(root.left,s,path + '0')
    else:
        return str_to_code(root.right,s,path + '1')
while True:
    try:
        s = input()
        root = tree[0]
        if s[0] in '10':
            print(code_to_str(root,s,''))
        else:
            print(str_to_code(root,s,''))
    except EOFError:
        break
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250422143129988](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250422143129988.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

让人心烦的思修pre终于结束了，这几天把树的那个文档过了一遍，因此作业题没有太卡思路的地方，主要是写代码时的各种细节问题（比如遇上空节点该怎么办）需要想一想

每日选做好像落下了很多，该补一补了









