# Assignment #8: 树为主

Updated 1704 GMT+8 Apr 8, 2025

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

### LC108.将有序数组转换为二叉树

dfs, https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/

思路：

不断二分，每次把当前节点对应的数左侧未进入树中的一段数组二分，把中间的数作为左子节点，右侧同理，不断操作直到所有数都进入树中

代码：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        n = len(nums)
        mid = n // 2
        root = TreeNode(nums[mid])
        def lefttree(i,j,root):
            if i < 0 or i > n - 1 or j < 0 or j > n - 1:
                return
            mid = (i + j) // 2
            child = TreeNode(nums[mid])
            root.left = child
            if mid > i:
                lefttree(i,mid - 1,child)
            if mid < j:
                righttree(mid + 1,j,child)
        def righttree(i,j,root):
            if i < 0 or i > n - 1 or j< 0 or j > n - 1:
                return
            mid = (i + j) // 2
            child = TreeNode(nums[mid])
            root.right = child
            if mid > i:
                lefttree(i,mid - 1,child)
            if mid < j:
                righttree(mid + 1,j,child)
        lefttree(0,mid - 1,root)
        righttree(mid + 1,n - 1,root)
        return root
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250412222029987](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250412222029987.png)



### M27928:遍历树

 adjacency list, dfs, http://cs101.openjudge.cn/practice/27928/

思路：

先建树，然后找到树根，最后dfs遍历

代码：

```python
class Tree:
    def __init__(self,value = 0,children = None,parent = None):
        self.value = value
        self.children = children
        self.parent = parent
    def is_child(self,parent):
        if not parent.children:
            parent.children = [(self.value,self)]
        else:
            parent.children.append((self.value,self))
        self.parent = parent
n = int(input())
tree = dict()
a = 0
for i in range(n):
    nums = list(map(int,input().split()))
    if i == 0:
        a = nums[0]
    if nums[0] not in tree:
        tree[nums[0]] = Tree(nums[0])
    for j in range(1,len(nums)):
        if nums[j] not in tree:
            tree[nums[j]] = Tree(nums[j])
        tree[nums[j]].is_child(tree[nums[0]])
root = tree[a]
while root.parent:
    root = root.parent
res = []
def dfs(root):
    if not root.children:
        res.append(root.value)
        return
    root.children.sort()
    if root.value < root.children[0][1].value:
        res.append(root.value)
    for i in range(len(root.children)):
        dfs(root.children[i][1])
        if i < len(root.children) - 1 and root.children[i][1].value < root.value< root.children[i + 1][1].value:
            res.append(root.value)
    if root.value > root.children[-1][1].value:
        res.append(root.value)
dfs(root)
for i in res:
    print(i)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250412231448339](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250412231448339.png)



### LC129.求根节点到叶节点数字之和

dfs, https://leetcode.cn/problems/sum-root-to-leaf-numbers/

思路：

dfs遍历即可，当节点没有子节点时终止，并把路径加入result里，最后把result的结果相加

代码：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        result = []
        def dfs(root,path):
            if not root.left and not root.right:
                result.append(''.join(path))
                return
            if root.left:
                dfs(root.left,path + [str(root.left.val)])
            if root.right:
                dfs(root.right,path + [str(root.right.val)])
        dfs(root,[str(root.val)])
        result = [int(x) for x in result]
        return sum(result)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250412234523845](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250412234523845.png)



### M22158:根据二叉树前中序序列建树

tree, http://cs101.openjudge.cn/practice/22158/

思路：

这题没能自己想出来（自己想的方法有bug打了一堆补丁还是不太行，最终放弃），看了题解，感觉最巧妙的一个地方是前序和中序的前面一些元素都是根节点以及左子树，利用这一点就可以写出递归的程序了

代码：

```python
class Tree:
    def __init__(self,value):
        self.value = value
        self.left = None
        self.right = None
while True:
    try:
        prefix = input()
        infix = input()
        def build_tree(prefix,infix):
            if not prefix:
                return None
            root = Tree(prefix[0])
            root_index = infix.index(prefix[0])
            root.left = build_tree(prefix[1:root_index + 1],infix[:root_index])
            root.right = build_tree(prefix[root_index + 1:],infix[root_index + 1:])
            return root
        root = build_tree(prefix,infix)
        res = []
        def postfix(root):
            if not root:
                return
            postfix(root.left)
            postfix(root.right)
            res.append(root.value)
        postfix(root)
        print(''.join(res))
    except EOFError:
        break
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250415205932523](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250415205932523.png)



### M24729:括号嵌套树

dfs, stack, http://cs101.openjudge.cn/practice/24729/

思路：

遇到左括号将前一节点入栈，遇到右括号将节点出栈，遇到字母就建树并加入到栈的最后一个节点的子节点列表中

代码：

```python
class Tree:
    def __init__(self,value):
        self.value = value
        self.children = []
tree = list(input())
stack = []
root = None
for i in range(len(tree)):
    if tree[i] == '(':
        if root:
            stack.append(root)
            root = None
    elif tree[i] == ')':
        if stack:
            root = stack.pop()
    elif tree[i] != ',':
        root = Tree(tree[i])
        if stack:
            stack[-1].children.append(root)
def prefix(root):
    if not root:
        return ''
    res = root.value
    for child in root.children:
        res += prefix(child)
    return res
def postfix(root):
    if not root:
        return ''
    res = ''
    for child in root.children:
        res += postfix(child)
    res += root.value
    return res
print(prefix(root))
print(postfix(root))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250415213244698](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250415213244698.png)



### LC3510.移除最小数对使数组有序II

doubly-linked list + heap, https://leetcode.cn/problems/minimum-pair-removal-to-sort-array-ii/

思路：



代码：

```python

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>





## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

这次作业1235都还算有思路，第四题感觉小难，主要还是关键方法没想到，至于第六题粗略看看感觉做不出，于是就先把前五题交了。这周天天操心思修pre，被整到崩溃，以后再也不想当pre组长了，好在还有两天就要苦尽甘来了······









