# Assignment #A: Graph starts

Updated 1830 GMT+8 Apr 22, 2025

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

### M19943:图的拉普拉斯矩阵

OOP, implementation, http://cs101.openjudge.cn/practice/19943/

要求创建Graph, Vertex两个类，建图实现。

思路：

用集合记录Vertex的邻居节点，用字典记录Graph中的节点

代码：

```python
class Vertex:
    def __init__(self,key):
        self.key = key
        self.neighbours = set()
    def add_neighbour(self,another):
        self.neighbours.add(another)
    def neighbour_nums(self):
        return len(self.neighbours)
class Graph:
    def __init__(self):
        self.vertices = {}
    def add_vertex(self,key):
        self.vertices[key] = Vertex(key)
    def add_edge(self,one,another):
        self.vertices[one].add_neighbour(another)
        self.vertices[another].add_neighbour(one)
    def has_edge(self,one,another):
        return another in self.vertices[one].neighbours
n,m = map(int,input().split())
graph = Graph()
for i in range(n):
    graph.add_vertex(i)
for i in range(m):
    a,b = map(int,input().split())
    graph.add_edge(a,b)
L = [[0]*n for i in range(n)]
for i in range(n):
    for j in range(n):
        if i == j:
            L[i][j] = graph.vertices[i].neighbour_nums()
        else:
            if graph.has_edge(i,j):
                L[i][j] = -1
for i in range(n):
    print(*L[i])
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250424161221259](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250424161221259.png)



### LC78.子集

backtracking, https://leetcode.cn/problems/subsets/

思路：

dfs+回溯

代码：

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        used = [False]*n
        res = []
        def dfs(nums,k,temp):
            if k >= n:
                res.append(temp)
                return
            dfs(nums,k + 1,temp)
            if not used[k]:
                used[k] = True
                dfs(nums,k + 1,temp + [nums[k]])
                used[k] = False
        dfs(nums,0,[])
        return res
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250424171848964](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250424171848964.png)



### LC17.电话号码的字母组合

hash table, backtracking, https://leetcode.cn/problems/letter-combinations-of-a-phone-number/

思路：

跟上一题类似

代码：

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        dic = {2:'abc',3:'def',4:'ghi',5:'jkl',6:'mno',7:'pqrs',8:'tuv',9:'wxyz'}
        res = []
        def dfs(digits,i,tmp):
            if i >= len(digits):
                if tmp:
                    res.append(tmp)
                return
            n = int(digits[i])
            for s in dic[n]:
                dfs(digits,i + 1,tmp + s)
        dfs(digits,0,'')
        return res
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250424173048112](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250424173048112.png)

### M04089:电话号码

trie, http://cs101.openjudge.cn/practice/04089/

思路：

只记得字典树是字典套字典，但课件里的代码只是粗略扫了一眼，没记住具体怎么写，于是自己想了一个。对于这题而言，通过递归计算总叶子节点数量，如果与输入的数据数量不一致就输出NO。但是一开始写的时候一直wa，和ai共同探索后才发现number里的数据一开始以int形式存了，导致有先导0时会出错，去了int之后果然ac了，感觉是值得进我错题本的题

代码：

```python
class Trie:
    def __init__(self,key = None):
        self.dic = {}
        self.key = key
    def add(self,s):
        if s[0] not in self.dic:
            self.dic[s[0]] = Trie(s[0])
        a = self.dic[s[0]]
        if len(s) > 1:
            a.add(s[1:])
    def leaf_nums(self):
        if len(self.dic) == 0:
            return 1
        num = 0
        for i in self.dic:
            num += self.dic[i].leaf_nums()
        return num
t = int(input())
for i in range(t):
    n = int(input())
    trie = Trie()
    number = []
    for j in range(n):
        number.append(input())
    number.sort(key = lambda x:-len(x))
    for k in number:
        trie.add(str(k))
    if trie.leaf_nums() == n:
        print('YES')
    else:
        print('NO')
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250427152139484](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250427152139484.png)



### T28046:词梯

bfs, http://cs101.openjudge.cn/practice/28046/

思路：

把每个单词之间逐个字母比较并构建邻居关系然后狠狠超时了。。。桶减少时间复杂度还挺猛的

代码：

```python
from collections import deque
class Vertex:
    def __init__(self,key):
        self.key = key
        self.neighbours = set()
    def add_neighbour(self,other):
        self.neighbours.add(other)
        other.neighbours.add(self)
def bucket(words):
    pattern_dict = {}
    for word in words:
        for i in range(len(word.key)):
            pattern = word.key[:i] + '_' + word.key[i + 1:]
            if pattern not in pattern_dict:
                pattern_dict[pattern] = []
            pattern_dict[pattern].append(word)
    for pattern in pattern_dict:
        for i in range(len(pattern_dict[pattern])):
            for j in range(i + 1,len(pattern_dict[pattern])):
                pattern_dict[pattern][i].add_neighbour(pattern_dict[pattern][j])
def bfs(w1,w2):
    q = deque()
    visited = set()
    q.append((w1,[w1.key]))
    visited.add(w1)
    while q:
        cur,path = q.popleft()
        if cur == w2:
            return path
        for i in cur.neighbours:
            if i not in visited:
                visited.add(i)
                q.append((i,path + [i.key]))
    return ['NO']
n = int(input())
words = {}
for i in range(n):
    w = input()
    words[w] = Vertex(w)
bucket(words.values())
w1,w2 = input().split()
res = bfs(words[w1],words[w2])
print(*res)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250427171859170](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250427171859170.png)



### T51.N皇后

backtracking, https://leetcode.cn/problems/n-queens/

思路：

dfs+回溯，需要判断一下对角线上是否都没有皇后

代码：

```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        res = []
        def judge(i,j,board):
            for k in range(i):
                if j + k - i >= 0 and board[k][j + k - i] == 'Q':
                    return False
                if j + i - k < n and board[k][j + i - k] == 'Q':
                    return False
            return True
        def dfs(board,used,i):
            if i >= n:
                board = [''.join(row) for row in board]
                res.append(board)
                return
            for j in range(n):
                if not used[j]:
                    if judge(i,j,board):
                        board[i][j] = 'Q'
                        used[j] = True
                        dfs(board,used,i + 1)
                        board[i][j] = '.'
                        used[j] = False
        dfs([['.']*n for _ in range(n)],[False]*n,0)
        return res
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250427174755738](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250427174755738.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

感觉这次作业的45两题做起来比较吃力，分别是对字典树和桶不太熟悉，剩下的倒感觉还好，五一多学学数算









