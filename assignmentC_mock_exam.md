# Assignment #C: 202505114 Mock Exam

Updated 1518 GMT+8 May 14, 2025

2025 spring, Complied by <mark>蔡东辰、工学院</mark>



> **说明：**
>
> 1. **⽉考**：AC3<mark>（请改为同学的通过数）</mark> 。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。
>
> 2. **解题与记录：**
>
>    对于每一个题目，请提供其解题思路（可选），并附上使用Python或C++编写的源代码（确保已在OpenJudge， Codeforces，LeetCode等平台上获得Accepted）。请将这些信息连同显示“Accepted”的截图一起填写到下方的作业模板中。（推荐使用Typora https://typoraio.cn 进行编辑，当然你也可以选择Word。）无论题目是否已通过，请标明每个题目大致花费的时间。
>
> 3. **提交安排：**提交时，请首先上传PDF格式的文件，并将.md或.doc格式的文件作为附件上传至右侧的“作业评论”区。确保你的Canvas账户有一个清晰可见的头像，提交的文件为PDF格式，并且“作业评论”区包含上传的.md或.doc附件。
>
> 4. **延迟提交：**如果你预计无法在截止日期前提交作业，请提前告知具体原因。这有助于我们了解情况并可能为你提供适当的延期或其他帮助。 
>
> 请按照上述指导认真准备和提交作业，以保证顺利完成课程要求。



## 1. 题目

### E06364: 牛的选举

http://cs101.openjudge.cn/practice/06364/

思路：

排序一下就行

代码：

```python
N,K = map(int,input().split())
cow = []
for i in range(N):
    a,b = map(int,input().split())
    cow.append((i+1,a,b))
cow.sort(key = lambda x:-x[1])
cow = cow[:K]
cow.sort(key = lambda x:-x[2])
print(cow[0][0])
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250515185811800](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250515185811800.png)



### M04077: 出栈序列统计

http://cs101.openjudge.cn/practice/04077/

思路：

stack，In，out分别用来模拟栈，待入栈元素，出栈序列，用dfs进行递归

代码：

```python
n = int(input())
res = []
def dfs(n,stack,In,out):
    global res
    if len(out) == n:
        res.append(out)
        return
    if In:
        new = In.pop()
        stack.append(new)
        dfs(n,stack,In,out)
        old = stack.pop()
        In.append(old)
    if stack:
        new = stack.pop()
        out.append(new)
        dfs(n,stack,In,out)
        old = out.pop()
        stack.append(old)
dfs(n,[],list(range(n)),[])
print(len(res))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250515185919396](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250515185919396.png)



### M05343:用队列对扑克牌排序

http://cs101.openjudge.cn/practice/05343/

思路：

按照题目里讲的规则模拟一下就行了

代码：

```python
n = int(input())
puke = list(map(str,input().split()))
queue = [[] for i in range(10)]
for i in puke:
    queue[int(i[1])].append(i)
queue1 = [[] for i in range(4)]
dic = {'A':1,'B':2,'C':3,'D':4}
for i in range(9):
    a=' '.join(queue[i + 1])
    print(f'Queue{i + 1}:{a}')
    while queue[i+1]:
        j = queue[i+1].pop()
        queue1[dic[j[0]]-1].append(j)
a=' '.join(queue1[0])
print(f'QueueA:{a}')
a=' '.join(queue1[1])
print(f'QueueB:{a}')
a=' '.join(queue1[2])
print(f'QueueC:{a}')
a=' '.join(queue1[3])
print(f'QueueD:{a}')
res = []
for i in range(4):
    res.extend(queue1[i])
print(*res)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250515191007723](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250515191007723.png)



### M04084: 拓扑排序

http://cs101.openjudge.cn/practice/04084/

思路：

考试时对拓扑排序不太熟就跳了，复习一遍后自己做发现还挺模板的，看了题解发现原来还可以用heapq进一步简化

代码：

```python
from collections import defaultdict,deque
v,a = map(int,input().split())
g = defaultdict(list)
indegree = [0]*(v+1)
for i in range(a):
    u,w = map(int,input().split())
    g[u].append(w)
    indegree[w] += 1
q = deque()
res = []
for u in range(1,v+1):
    if indegree[u] == 0:
        q.append(u)
        break
while q:
    u = q.popleft()
    res.append(u)
    indegree[u] -= 1
    for i in g[u]:
        indegree[i] -= 1
    for i in range(1,v+1):
        if indegree[i] == 0:
            q.append(i)
            break
res = ['v'+str(x) for x in res]
print(*res)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250517220919077](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250517220919077.png)



### M07735:道路

Dijkstra, http://cs101.openjudge.cn/practice/07735/

思路：

如果visited中只存节点的名称的话会导致一些合理的情况不被进行计算，最后得到错误的结果；但如果把已走的节点集合放进数组中一起存在q里就会爆内存，百思不得其解之际瞅了一眼题解发现只要把节点名称和花费一起存进visited里不就好了！这么简单的解决方法居然没想到，真是糊涂了...

代码：

```python
import heapq
from collections import defaultdict
g = defaultdict(list)
K = int(input())
N = int(input())
R = int(input())
for i in range(R):
    S,D,L,T = map(int,input().split())
    g[S].append((D,L,T))
q = [(0,1,0)]#距离，节点名，花费
vis = set()
vis.add((1,0))
heapq.heapify(q)
res = -1
while q:
    length,name,cost = heapq.heappop(q)
    if name == N:
        res = length
        break
    else:
        vis.add((name,cost))
        for D,L,T in g[name]:
            if cost + T <= K and (D,cost + T) not in vis:
                heapq.heappush(q,(length + L,D,cost + T))
print(res)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250518120859735](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250518120859735.png)



### T24637:宝藏二叉树

dp, http://cs101.openjudge.cn/practice/24637/

思路：

可以说是难度诈骗的一道题，无论是思维量还是写代码难度都不高，应该算M，所以为啥我考试的时候写了个麻烦得莫名其妙的超时代码呢（沉思）

代码：

```python
class Tree:
    def __init__(self,value):
        self.value = value
        self.left = None
        self.right = None
def treasure(root):
    if not root:
        return 0
    a = treasure(root.left) + treasure(root.right)
    b = root.value
    if root.left:
        b += treasure(root.left.left) + treasure(root.left.right)
    if root.right:
        b += treasure(root.right.left) + treasure(root.right.right)
    return max(a,b)
N = int(input())
tree = list(map(int,input().split()))
tree = [Tree(x) for x in tree]
for i in range(N // 2):
    tree[i].left = tree[2 * i + 1]
    if 2 * i + 2 < N:
        tree[i].right = tree[2 * i + 2]
print(treasure(tree[0]))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250517222208318](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250517222208318.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

机考ac3，考得不是很好，先是看到不咋熟的拓扑排序一下子乱了阵脚，跳过之后扑克那题还compile error，道路更是毫无思路，宝藏二叉树虽然想到了思路但是脑子一团乱了写了半天整了个超时，还好后来把扑克那题改过来了，现在回看感觉题目确实不算特别难，感觉考场心态还是不太好，另外对于新学的这几种图算法掌握得还是不太牢
