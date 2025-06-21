# Assignment #B: 图为主

Updated 2223 GMT+8 Apr 29, 2025

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

### E07218:献给阿尔吉侬的花束

bfs, http://cs101.openjudge.cn/practice/07218/

思路：

bfs模板题

代码：

```python
from collections import deque
def bfs(xs,ys,e,R,C,maze):
    q = deque()
    visited = set()
    q.append((xs,ys,0))
    visited.add((xs,ys))
    while q:
        x,y,n = q.popleft()
        if maze[x][y] == e:
            return n
        dr = [(0,1),(0,-1),(1,0),(-1,0)]
        for i in range(4):
            for j in range(4):
                nx,ny = x+dr[i][0],y+dr[i][1]
                if 0 <= nx < R and 0 <= ny < C:
                    if maze[nx][ny] != '#' and (nx,ny) not in visited:
                        visited.add((nx,ny))
                        q.append((nx,ny,n + 1))
    return False
T = int(input())
for _ in range(T):
    R,C = map(int,input().split())
    maze = []
    for i in range(R):
        maze.append(list(input()))
    xs,ys = 0,0
    for i in range(R):
        for j in range(C):
            if maze[i][j] == 'S':
               xs,ys = i,j
               break
    if bfs(xs,ys,'E',R,C,maze):
        print(bfs(xs,ys,'E',R,C,maze))
    else:
        print('oop!')
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250505182750686](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250505182750686.png)



### M3532.针对图的路径存在性查询I

disjoint set, https://leetcode.cn/problems/path-existence-queries-in-a-graph-i/

思路：

运用并查集的思路，为了减少比较的次数只要比较相邻的数差值是否<=maxDiff，满足条件的话就把他们合并到一个集合里（也就是归为同一个parent），最后只要比较parent是否相同就行了

代码：

```python
class Solution:
    def pathExistenceQueries(self, n: int, nums: List[int], maxDiff: int, queries: List[List[int]]) -> List[bool]:
        res = []
        rank = [1]*n
        parent = list(range(n))
        def find(i):
            if parent[i] == i:
                return i
            return(find(parent[i]))
        for i in range(n - 1):
            if nums[i + 1] - nums[i] <= maxDiff:
                j = find(i)
                if rank[j] < rank[i + 1]:
                    parent[j] = i + 1
                elif rank[j] > rank[i + 1]:
                    parent[i + 1] = j
                else:
                    parent[j] = i + 1
                    rank[i + 1] += 1
        for case in queries:
            if find(case[0]) == find(case[1]):
                res.append(True)
            else:
                res.append(False)
        return res
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250505200100451](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250505200100451.png)



### M22528:厚道的调分方法

binary search, http://cs101.openjudge.cn/practice/22528/

思路：

先排序找到六十分位数再二分即可，期末能不能这么调分（

代码：

```python
score = sorted(list(map(float,input().split())),reverse = True)
n = len(score)
a = 3*n//5 if 3*n/5 == 3*n//5 else 3*n//5+1
x = score[a - 1]
def new_score(x,b):
    return b*x/1000000000+1.1**(b*x/1000000000)
l = 0
r = 1000000000
while l < r:
    mid = (l + r) // 2
    if new_score(x,mid) >= 85:
        r = mid
    else:
        l = mid + 1
print(l)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250505211437858](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250505211437858.png)



### Msy382: 有向图判环 

dfs, https://sunnywhy.com/sfbj/10/3/382

思路：

运用字典建立有向图，然后对于每个节点进行dfs，一旦找到一个环就停止  

代码：

```python
def dfs(a,b,Graph,visited):
    global has_circle
    if not Graph[a]:
        return
    for i in Graph[a]:
        if i == b:
            has_circle = True
            return
        if i not in visited:
            visited.add(i)
            dfs(i,b,Graph,visited)

from collections import defaultdict
Graph = defaultdict(list)
n,m = map(int,input().split())
for i in range(m):
    u,v = map(int,input().split())
    Graph[u].append(v)
has_circle = False
for i in range(n):
    dfs(i,i,Graph,{i})
    if has_circle:
        break
print('Yes' if has_circle else 'No')
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250505215230610](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250505215230610.png)



### M05443:兔子与樱花

Dijkstra, http://cs101.openjudge.cn/practice/05443/

思路：

运用bfs找到路程最短的路线

代码：

```python
from collections import defaultdict
import heapq
def bfs(s,e,Graph):
    q = []
    heapq.heapify(q)
    heapq.heappush(q,(0,s,[s],{s}))
    while q:
        distance,position,path,visited = heapq.heappop(q)
        if position == e:
            return ''.join(path)
        for next_position,next_distance in Graph[position]:
            if next_position not in visited:
                heapq.heappush(q,(distance+next_distance,next_position,path+['->(',str(next_distance),')->',next_position],visited.union({next_position})))

P = int(input())
for i in range(P):
    input()
Q = int(input())
Graph = defaultdict(list)
for i in range(Q):
    u,v,w = input().split()
    Graph[u].append((v,int(w)))
    Graph[v].append((u,int(w)))
R = int(input())
for i in range(R):
    s,e = input().split()
    print(bfs(s,e,Graph))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250505223146069](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250505223146069.png)



### T28050: 骑士周游

dfs, http://cs101.openjudge.cn/practice/28050/

思路：

好难啊啊啊啊啊

代码：

```python
def is_valid_move(x,y,board,n):
    return 0 <= x < n and 0 <= y < n and board[x][y] == 0
def get_degree(x,y,move):
    cnt = 0
    for dx,dy in move:
        nx,ny = x+dx,y+dy
        if is_valid_move(nx,ny,board,n):
            cnt += 1
    return cnt
n = int(input())
sr,sc = map(int,input().split())
board = [[0]*n for _ in range(n)]
board[sr][sc] = 1
move = [(1,2),(1,-2),(2,1),(2,-1),(-1,2),(-1,-2),(-2,1),(-2,-1)]
def dfs(x,y,k):
    if k == n**2-1:
        return True
    next_moves = []
    for dx,dy in move:
        nx,ny = x+dx,y+dy
        if is_valid_move(nx,ny,board,n):
            degree = get_degree(nx,ny,move)
            next_moves.append((degree,nx,ny))
        next_moves.sort()
    for degree,nx,ny in next_moves:
        board[nx][ny] = 1
        if dfs(nx,ny,k+1):
            return True
        board[nx][ny] = 0
    return False
print('success' if dfs(sr,sc,0) else 'fail')
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250506115426482](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250506115426482.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

作业中搜索题目占比较大， 第二题让我复习了一下并查集的内容，前面几题都还行，骑士周游真是太恶心了，这个Warnsdorff规则真是完全想不到









