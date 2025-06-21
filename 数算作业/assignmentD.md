# Assignment #D: 图 & 散列表

Updated 2042 GMT+8 May 20, 2025

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

### M17975: 用二次探查法建立散列表

http://cs101.openjudge.cn/practice/17975/

<mark>需要用这样接收数据。因为输入数据可能分行了，不是题面描述的形式。OJ上面有的题目是给C++设计的，细节考虑不周全。</mark>

```python
import sys
input = sys.stdin.read
data = input().split()
index = 0
n = int(data[index])
index += 1
m = int(data[index])
index += 1
num_list = [int(i) for i in data[index:index+n]]
```



思路：

自己写的时候忽略了关键字相同的情况导致一直wa，看了题解后发现可以用一个列表来储存每个关键字的位置

代码：

```python
import sys
input = sys.stdin.read
data = input().split()
index = 0
n = int(data[index])
index += 1
m = int(data[index])
index += 1
num_list = [int(i) for i in data[index:index+n]]
table = [None] * m
cnt = 0
res = []
for key in num_list:
    pos = key % m
    if table[pos] == None or table[pos] == key:
        table[pos] = key
        res.append(pos)
    else:
        ok = False
        while not ok:
            cnt += 1
            if cnt % 2 == 0:
                pos -= (cnt // 2) ** 2
            else:
                pos += ((cnt + 1) // 2) ** 2
            if table[pos] == None or table[pos] == key:
                table[pos] = key
                res.append(pos)
                ok = True
print(*res)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250525104507390](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250525104507390.png)



### M01258: Agri-Net

MST, http://cs101.openjudge.cn/practice/01258/

思路：

Prim算法，随便选一个节点开始然后遍历与它相连的节点，不断更新dist

代码：

```python
import heapq
while True:
    try:
        N = int(input())
        farm = []
        for i in range(N):
            farm.append(list(map(int,input().split())))
        dist = [float('inf')]*N
        dist[0] = 0
        pq = []
        heapq.heapify(pq)
        heapq.heappush(pq,(0,0))
        visited = set()
        while pq:
            distance,node = heapq.heappop(pq)
            if node not in visited:
                visited.add(node)
                for another in range(N):
                    if another != node and dist[another] > farm[another][node] and another not in visited:
                        dist[another] = farm[another][node]
                        heapq.heappush(pq,(dist[another],another))
        print(sum(dist))
    except EOFError:
        break
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250525104713487](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250525104713487.png)



### M3552.网络传送门旅游

bfs, https://leetcode.cn/problems/grid-teleportation-traversal/

思路：

超时整得人要疯了。。。走到传送门处先考虑进行传送，把传送门出口先变成已访问，再考虑正常走，对于传送门相邻的情况起到一个剪枝的作用；由于传送不消耗步数，所以哪怕出口已访问也可以传送一下试试看（否则会wa。。。debug老半天）

看了题解发现居然可以转化成Dijkstra+BFS，使用距离数组dis就可以做各种判定，还顺带着记录了距离，绝了！

代码：

```python
class Solution:
    def minMoves(self, matrix: List[str]) -> int:
        doors = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        same_doors = {i:[] for i in doors}
        m = len(matrix)
        n = len(list(matrix[0]))
        for i in range(m):
            for j in range(n):
                if matrix[i][j] in doors:
                    same_doors[matrix[i][j]].append((i,j))
        visited = [[False]*n for _ in range(m)]
        visited[0][0] = True
        q = [(0,0,0,set(),visited)]
        heapq.heapify(q)
        while q:
            dist,x,y,used,visited = heapq.heappop(q)
            if x == m - 1 and y == n - 1:
                return dist
            if matrix[x][y] in doors:
                door = matrix[x][y]
                if door not in used:
                    this_door = same_doors[door]
                    if len(this_door) > 1:
                        used.add(door)
                        for position in this_door:
                            if position != (x,y):
                                nx,ny = position
                                visited[nx][ny] = True
                                heapq.heappush(q,(dist,nx,ny,used,visited))
            move = [(1,0),(-1,0),(0,1),(0,-1)]
            for i in range(4):
                dx,dy = move[i]
                nx,ny = x + dx,y + dy
                if 0 <= nx < m and 0 <= ny < n and matrix[nx][ny] != '#' and not visited[nx][ny]:
                    visited[nx][ny] = True
                    heapq.heappush(q,(dist + 1,nx,ny,used,visited))
        return -1
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250525130740859](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250525130740859.png)



### M787.K站中转内最便宜的航班

Bellman Ford, https://leetcode.cn/problems/cheapest-flights-within-k-stops/

思路：

最多K站中转，那么就是遍历K+1次，但是如果直接用dist进行比较并修改的话遍历K+1次可能仍会出现中转站多于K的情况，所以需要把dist拷贝后进行比较

代码：

```python
class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        dist = [float('inf')] * n
        dist[src] = 0
        for i in range(k + 1):
            dist1 = dist[:]
            for f,t,p in flights:
                if dist[t] > dist1[f] + p:
                    dist[t] = dist1[f] + p
        if dist[dst] != float('inf'):
            return dist[dst]
        else:
            return -1
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250525115610054](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250525115610054.png)



### M03424: Candies

Dijkstra, http://cs101.openjudge.cn/practice/03424/

思路：

实质上就是求第N个节点到第一个节点的最短距离

代码：

```python
from collections import defaultdict
import heapq
N,M = map(int,input().split())
g = defaultdict(list)
for i in range(M):
    a,b,c = map(int,input().split())
    g[a-1].append((b-1,c))
more_than_snoopy = [float('inf')]*N
more_than_snoopy[0] = 0
q = []
heapq.heapify(q)
heapq.heappush(q,(0,0))
vis = set()
while q:
    dist,child = heapq.heappop(q)
    if child not in vis:
        vis.add(child)
        for another,more in g[child]:
            if more_than_snoopy[another] > more_than_snoopy[child] + more:
                more_than_snoopy[another] = more_than_snoopy[child] + more
                heapq.heappush(q,(dist + more,another))
print(more_than_snoopy[N - 1])
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250525115835907](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250525115835907.png)



### M22508:最小奖金方案

topological order, http://cs101.openjudge.cn/practice/22508/

思路：

每轮的胜负关系可以视为由败者指向胜者的有向边，建立好图之后进行拓扑排序，同时维护奖金数组reward，由于奖金为整数且要和最小，所以如果胜者的奖金小于等于败者就把它变成比败者多1，排序好之后把reward求和即可

代码：

```python
from collections import deque
n,m = map(int,input().split())
indegree = [0] * n
reward = [100] * n
g = {i:[] for i in range(n)}
for i in range(m):
    win,lose = map(int,input().split())
    indegree[win] += 1
    g[lose].append(win)
q = deque([i for i in range(n) if indegree[i] == 0])
while q:
    lose = q.popleft()
    for win in g[lose]:
        if reward[win] <= reward[lose]:
            reward[win] = reward[lose] + 1
        indegree[win] -= 1
        if indegree[win] == 0:
            q.append(win)
print(sum(reward))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250525120737107](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250525120737107.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

重新恶补了一遍图的课件，把各种算法的名称以及思路狠狠记下来了，感觉效果还行，至少这次作业除了传送门都感觉不算恶心，虽然说传送门的标签也是中等吧...机考临近，未来一周多把前面的课件也再重新过一遍，再针对性补一补每日选做，希望机考能ac5吧









