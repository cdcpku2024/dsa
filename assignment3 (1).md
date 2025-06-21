# Assignment #3: 惊蛰 Mock Exam

Updated 1641 GMT+8 Mar 5, 2025

2025 spring, Complied by <mark>同学的姓名、院系</mark>



> **说明：**
>
> 1. **惊蛰⽉考**：AC4<mark>（请改为同学的通过数）</mark> 。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。
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

### E04015: 邮箱验证

strings, http://cs101.openjudge.cn/practice/04015



思路：

按照题目里的条件一个一个检验就行，小坑点是.@的情况容易漏

代码：

```python
def ok(email):
    at = 0
    dot = 0
    if email[0] == '@' or email[0] == '.' or email[-1] == '@' or email[-1] == '.':
        return 'NO'
    else:
        for i in range(len(email)):
            if email[i] == '@':
                if at == 0 and email[i-1] != '.':
                    at += 1
                else:
                    return 'NO'
            if email[i] == '.' and at == 1:
                dot += 1
                if email[i - 1] == '@':
                    return 'NO'
        if at == 1 and dot >= 1:
            return 'YES'
        else:
            return 'NO'
while True:
    try:
        email = input().strip()
        print(ok(email))
    except EOFError:
        break
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250309190131174](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250309190131174.png)



### M02039: 反反复复

implementation, http://cs101.openjudge.cn/practice/02039/



思路：

创建一个字典，把字母挨个塞进字典里就行，不过i%n==0的判断方法会比较特殊一点，需要看i//n的奇偶，最后把字母按字典中的顺序连到一起即可

代码：

```python
n = int(input())
s = input()
message = dict()
for i in range(1,n+1):
    message[i] = []
for i in range(1,len(s)+1):
    if i%n != 0:
        if i%n == i%(2*n):
            message[i%n].append(s[i-1])
        else:
            message[n+1-i%n].append(s[i-1])
    if i%n == 0:
        if (i//n)%2 == 0:
            message[1].append(s[i-1])
        else:
            message[n].append(s[i-1])
output = []
for i in range(1,n+1):
    output.extend(message[i])
print(''.join(output))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250309190207092](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250309190207092.png)



### M02092: Grandpa is Famous

implementation, http://cs101.openjudge.cn/practice/02092/



思路：

英文题目好恶心...读题读了半天才搞懂怎么排，其实就是按照序号出现次数进行排序，然后找排在第二的，不过多个人并列的情况确实比较烦，思路是把所有人的数据排好之后遍历找到第二，再遍历一次找出所有并列第二的人

代码：

```python
from collections import defaultdict
while True:
    N,M =map(int,input().split())
    if N == M == 0:
        break
    rank_time = defaultdict(int)
    for i in range(N):
        ranking = list(map(int,input().split()))
        for j in ranking:
            rank_time[j] += 1
    result = sorted(rank_time.items(),key=lambda x:-x[1])
    max = result[0][1]
    index = 0
    for i in range(len(result)):
        if result[i][1] < max:
            max = result[i][1]
            index = i
            break
    output = []
    for i in range(index,len(result)):
        if result[i][1] == max:
            output.append(result[i][0])
        else:
            break
    output.sort()
    print(*output)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250309190550895](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250309190550895.png)



### M04133: 垃圾炸弹

matrices, http://cs101.openjudge.cn/practice/04133/



思路：

计概旧题，但是考场上一下子忘了咋写了，遂重写，写得比较麻烦

代码：

```python
d = int(input())
n = int(input())
rubbish = []
for j in range(n):
    x,y,i = map(int,input().split())
    rubbish.append((x,y,i))
bomb = [[0]*1025 for i in range(1025)]
cnt,max_n = 0,0
for i in range(1025):
    for j in range(1025):
        for k in rubbish:
            if abs(i-k[0]) <= d and abs(j-k[1]) <= d:
                bomb[i][j] += k[2]
        if bomb[i][j] == max_n:
            cnt += 1
        if bomb[i][j] > max_n:
            cnt = 1
            max_n = bomb[i][j]
print(cnt,max_n)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250309191506324](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250309191506324.png)



### T02488: A Knight's Journey

backtracking, http://cs101.openjudge.cn/practice/02488/



思路：

好久没做搜索题目了，代码怎么写都快忘了，考场上遇到时误解了题目的意思（以为必须从A1开始走），然后想着用bfs解决，最后没写完，但课下写好代码发现会mle跟tle，之后又尝试dfs，发现还是dfs写起来更顺手，另外这个代码如果是先找到所有路径再输出字典序最小的会超时，所以就从起点字典序较小的情况开始遍历，一旦在某种情况下找到合适的路径了就直接终止循环输出结果

代码：

```python
n = int(input())
s = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
for i in range(n):
    p,q = map(int,input().split())
    print(f'Scenario #{i+1}:')
    result = []
    chessboard = [[False]*q for i in range(p)]
    def horse(x,y,path):
        if len(path) == 2*p*q:
            result.append(path)
            return
        chessboard[x][y] = True
        dir = [(1,2),(1,-2),(-1,2),(-1,-2),(2,1),(2,-1),(-2,1),(-2,-1)]
        for j in range(8):
            dx,dy = dir[j]
            nx,ny = x+dx,y+dy
            if 0 <= nx < p and 0 <= ny < q:
                if not chessboard[nx][ny]:
                    horse(nx,ny,path+f'{s[ny]}'+f'{nx+1}')
        chessboard[x][y] = False
    def output():
        for j in range(p):
            for k in range(q):
                horse(j,k,f'{s[k]}'+f'{j+1}')
                if result:
                    result.sort()
                    return result[0]
        return 'impossible'
    print(output())
    print()
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250310175458805](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250310175458805.png)



### T06648: Sequence

heap, http://cs101.openjudge.cn/practice/06648/



思路：

没想出来，看了题解觉得很巧妙，每次只要把两个列表中的元素相加，找出最小的n个和，那么最终最小的n个和就一定是由这n个和中的某一些加上后面的序列中的元素构成，这样就能大幅减少计算量了

代码：

```python
import heapq
T = int(input())
for i in range(T):
    m,n = map(int,input().split())
    seq1 = sorted(list(map(int,input().split())))
    for j in range(m-1):
        seq2 = sorted(list(map(int,input().split())))
        min_sum = [(seq1[k]+seq2[0],k,0) for k in range(n)]
        heapq.heapify(min_sum)
        result = []
        for k in range(n):
            current_sum,x,y = heapq.heappop(min_sum)
            result.append(current_sum)
            if y+1 < n:
                heapq.heappush(min_sum,(seq1[x]+seq2[y+1],x,y+1))
        seq1 = result
    print(*seq1)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20250311142006918](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250311142006918.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

ac4，只能说中规中矩吧，第五题误解了题意方向搞错了，白白浪费半个多小时，如果一开始读明白题并且写dfs的话应该能ac5，不过第六题是真没思路

缓慢跟进每日选做...









