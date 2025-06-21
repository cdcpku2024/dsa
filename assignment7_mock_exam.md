# Assignment #7: 20250402 Mock Exam

Updated 1624 GMT+8 Apr 2, 2025

2025 spring, Complied by <mark>蔡东辰、工学院</mark>



> **说明：**
>
> 1. **⽉考**：AC6<mark>（请改为同学的通过数）</mark> 。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。
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

### E05344:最后的最后

http://cs101.openjudge.cn/practice/05344/



思路：

简单的双端队列

代码：

```python
from collections import deque
n,k = map(int,input().split())
people = deque(list(range(1,n + 1)))
die = []
while len(people) > 1:
    for i in range(k - 1):
        out = people.popleft()
        people.append(out)
    si = people.popleft()
    die.append(si)
print(' '.join([str(i) for i in die]))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250406173158309](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250406173158309.png)



### M02774: 木材加工

binary search, http://cs101.openjudge.cn/practice/02774/



思路：

二分加贪心，这种题也留过好几次了

代码：

```python
N,K = map(int,input().split())
wood = []
r = 0
for i in range(N):
    a = int(input())
    wood.append(a)
    r = max(r,a)
if sum(wood) < K:
    print(0)
else:
    l = 1
    while l < r-1:
        mid = (l + r) // 2
        small = 0
        for i in wood:
            small += (i // mid)
        if small >= K:
            l = mid
        elif small < K:
            r = mid - 1
    if l == r:
        print(l)
    else:
        small = 0
        for i in wood:
            small += (i // r)
        if small >= K:
            print(r)
        else:
            print(l)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250406173640236](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250406173640236.png)



### M07161:森林的带度数层次序列存储

tree, http://cs101.openjudge.cn/practice/07161/



思路：

把每个节点依次从tree1里pop出来，如果节点有子节点则将子节点从tree2中popleft出来，与当前节点连上，再把当前节点加入tree2，相当于是进行了一个“从底部往顶部建树”的过程，因为先进tree2的树是底层的树，所以可以保证建树的过程中每个节点和子节点对应关系正确

代码：

```python
from collections import deque
class Tree():
    def __init__(self, n=0, name=None):
        self.children = []
        self.name = name
        for i in range(n):
            self.children.append(Tree())
        self.children_nums = n
    def child(self, another, i):
        self.children[i] = another
result = []
n = int(input())
forest = []
for i in range(n):
    tree1 = list(map(str, input().split()))
    tree2 = deque()
    while tree1:
        children_nums = int(tree1.pop())
        name = tree1.pop()
        tree = Tree(children_nums, name)
        if children_nums > 0:
            for j in range(children_nums):
                child_tree = tree2.popleft()
                Tree.child(tree, child_tree, children_nums - j - 1)
        tree2.append(tree)
    res = []
    def dfs(tree):
        if tree.children_nums == 0:
            res.append(tree.name)
            return
        for j in range(tree.children_nums):
            dfs(tree.children[j])
        res.append(tree.name)
    dfs(tree2[0])
    result.extend(res)
print(*result)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250406174304003](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250406174304003.png)



### M18156:寻找离目标数最近的两数之和

two pointers, http://cs101.openjudge.cn/practice/18156/



思路：

双指针，不断更新close1和close2

代码：

```python
T = int(input())
S = sorted(list(map(int,input().split())))
l = 0
r = len(S) - 1
close1 = 0
close2 = float('inf')
ok = False
while l < r:
    if S[l] + S[r] == T:
        print(T)
        ok = True
        break
    if S[l] + S[r] < T:
        close1 = max((S[l] + S[r]),close1)
        l += 1
    elif S[l] + S[r] > T:
        close2 = min((S[l] + S[r]),close2)
        r -= 1
if not ok:
    if T - close1 <= close2 - T:
        print(close1)
    else:
        print(close2)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250406191055687](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250406191055687.png)



### M18159:个位为 1 的质数个数

sieve, http://cs101.openjudge.cn/practice/18159/



思路：

欧拉筛

代码：

```python
T = int(input())
case = []
for i in range(T):
    case.append(int(input()))
MAX = max(case)
nums = [True]*(MAX + 1)
nums[0] = nums[1] = False
for i in range(2,MAX + 1):
    if nums[i]:
        for j in range(2,(MAX // i) + 1):
            nums[j*i] = False
for i in range(T):
    print(f'Case{i + 1}:')
    num = case[i]
    result = []
    for j in range(2,num):
        if nums[j] and str(j)[-1] == '1':
            result.append(str(j))
    print(' '.join(result) if result else 'NULL')
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250406191438188](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250406191438188.png)



### M28127:北大夺冠

hash table, http://cs101.openjudge.cn/practice/28127/



思路：

拿字典把每队对应的各种数据都存起来，再进行排序即可

代码：

```python
M = int(input())
dic = dict()
for i in range(M):
    name,q,finish = map(str,input().split(','))
    if name not in dic:
        dic[name] = [0,0,set()]#提交次数，ac数，ac题目
    dic[name][0] += 1
    if finish == 'yes' and q not in dic[name][2]:
        dic[name][1] += 1
        dic[name][2].add(q)
res = [(-dic[i][1],dic[i][0],i) for i in dic]
res.sort()
res = res[:12] if len(res) > 12 else res
for i in range(len(res)):
    print(f'{i + 1} {res[i][2]} {-res[i][0]} {res[i][1]}')
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20250406192236193](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250406192236193.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

这次月考不是很难，但由于对树还是不是很熟悉（可以叫作“还是不树”），树的那题想了一会就直接跳了，把后面的题做完后发现居然刚过了四十多分钟，回来接着想第三题，费了老大劲终于在结束前十多分钟弄出来了

下周要期中，最近几乎没看数算，期中后再恶补吧。假期在复习期中和出门看花之中度过了，海棠花真好看，错过就得再等一年了









