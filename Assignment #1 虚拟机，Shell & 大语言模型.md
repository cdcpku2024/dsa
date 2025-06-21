# Assignment #1: 虚拟机，Shell & 大语言模型

Updated 2309 GMT+8 Feb 20, 2025

2025 spring, Complied by <mark>同学的姓名、院系</mark>



**作业的各项评分细则及对应的得分**

| 标准                                 | 等级                                                         | 得分 |
| ------------------------------------ | ------------------------------------------------------------ | ---- |
| 按时提交                             | 完全按时提交：1分<br/>提交有请假说明：0.5分<br/>未提交：0分  | 1 分 |
| 源码、耗时（可选）、解题思路（可选） | 提交了4个或更多题目且包含所有必要信息：1分<br/>提交了2个或以上题目但不足4个：0.5分<br/>少于2个：0分 | 1 分 |
| AC代码截图                           | 提交了4个或更多题目且包含所有必要信息：1分<br/>提交了2个或以上题目但不足4个：0.5分<br/>少于：0分 | 1 分 |
| 清晰头像、PDF文件、MD/DOC附件        | 包含清晰的Canvas头像、PDF文件以及MD或DOC格式的附件：1分<br/>缺少上述三项中的任意一项：0.5分<br/>缺失两项或以上：0分 | 1 分 |
| 学习总结和个人收获                   | 提交了学习总结和个人收获：1分<br/>未提交学习总结或内容不详：0分 | 1 分 |
| 总得分： 5                           | 总分满分：5分                                                |      |
>
>
>
>**说明：**
>
>1. **解题与记录：**
>      - 对于每一个题目，请提供其解题思路（可选），并附上使用Python或C++编写的源代码（确保已在OpenJudge， Codeforces，LeetCode等平台上获得Accepted）。请将这些信息连同显示“Accepted”的截图一起填写到下方的作业模板中。（推荐使用Typora https://typoraio.cn 进行编辑，当然你也可以选择Word。）无论题目是否已通过，请标明每个题目大致花费的时间。
>
>2. **课程平台与提交安排：**
>
>  - 我们的课程网站位于Canvas平台（https://pku.instructure.com ）。该平台将在第2周选课结束后正式启用。在平台启用前，请先完成作业并将作业妥善保存。待Canvas平台激活后，再上传你的作业。
>
>      - 提交时，请首先上传PDF格式的文件，并将.md或.doc格式的文件作为附件上传至右侧的“作业评论”区。确保你的Canvas账户有一个清晰可见的头像，提交的文件为PDF格式，并且“作业评论”区包含上传的.md或.doc附件。
>
>3. **延迟提交：**
>
>  - 如果你预计无法在截止日期前提交作业，请提前告知具体原因。这有助于我们了解情况并可能为你提供适当的延期或其他帮助。 
>
>请按照上述指导认真准备和提交作业，以保证顺利完成课程要求。



## 1. 题目

### 27653: Fraction类

http://cs101.openjudge.cn/practice/27653/



思路：

课件里的原题，主要就是定义好Fraction类的相加运算，用辗转相除法找最大公因数进行约分

代码：

```python
def gcd(m,n):
    while m % n != 0:
        oldm = m
        oldn = n
        m = oldn
        n = oldm % oldn
    return n
class Fraction:
    def __init__(self,top,bottom):
        self.top = top
        self.bottom = bottom
    def __str__(self):
        a = gcd(self.top,self.bottom)
        return str(self.top//a)+'/'+str(self.bottom//a)
    def __add__(self,other):
        bottom = self.bottom * other.bottom
        top = self.top * other.bottom + self.bottom * other.top
        return Fraction(top,bottom)
a,b,c,d = map(int,input().split())
print(Fraction(a,b) + Fraction(c,d))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250225143612250](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250225143612250.png)



### 1760.袋子里最少数目的球

 https://leetcode.cn/problems/minimum-limit-of-balls-in-a-bag/




思路：

偷看了一眼力扣的提示才想到要二分，对于球数最多的箱子的球数进行讨论，用二分查找缩小范围，并检验移动次数是否超了

代码：

```python
class Solution:
    def minimumSize(self, nums: List[int], maxOperations: int) -> int:
        max_cost = max(nums)
        min_cost = 0
        while max_cost > min_cost:
            mid = (max_cost + min_cost) // 2
            if mid == 0:
                return 1
            op = 0
            new_nums = nums.copy()
            for num in new_nums:
                if num > mid:
                    if num % mid == 0:
                        op += (num // mid - 1)
                    else:
                        op += num // mid
            if op > maxOperations:
                min_cost = mid + 1
            else:
                max_cost = mid
        return min_cost
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250225163954341](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250225163954341.png)



### 04135: 月度开销

http://cs101.openjudge.cn/practice/04135



思路：

试图递归，不出所料喜提超时，扫了一眼题解发现又是二分，所幸知道方向之后能独立写出代码

代码：

```python
N,M=map(int,input().split())
cost=[]
for i in range(N):
    cost.append(int(input()))
min_cost = max(cost)
max_cost = sum(cost)
while min_cost < max_cost:
    mid = (min_cost + max_cost) // 2
    fajo = 1
    cost_in_this_fajo = 0
    for money in cost:
        if cost_in_this_fajo + money <= mid:
            cost_in_this_fajo += money
        else:
            fajo += 1
            cost_in_this_fajo = money
    if fajo <= M:
        max_cost = mid
    else:
        min_cost = mid + 1
print(max_cost)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250225173447747](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250225173447747.png)



### 27300: 模型整理

http://cs101.openjudge.cn/practice/27300/



思路：先对模型名称排序，再对参数进行排序，起初先是Compile Error然后又是Presentation Error，对比题解的代码后发现输出中的':'和','后面都要加空格。。。希望以后的题目在这方面的要求能写在题目中



代码：

```python
n = int(input())
models=[]
dic={}
for i in range(n):
    model,parameter = map(str,input().split('-'))
    if parameter[-1] == 'M':
        parameter1 = float(parameter[:-1])*10**6
    else:
        parameter1 = float(parameter[:-1])*10**9
    if model not in models:
        models.append(model)
        dic[model] = []
    dic[model].append((parameter1,parameter))
models.sort()
for model in models:
    dic[model].sort()
    value = ', '.join([i[1] for i in dic[model]])
    print(f'{model}: {value}')
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250302141749321](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250302141749321.png)



### Q5. 大语言模型（LLM）部署与测试

本任务旨在本地环境或通过云虚拟机（如 https://clab.pku.edu.cn/ 提供的资源）部署大语言模型（LLM）并进行测试。用户界面方面，可以选择使用图形界面工具如 https://lmstudio.ai 或命令行界面如 https://www.ollama.com 来完成部署工作。

测试内容包括选择若干编程题目，确保这些题目能够在所部署的LLM上得到正确解答，并通过所有相关的测试用例（即状态为Accepted）。选题应来源于在线判题平台，例如 OpenJudge、Codeforces、LeetCode 或洛谷等，同时需注意避免与已找到的AI接受题目重复。已有的AI接受题目列表可参考以下链接：
https://github.com/GMyhf/2025spring-cs201/blob/main/AI_accepted_locally.md

请提供你的最新进展情况，包括任何关键步骤的截图以及遇到的问题和解决方案。这将有助于全面了解项目的推进状态，并为进一步的工作提供参考。





### Q6. 阅读《Build a Large Language Model (From Scratch)》第一章

作者：Sebastian Raschka

请整理你的学习笔记。这应该包括但不限于对第一章核心概念的理解、重要术语的解释、你认为特别有趣或具有挑战性的内容，以及任何你可能有的疑问或反思。通过这种方式，不仅能巩固你自己的学习成果，也能帮助他人更好地理解这一部分内容。





## 2. 学习总结和个人收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

这次作业主要是中间两个题稍恶心，两道题方法很类似，都是对于某个核心变量二分查找然后检验变量取值是否符合题目要求，跟河中跳房子很像，恶心之处在于没想到要二分，见了几次后下次遇见类似的应该能反应过来了，准备开始补这两周的每日选做了