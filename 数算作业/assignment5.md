# Assignment #5: 链表、栈、队列和归并排序

Updated 1348 GMT+8 Mar 17, 2025

2025 spring, Complied by <mark>同学的姓名、院系</mark>



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

### LC21.合并两个有序链表

linked list, https://leetcode.cn/problems/merge-two-sorted-lists/

思路：

new_list为已经连接好的新链表的最后一个节点，它属于原先的其中一个链表，而另一个链表中还没进入新链表的第一个节点记作another，通过比较new_list与another的大小来判断新链表下一个节点为哪个，改变new_list.next之后把another变为另一个链表未进入新链表的第一个节点，如此循环直至其中一个链表所有节点都进入新链表，new_list.next为None，再把another接到new_list后面就行了

代码：

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        head1,head2 = list1,list2
        if not head1:
            return head2
        if not head2:
            return head1
        if head1.val <= head2.val:
            result = head1
            new_list = head1
            another = head2
        else:
            result = head2
            new_list = list2
            another = list1
        while new_list and another:
            tmp = new_list.next
            if tmp and tmp.val <= another.val:
                new_list = tmp
            else:
                new_list.next = another
                another = tmp
        return result
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250318143756416](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250318143756416.png)



### LC234.回文链表

linked list, https://leetcode.cn/problems/palindrome-linked-list/

<mark>请用快慢指针实现。</mark>



代码：

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        def eq(a,b):
            while a:
                if a.val == b.val:
                    a = a.next
                    b = b.next
                else:
                    return False
            return True
        slow = head
        fast = head
        if slow.next == None:
            return True
        pre = None
        while fast:
            if fast.next == None:
                return eq(pre,slow.next)
            fast = fast.next.next
            tmp = slow.next
            slow.next = pre
            pre = slow
            slow = tmp
        return eq(pre,slow)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250318155021195](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250318155021195.png)



### LC1472.设计浏览器历史记录

doubly-lined list, https://leetcode.cn/problems/design-browser-history/

<mark>请用双链表实现。</mark>



代码：

```python
class page:
    def __init__(self,page):
        self.name = page
        self.next1 = None
        self.next2 = None
class BrowserHistory:
    def __init__(self, homepage: str):
        self.now_page = page(homepage)
    def visit(self, url: str) -> None:
        newpage = page(url)
        if self.now_page.next1 != None:
            self.now_page.next1.next2 = None
        self.now_page.next1 = newpage
        newpage.next2 = self.now_page
        self.now_page = newpage
    def back(self, steps: int) -> str:
        for i in range(steps):
            if self.now_page.next2 == None:
                break
            self.now_page = self.now_page.next2
        return self.now_page.name
    def forward(self, steps: int) -> str:
        for i in range(steps):
            if self.now_page.next1 == None:
                break
            self.now_page = self.now_page.next1
        return self.now_page.name

# Your BrowserHistory object will be instantiated and called as such:
# obj = BrowserHistory(homepage)
# obj.visit(url)
# param_2 = obj.back(steps)
# param_3 = obj.forward(steps)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250318171556415](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250318171556415.png)



### 24591: 中序表达式转后序表达式

stack, http://cs101.openjudge.cn/practice/24591/

思路：

用一个栈储存运算符以及括号，按照各种运算的优先级将他们出栈添加到输出中，不过因为各种小bug调代码调到崩溃。。。

代码：

```python
precedence = dict()
precedence['+'] = 1
precedence['-'] = 1
precedence['*'] = 2
precedence['/'] = 2
n = int(input())
for _ in range(n):
    s = input().strip()
    output = []
    op = []
    num = ''
    for i in s:
        if i in '1234567890.':
            num += i
        else:
            if num:
                num = float(num)
                output.append(int(num) if num == int(num) else num)
                num = ''
            if i == '(':
                op.append(i)
            elif i == ')':
                while True:
                    a = op.pop()
                    if a == '(':
                        break
                    output.append(a)
            elif i in '+-*/':
                while op and op[-1] != '(' and precedence[op[-1]] >= precedence[i]:
                    a = op.pop()
                    output.append(a)
                op.append(i)
    if num:
        num = float(num)
        output.append(int(num) if num == int(num) else num)
    while op:
        a = op.pop()
        output.append(a)
    output = [str(i) for i in output]
    print(' '.join(output))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250322204811121](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250322204811121.png)



### 03253: 约瑟夫问题No.2

queue, http://cs101.openjudge.cn/practice/03253/

<mark>请用队列实现。</mark>



代码：

```python
from collections import deque
while True:
    n,p,m = map(int,input().split())
    if n == p == m == 0:
        break
    children = deque([i for i in range(p,n + 1)] + [i for i in range(1,p)])
    out = []
    while children:
        for i in range(m - 1):
            child = children.popleft()
            children.append(child)
        out_child = children.popleft()
        out.append(out_child)
    out = [str(i) for i in out]
    print(','.join(out))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250322232828079](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250322232828079.png)



### 20018: 蚂蚁王国的越野跑

merge sort, http://cs101.openjudge.cn/practice/20018/

思路：

进行归并排序时记录下原本排序的“顺序数”，归并时对于左侧子序列中每一个元素，记下右侧子序列中比它大的元素个数，最后把它们相加再加上左右两个子序列的顺序数，就是整个序列的顺序数

代码：

```python
N = int(input())
ant = []
for i in range(N):
    ant.append(int(input()))
def mergesort(arr):
    if len(arr) <= 1:
        return (0,arr)
    n = len(arr)
    mid = n // 2
    left = arr[:mid]
    right = arr[mid:]
    (l,left) = mergesort(left)
    (r,right) = mergesort(right)
    i = j = 0
    cnt = 0
    result = []
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
            cnt += len(right) - j
        else:
            result.append(right[j])
            j += 1
    if i < len(left):
        result.extend(left[i:])
    if j < len(right):
        result.extend(right[j:])
    return (l + r + cnt,result)
print(mergesort(ant)[0])
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250323113132904](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250323113132904.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

这次作业感觉难度还行，中序转后序那题自己没想出怎么弄，其它题倒是都独立把代码写出来了（借助了tag的提示），感觉蚂蚁王国这题这种递归思想非常有意思，OOP写起来就像在创造一种新的运算也很好玩









