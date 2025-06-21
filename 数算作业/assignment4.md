# Assignment #4: 位操作、栈、链表、堆和NN

Updated 1203 GMT+8 Mar 10, 2025

2025 spring, Complied by <mark>蔡东辰、工学院·</mark>



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

### 136.只出现一次的数字

bit manipulation, https://leetcode.cn/problems/single-number/



<mark>请用位操作来实现，并且只使用常量额外空间。</mark>



代码：

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        a = 0
        for i in nums:
            a ^= i
        return a
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250316093223663](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250316093223663.png)



### 20140:今日化学论文

stack, http://cs101.openjudge.cn/practice/20140/



思路：

先用一个栈result把元素存进去，遇到']'就出栈，直到遇见'['再停止出栈，把出栈的元素存进tem，翻转回原来的顺序，找到开头的数字n，把后面的字母重复n遍后重新放入result，重复这些操作直到遍历完字符串s中所有元素

代码：

```python
s = input()
letter = 'abcdefghijklmnopqrstuvwxyz'
result = []
tem = []
for i in s:
    if i != ']':
        result.append(i)
    else:
        while True:
            a = result.pop()
            if a == '[':
                break
            tem.append(a)
        tem = tem[::-1]
        n = 0
        for j in range(1,4):
            if tem[j] in letter:
                n = int(''.join(tem[:j]))
                tem = tem[j:]
                break
        result.extend(tem*n)
        tem = []
print(''.join(result))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250316100059873](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250316100059873.png)



### 160.相交链表

linked list, https://leetcode.cn/problems/intersection-of-two-linked-lists/



思路：

参考了题解，一个指针先走A后走B，另一个指针先走B后走A，最后一定会在公共部分重合

代码：

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        nowA = headA
        nowB = headB
        while nowA != nowB:
            nowA = nowA.next
            nowB = nowB.next
            if nowA == None and nowB != None:
                nowA = headB
            if nowB == None and nowA != None:
                nowB = headA
        return nowA
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250316105710390](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250316105710390.png)



### 206.反转链表

linked list, https://leetcode.cn/problems/reverse-linked-list/



思路：

不断改变链表下一个节点的值

代码：

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        cur = head
        pre = None
        while cur:
            tmp = cur.next
            cur.next = pre
            pre = cur
            cur = tmp
        return pre
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250316115534454](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250316115534454.png)



### 3478.选出和最大的K个元素

heap, https://leetcode.cn/problems/choose-k-elements-with-maximum-sum/



思路：

先升序排nums1中的元素，如果nums1中两个元素相同，那么对应的结果也相同，再用一个堆去找最大的k个数之和

代码：

```python
class Solution:
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        n = len(nums1)
        nums = [(nums1[i],nums2[i],i) for i in range(n)]
        nums.sort()
        result = [0]*n
        nums3 = []
        sum = 0
        cnt = 0
        for i in range(n):
            if i > 0 and nums[i][0] == nums[i - 1][0]:
                result[nums[i][2]] = result[nums[i - 1][2]]
            else:
                result[nums[i][2]] = sum
            heappush(nums3,nums[i][1])
            sum += nums[i][1]
            cnt += 1
            if cnt > k:
                cnt -= 1
                sum -= heappop(nums3)
        return result
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250316135222287](C:\Users\15245\AppData\Roaming\Typora\typora-user-images\image-20250316135222287.png)



### Q6.交互可视化neural network

https://developers.google.com/machine-learning/crash-course/neural-networks/interactive-exercises

**Your task:** configure a neural network that can separate the orange dots from the blue dots in the diagram, achieving a loss of less than 0.2 on both the training and test data.

**Instructions:**

In the interactive widget:

1. Modify the neural network hyperparameters by experimenting with some of the following config settings:
   - Add or remove hidden layers by clicking the **+** and **-** buttons to the left of the **HIDDEN LAYERS** heading in the network diagram.
   - Add or remove neurons from a hidden layer by clicking the **+** and **-** buttons above a hidden-layer column.
   - Change the learning rate by choosing a new value from the **Learning rate** drop-down above the diagram.
   - Change the activation function by choosing a new value from the **Activation** drop-down above the diagram.
2. Click the Play button above the diagram to train the neural network model using the specified parameters.
3. Observe the visualization of the model fitting the data as training progresses, as well as the **Test loss** and **Training loss** values in the **Output** section.
4. If the model does not achieve loss below 0.2 on the test and training data, click reset, and repeat steps 1–3 with a different set of configuration settings. Repeat this process until you achieve the preferred results.

给出满足约束条件的<mark>截图</mark>，并说明学习到的概念和原理。





## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

第一题不看题解确实很难想到用位操作弄，今日化学论文和反转链表属于是能想到怎么弄但是写起代码来觉得比较吃力，相交链表

确实是不看题解不会写，第五题主要是维护堆这一步比较关键，不然很容易就超时了

感觉作业里面链表相关的题写起来还是手生，最近在缓慢补之前的每日选做
