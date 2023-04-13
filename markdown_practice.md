# Markdown练习
## 标题有6级
### 几个井号即几级标题
#### 每个操作后面必须有空格

### 引用
> 大于符号

### 有序列表
1. 数字加点
2. 点后也有空格
3. 第三步

### 无序列表
- 短横线
* 或者星号

### 任务列表
- [ ] 短横线和方括号
- [ ] 短横线与方括号之间有空格
- [ ] 括号里面也要空格
- [x] 括号加x就能显示勾

### 代码块
```c
int main(){
  return 0;  // 三个反引号围住（制表符上面那个）+ 代码语言
}
```

### 数学公式
使用latex，四个美元符号括起来，且要空行

$$ \frac{\partial f}{\partial x} = 2\sqrt{a}x $$

### 表格
表头1|表头2|表头3
|:---|---:|:---:|
|冒号在左|冒号在右|冒号两边都有|
|左对齐|右对齐|居中对齐|

### 脚注
脚注[^一个脚注]

[^一个脚注]:方括号，前面一个^符号

### 横线
---

### 链接
[Markdown练习](https://github.com/MyJoyfulLove/Lattice-Based-QIM/edit/main/markdown_practice.md "鼠标放上去有说明")

### 引用链接
[Lattice][id],[Lattice][id],

[Lattice][id],[Lattice][id]

[id]:https://github.com/MyJoyfulLove/Lattice-Based-QIM "通过修改id，即可修改所有链接"

### 跳转
回到[标题](#Markdown练习 "本文内部需要加井号")

