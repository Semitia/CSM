# CSM

## ToDo

- [x] 代入真实物理参数
- [x] 控制器有点问题
- [ ] 统计实验,分析脚本
- [ ] 曲线
- [ ] 分析实验
- [ ] 计算效率
- [ ] 两种限制
- [ ] 工程结构
- [ ] 大量雅可比矩阵等成员使用直接命名的方式，比较臃肿，尝试优化为高维数据
- [ ] CSM和LineGenerator绘制代码重复，需优化复用
- [ ] display experiment代码重复，需要逐步封装
- [ ] package整理

## 疑惑

1. 雅可比矩阵
![alt text](/imgs/image.png)
等式右边应该是$J_{(tv3)},J_{(t\omega3)}$吧

2. {1b}获取
![alt text](/imgs/image-1.png)
应该是$\hat{z}_w$ 而非 $\hat{z}_s$。

3. J1 & J2
![alt text](/imgs/image-2.png)
![alt text](/imgs/image-3.png)
是$p^{1b}_{2e}$么？

4. 速度雅可比矩阵
![alt text](/imgs/image-5.png)
![alt text](/imgs/image-6.png)
