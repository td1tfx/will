# WILL
<img src='https://raw.githubusercontent.com/scarsty/will/master/logo-will.png' width = '20%'/>
* <https://scarsty.gitbooks.io/will/content/>
* 这个项目包含两个部分：

## crawl
* 一个爬虫，利用bing从网络中提取有用的信息。效果并不太好。

## net
* 神经网络。利用反向传播算法构造，用于机器学习。
* 支持CPU和GPU运算。
* GPU模式下使用cuDNN，cuBlas。CPU模式下使用Blas，建议使用OpenBlas。
* dll文件尺寸较大，未包含在工程中，请在<https://github.com/scarsty/will/issues/6>下载。

### neural-demo
* 演示的神经网络，是按照教科书中的讲解以面向对象方法构造。已停止更新，分离至<https://github.com/scarsty/neural-demo>。
* 所有部分都直接计算，实际上应当使用矩阵计算来优化。
