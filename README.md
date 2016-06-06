# will
* 这个项目包含两个部分：

## crawl
* 一个爬虫，利用bing从网络中提取有用的信息。效果并不太好。

## neural
* 神经网络。利用反向传播算法构造，用于机器学习。目前在设法利用高性能库重构中。
* CBLAS的代码需少量修改才能在VS中使用。主要是出在vnsprintf和stderr的定义不同。
* dll文件在[https://github.com/scarsty/will/issues/6]下载。

### neural-demo
* 演示的神经网络，是按照教科书中的讲解以面向对象方法构造。已停止更新，分离至[https://github.com/scarsty/neural-demo]。
* 所有部分都直接计算，实际上应当使用矩阵计算来优化。
