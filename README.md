# 3D-connect-component-label-with-GPU
This is my 3D parallel algorithm, but it is not perfect. It provides a way for you to optimize it. I hope someone can give a good opinion


由于医学项目中需要实现3D连通域算法 
	 matlab有自带的3D连通域算法，挺快的
	 但是要实现c++版本，2d连通域实现的算法思路上就是two pass method，
	 当时我为了图简单，就直接写了一个广度搜索的3d连通域算法，可想而知，时间上会很慢
	 于是，我想能不能实现一个并行的3D连通域算法呢，于是我google，发现了几篇很好的with gpu的2D CCL，但并没有3D Parallel，
	 download了下来，发现主要思想还是two pass method，不过因为并行，所以需要考虑每个像素点的独立，在这里进行了处理
	 论文分别是：
		Parallel graph component labelling with GPUs and CUDA ----K.A. Hawick
		Connected component labeling on a 2D grid using CUDA ----Oleksandr Kalentev
		Connected Component Labeling in CUDA ----Ondrej Stava
		An Improved Parallel Connected Component Labeling Algorithm and Its GPU Implementation ----WANG Ze-Huan
		论文我都仔细看了一下，由于我没有接触过cuda编程，因此我是从学习cuda编程开始
	这两本书给了我很大的帮助：
		CUDA C编程权威指南
		CUDA 高性能并行计算
	由于我给自己两个星期的时间，因此，我利用第一本书大体上学习了gpu的架构和理论部分，然后利用第二本进行简单的实践
	我花了一个星期的时间学习了2D Parallel 算法和gpu 知识，然后写完了代码，但是gpu的调试，不简单，

工具：
		vs2015+cuda8.0+nsight visual studio

算法思路：
		由于我是刚入门cuda编程，且时间有限，所以我并没有尝试每一篇论文的算法3D实现
		我选择了Oleksandr Kalentev 的算法，比较简单，其它几篇都有相应的优化，我暂时没有做
	1、 将图片进行初始化标记，每个前景区域点标记为像素点索引值，这个很重要，这样把每个像素点当作一个独立的线程，就不需要考虑相互依赖的关系，如下图：
<center class="half">
                      					<img src="https://github.com/Yonhoo/3D-connect-component-label-with-GPU/blob/master/image/image.png" width="200"/>
</center>  
        2、 主循环内进行scann分析。每个 GPU 线程对应一个像素点，首先判断是否为前景点， 如果是，在每个前景点周围 26 or 18 or 6 邻域内搜索，将此像素点周围 26 or 18 or 6邻域内（包括自身）所有的像素点 中最小的标记值赋值给此像素点作为标记，如下图所示。(此展示的是2维的情况)
<center class="half">
                     <img src="https://github.com/Yonhoo/3D-connect-component-label-with-GPU/blob/master/image/1577113195(1).png" width="200"/>
</center>    
        3、 analysis环节：每个 GPU 线程对应一个像素点，将此像素点标记值进行迭代，找到其局部最小的根节点(即最小索引值如下图：
<center class="half">
                     <img src="https://github.com/Yonhoo/3D-connect-component-label-with-GPU/blob/master/image/1577113321(1).png" width="200"/>
</center>  
        4、 设置标志，每次新的循环中如果scann环节更新了图中的标记，将标志置为 1，直到在新的循环中 不再更新标记，scann结束

缺点：
	很明显，这里有个缺点就是得到的标签矩阵并不是从1开始的标记值，并且是不连续的，他们都只是局部最小的索引值
	而且，并没有每个连通域的size
	
我的实现优化部分：
	1.使用共享内存
	我们知道26连通域，需要判断原始矩阵中当前像素点的值和领域的值是否相同，这就会频繁的访问全局内存，而每一个block到全局内存
是有一定距离的，这样就会增加来回访问的次数，而我们知道CT文件一般都是512*512*200左右的像素点，block最多只能放下1024个线程，当然算力越高，这个值会高一点，所以需要很多个block，因此增加了访问全局内存的时间，而每一个block都有一个共享内存，一个block里面的线程离共享内存的距离很近，因此我就创建了一个block的共享内存，减少访问全局内存的时间，当然需要注意边界问题
	
	


结果比较：
	这里通过我的GPU Parallel 算法，可以得到时间为2.799，比我之前写的CPU的广度搜索快了接近3倍
	但是我第一次写的GPU算法还是有一点不足的情况，就是我比较了github上一个star最多的人写的CPU  two pass method 3D CCL算法比他写的优化版本，慢了0.9秒，这里我猜测一个是我需要进行cpu和gpu的切换，浪费了一点，然后grid和block的设置还是差强人意，然后就是还没学会使用gpu性能分析工具来优化我的cuda代码
	矩阵的数据是一维的，但我的block设置是(32,32,1),grid(divUp(WIDTH, TX), divUp(HEIGHT, TY), divUp(SLICE, TZ));也就是我的block是二维的，grid是3维的

注意：
	这里我刚开始写完的时候，出了一点错误，但是调试却很困难，因为数据量太大，于是我就实验性的用8*8*8的数据进行测试，然后修改了一点错误，但是还有一点错误，弄了一天，终于解决了，需要提醒的是cuda编程，调试的话比较困难，最好是在运行前用肉眼调试，看看自己的代码逻辑和小细节是否有错误

待改进之处：
	实现连续标记值+得到每个连通域的size；
	我想的是，为了避免在cpu上一个一个的找，我可以在gpu执行过程中记录相应的根节点，并排序
	然后通过矩阵相乘，比如，有个局部最小根节点573，那么就用一列和标记矩阵相乘，某一行是573的为1，否则为0，
	可以用opencv的bitwise_xor函数，这是我的想法，当然我没有直接实现它
