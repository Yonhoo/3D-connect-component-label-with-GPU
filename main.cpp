/*
	 *	2019/12/23
	 *	@Yonhoo
	 *	3D connect component label with GPU
		这里只给出26连通域的并行算法，18联通和6联通只需要修改找领域函数即可
	 */
	
#include "kernel.h"
#include <map>
using namespace std;
int main()
{
	
	CCL api;
	int tem = 9;
	int *slice_num=&tem;
	
	std::shared_ptr<cv::Mat> b_m = api.binary_mat(slice_num);

	std::unordered_map<int, int> cpu_conn_map;


	int sz[] = { b_m->size[1],b_m->size[2],b_m->size[0] };

	

	int num = *slice_num;
	//vector<vector<cv::Vec3i>> cpu_conn_map;
	std::unordered_map<int,int> gpu_conn_map;
	double start = get_time();
	api.GPU_conn_26(b_m, num, gpu_conn_map);
	double end = get_time();
	cout <<"GPU time:  " <<end - start << endl;
	start = get_time();
	api.git_3d_conn(b_m->data, cpu_conn_map, sz);
	//api.CPU_conn_26(b_m, cpu_conn_map);
	end = get_time();
	cout << "CPU time:  " << end - start << endl;
	api.conn_check(gpu_conn_map, cpu_conn_map);
	



	system("pause");
}


