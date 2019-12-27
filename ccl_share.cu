#include "kernel.h"
#include <cuda_occupancy.h>
#define RAD 1		//local block radius
#define TX 32		//number of threads per block along x-axis 
#define TY 32		//number of threads per block along y-axis
#define TZ 1		//number of threads per block along z-axis

using namespace std;




__device__ int diff(int d1, int d2)
{
	return abs(((d1 >> 16) & 0xff) - ((d2 >> 16) & 0xff)) + abs(((d1 >> 8) & 0xff) - ((d2 >> 8) & 0xff)) + abs((d1 & 0xff) - (d2 & 0xff));
}

//this is Prevent cross-border
__device__ int idxClip(int idx, int idxMax)
{
	//Prevent cross-border
	return idx > (idxMax - 1) ? (idxMax - 1) : (idx < 0 ? 0 : idx);
}

__device__ int flatten(int col, int row, int slice, int w, int h, int z)
{
	return  idxClip(col, w) + idxClip(row, h)*w + idxClip(slice, z)*w*h;
}


//not use share memory
__global__ void init_CCL_26(int *L,
	int * R,
	int w,
	int h,
	int z)
{
	const int col = blockIdx.x*blockDim.x + threadIdx.x;
	const int row = blockIdx.y*blockDim.y + threadIdx.y;
	const int sli = blockIdx.z*blockDim.z + threadIdx.z;
	if ((col >= w) || (row >= h) || (sli >= z))	return;
	const int id = flatten(col, row, sli, w, h, z);
	L[id] = R[id] = id;
}

inline
__device__ void Host2share_26(const uint3 threadx,
	const dim3 blockdim,
	int * origin_DATA,
	int *s_in,
	const int *s_l,
	const int *s_i,
	const int *g_l,
	const int *g_i)
{
	// Resolving overlapping parts
	if (threadIdx.x < RAD&&threadIdx.y < RAD)
	{
		s_in[flatten(s_i[0] - RAD, s_i[1] - RAD, s_i[2], s_l[0], s_l[1], s_l[2])] =
			origin_DATA[flatten(g_i[0] - RAD, g_i[1] - RAD, g_i[2], g_l[0], g_l[1], g_l[2])];
		s_in[flatten(s_i[0] + blockDim.x, s_i[1] - RAD, s_i[2], s_l[0], s_l[1], s_l[2])] =
			origin_DATA[flatten(g_i[0] + blockDim.x, g_i[1] - RAD, g_i[2], g_l[0], g_l[1], g_l[2])];
		s_in[flatten(s_i[0] - RAD, s_i[1] + blockDim.y, s_i[2], s_l[0], s_l[1], s_l[2])] =
			origin_DATA[flatten(g_i[0] - RAD, g_i[1] + blockDim.y, g_i[2], g_l[0], g_l[1], g_l[2])];
		s_in[flatten(s_i[0] + blockDim.x, s_i[1] + blockDim.y, s_i[2], s_l[0], s_l[1], s_l[2])] =
			origin_DATA[flatten(g_i[0] + blockDim.x, g_i[1] + blockDim.y, g_i[2], g_l[0], g_l[1], g_l[2])];
	}
	if (threadIdx.x < RAD&&threadIdx.z < RAD)
	{
		s_in[flatten(s_i[0] - RAD, s_i[1], s_i[2] - RAD, s_l[0], s_l[1], s_l[2])] =
			origin_DATA[flatten(g_i[0] - RAD, g_i[1], g_i[2] - RAD, g_l[0], g_l[1], g_l[2])];
		s_in[flatten(s_i[0] + blockDim.x, s_i[1], s_i[2] - RAD, s_l[0], s_l[1], s_l[2])] =
			origin_DATA[flatten(g_i[0] + blockDim.x, g_i[1], g_i[2] - RAD, g_l[0], g_l[1], g_l[2])];
		s_in[flatten(s_i[0] - RAD, s_i[1], s_i[2] + blockDim.z, s_l[0], s_l[1], s_l[2])] =
			origin_DATA[flatten(g_i[0] - RAD, g_i[1], g_i[2] + blockDim.z, g_l[0], g_l[1], g_l[2])];
		s_in[flatten(s_i[0] + blockDim.x, s_i[1], s_i[2] + blockDim.z, s_l[0], s_l[1], s_l[2])] =
			origin_DATA[flatten(g_i[0] + blockDim.x, g_i[1], g_i[2] + blockDim.z, g_l[0], g_l[1], g_l[2])];
	}
	if (threadIdx.y < RAD&&threadIdx.z < RAD)
	{
		s_in[flatten(s_i[0], s_i[1] - RAD, s_i[2] - RAD, s_l[0], s_l[1], s_l[2])] =
			origin_DATA[flatten(g_i[0], g_i[1] - RAD, g_i[2] - RAD, g_l[0], g_l[1], g_l[2])];
		s_in[flatten(s_i[0], s_i[1] + blockDim.y, s_i[2] - RAD, s_l[0], s_l[1], s_l[2])] =
			origin_DATA[flatten(g_i[0], g_i[1] + blockDim.y, g_i[2] - RAD, g_l[0], g_l[1], g_l[2])];
		s_in[flatten(s_i[0], s_i[1] - RAD, s_i[2] + blockDim.z, s_l[0], s_l[1], s_l[2])] =
			origin_DATA[flatten(g_i[0], g_i[1] - RAD, g_i[2] + blockDim.z, g_l[0], g_l[1], g_l[2])];
		s_in[flatten(s_i[0], s_i[1] + blockDim.y, s_i[2] + blockDim.z, s_l[0], s_l[1], s_l[2])] =
			origin_DATA[flatten(g_i[0], g_i[1] + blockDim.y, g_i[2] + blockDim.z, g_l[0], g_l[1], g_l[2])];
	}
	if (threadIdx.x < RAD) {
		s_in[flatten(s_i[0] - RAD, s_i[1], s_i[2], s_l[0], s_l[1], s_l[2])] =
			origin_DATA[flatten(g_i[0] - RAD, g_i[1], g_i[2], g_l[0], g_l[1], g_l[2])];
		s_in[flatten(s_i[0] + blockDim.x, s_i[1], s_i[2], s_l[0], s_l[1], s_l[2])] =
			origin_DATA[flatten(g_i[0] + blockDim.x, g_i[1], g_i[2], g_l[0], g_l[1], g_l[2])];
	}
	if (threadIdx.y < RAD) {
		s_in[flatten(s_i[0], s_i[1] - RAD, s_i[2], s_l[0], s_l[1], s_l[2])] =
			origin_DATA[flatten(g_i[0], g_i[1] - RAD, g_i[2], g_l[0], g_l[1], g_l[2])];
		s_in[flatten(s_i[0], s_i[1] + blockDim.y, s_i[2], s_l[0], s_l[1], s_l[2])] =
			origin_DATA[flatten(g_i[0], g_i[1] + blockDim.y, g_i[2], g_l[0], g_l[1], g_l[2])];
	}
	if (threadIdx.z < RAD) {
		s_in[flatten(s_i[0], s_i[1], s_i[2] - RAD, s_l[0], s_l[1], s_l[2])] =
			origin_DATA[flatten(g_i[0], g_i[1], g_i[2] - RAD, g_l[0], g_l[1], g_l[2])];
		s_in[flatten(s_i[0], s_i[1], s_i[2] + blockDim.z, s_l[0], s_l[1], s_l[2])] =
			origin_DATA[flatten(g_i[0], g_i[1], g_i[2] + blockDim.z, g_l[0], g_l[1], g_l[2])];
	}
	if (threadIdx.x < RAD&&threadIdx.y < RAD&&threadIdx.z < RAD) {
		s_in[flatten(s_i[0] - RAD, s_i[1] - RAD, s_i[2] + blockDim.z, s_l[0], s_l[1], s_l[2])] 
			= origin_DATA[flatten(g_i[0] - RAD, g_i[1] - RAD, g_i[2] + blockDim.z, g_l[0], g_l[1], g_l[2])];
		s_in[flatten(s_i[0] - RAD, s_i[1] - RAD, s_i[2] - RAD, s_l[0], s_l[1], s_l[2])]
			= origin_DATA[flatten(g_i[0] - RAD, g_i[1] - RAD, g_i[2] - RAD, g_l[0], g_l[1], g_l[2])];
		s_in[flatten(s_i[0] - RAD, s_i[1] + blockDim.y, s_i[2] + blockDim.z, s_l[0], s_l[1], s_l[2])]
			= origin_DATA[flatten(g_i[0] - RAD, g_i[1]+blockDim.y, g_i[2] + blockDim.z, g_l[0], g_l[1], g_l[2])];
		s_in[flatten(s_i[0] - RAD, s_i[1] + blockDim.y, s_i[2] - RAD, s_l[0], s_l[1], s_l[2])]
			= origin_DATA[flatten(g_i[0] - RAD, g_i[1] + blockDim.y, g_i[2] - RAD, g_l[0], g_l[1], g_l[2])];
		s_in[flatten(s_i[0] + blockDim.x, s_i[1] - RAD, s_i[2] - RAD, s_l[0], s_l[1], s_l[2])]
			= origin_DATA[flatten(g_i[0] + blockDim.x, g_i[1] - RAD, g_i[2] - RAD, g_l[0], g_l[1], g_l[2])];
		s_in[flatten(s_i[0] + blockDim.x, s_i[1] + blockDim.y, s_i[2] - RAD, s_l[0], s_l[1], s_l[2])]
			= origin_DATA[flatten(g_i[0] + blockDim.x, g_i[1] + blockDim.y, g_i[2] - RAD, g_l[0], g_l[1], g_l[2])];
		s_in[flatten(s_i[0] + blockDim.x, s_i[1] + blockDim.y, s_i[2] + blockDim.z, s_l[0], s_l[1], s_l[2])]
			= origin_DATA[flatten(g_i[0] + blockDim.x, g_i[1] + blockDim.y, g_i[2] + blockDim.z, g_l[0], g_l[1], g_l[2])];
		s_in[flatten(s_i[0] + blockDim.x, s_i[1] - RAD, s_i[2] + blockDim.z, s_l[0], s_l[1], s_l[2])]
			= origin_DATA[flatten(g_i[0] + blockDim.x, g_i[1] - RAD, g_i[2] + blockDim.z, g_l[0], g_l[1], g_l[2])];
	}
}

inline
__device__ void Host2share_26_C(const uint3 threadx,
	const dim3 blockdim,
	uchar * origin_DATA,
	uchar *s_in,
	const int *s_l,
	const int *s_i,
	const int *g_l,
	const int *g_i)
{
	// Resolving overlapping parts
	if (threadIdx.x < RAD&&threadIdx.y < RAD)
	{
		s_in[flatten(s_i[0] - RAD, s_i[1] - RAD, s_i[2], s_l[0], s_l[1], s_l[2])] =
			origin_DATA[flatten(g_i[0] - RAD, g_i[1] - RAD, g_i[2], g_l[0], g_l[1], g_l[2])];
		s_in[flatten(s_i[0] + blockDim.x, s_i[1] - RAD, s_i[2], s_l[0], s_l[1], s_l[2])] =
			origin_DATA[flatten(g_i[0] + blockDim.x, g_i[1] - RAD, g_i[2], g_l[0], g_l[1], g_l[2])];
		s_in[flatten(s_i[0] - RAD, s_i[1] + blockDim.y, s_i[2], s_l[0], s_l[1], s_l[2])] =
			origin_DATA[flatten(g_i[0] - RAD, g_i[1] + blockDim.y, g_i[2], g_l[0], g_l[1], g_l[2])];
		s_in[flatten(s_i[0] + blockDim.x, s_i[1] + blockDim.y, s_i[2], s_l[0], s_l[1], s_l[2])] =
			origin_DATA[flatten(g_i[0] + blockDim.x, g_i[1] + blockDim.y, g_i[2], g_l[0], g_l[1], g_l[2])];
	}
	if (threadIdx.x < RAD&&threadIdx.z < RAD)
	{
		s_in[flatten(s_i[0] - RAD, s_i[1], s_i[2] - RAD, s_l[0], s_l[1], s_l[2])] =
			origin_DATA[flatten(g_i[0] - RAD, g_i[1], g_i[2] - RAD, g_l[0], g_l[1], g_l[2])];
		s_in[flatten(s_i[0] + blockDim.x, s_i[1], s_i[2] - RAD, s_l[0], s_l[1], s_l[2])] =
			origin_DATA[flatten(g_i[0] + blockDim.x, g_i[1], g_i[2] - RAD, g_l[0], g_l[1], g_l[2])];
		s_in[flatten(s_i[0] - RAD, s_i[1], s_i[2] + blockDim.z, s_l[0], s_l[1], s_l[2])] =
			origin_DATA[flatten(g_i[0] - RAD, g_i[1], g_i[2] + blockDim.z, g_l[0], g_l[1], g_l[2])];
		s_in[flatten(s_i[0] + blockDim.x, s_i[1], s_i[2] + blockDim.z, s_l[0], s_l[1], s_l[2])] =
			origin_DATA[flatten(g_i[0] + blockDim.x, g_i[1], g_i[2] + blockDim.z, g_l[0], g_l[1], g_l[2])];
	}
	if (threadIdx.y < RAD&&threadIdx.z < RAD)
	{
		s_in[flatten(s_i[0], s_i[1] - RAD, s_i[2] - RAD, s_l[0], s_l[1], s_l[2])] =
			origin_DATA[flatten(g_i[0], g_i[1] - RAD, g_i[2] - RAD, g_l[0], g_l[1], g_l[2])];
		s_in[flatten(s_i[0], s_i[1] + blockDim.y, s_i[2] - RAD, s_l[0], s_l[1], s_l[2])] =
			origin_DATA[flatten(g_i[0], g_i[1] + blockDim.y, g_i[2] - RAD, g_l[0], g_l[1], g_l[2])];
		s_in[flatten(s_i[0], s_i[1] - RAD, s_i[2] + blockDim.z, s_l[0], s_l[1], s_l[2])] =
			origin_DATA[flatten(g_i[0], g_i[1] - RAD, g_i[2] + blockDim.z, g_l[0], g_l[1], g_l[2])];
		s_in[flatten(s_i[0], s_i[1] + blockDim.y, s_i[2] + blockDim.z, s_l[0], s_l[1], s_l[2])] =
			origin_DATA[flatten(g_i[0], g_i[1] + blockDim.y, g_i[2] + blockDim.z, g_l[0], g_l[1], g_l[2])];
	}
	if (threadIdx.x < RAD) {
		s_in[flatten(s_i[0] - RAD, s_i[1], s_i[2], s_l[0], s_l[1], s_l[2])] =
			origin_DATA[flatten(g_i[0] - RAD, g_i[1], g_i[2], g_l[0], g_l[1], g_l[2])];
		s_in[flatten(s_i[0] + blockDim.x, s_i[1], s_i[2], s_l[0], s_l[1], s_l[2])] =
			origin_DATA[flatten(g_i[0] + blockDim.x, g_i[1], g_i[2], g_l[0], g_l[1], g_l[2])];
	}
	if (threadIdx.y < RAD) {
		s_in[flatten(s_i[0], s_i[1] - RAD, s_i[2], s_l[0], s_l[1], s_l[2])] =
			origin_DATA[flatten(g_i[0], g_i[1] - RAD, g_i[2], g_l[0], g_l[1], g_l[2])];
		s_in[flatten(s_i[0], s_i[1] + blockDim.y, s_i[2], s_l[0], s_l[1], s_l[2])] =
			origin_DATA[flatten(g_i[0], g_i[1] + blockDim.y, g_i[2], g_l[0], g_l[1], g_l[2])];
	}
	if (threadIdx.z < RAD) {
		s_in[flatten(s_i[0], s_i[1], s_i[2] - RAD, s_l[0], s_l[1], s_l[2])] =
			origin_DATA[flatten(g_i[0], g_i[1], g_i[2] - RAD, g_l[0], g_l[1], g_l[2])];
		s_in[flatten(s_i[0], s_i[1], s_i[2] + blockDim.z, s_l[0], s_l[1], s_l[2])] =
			origin_DATA[flatten(g_i[0], g_i[1], g_i[2] + blockDim.z, g_l[0], g_l[1], g_l[2])];
	}
	if (threadIdx.x < RAD&&threadIdx.y < RAD&&threadIdx.z < RAD) {
		s_in[flatten(s_i[0] - RAD, s_i[1] - RAD, s_i[2] + blockDim.z, s_l[0], s_l[1], s_l[2])]
			= origin_DATA[flatten(g_i[0] - RAD, g_i[1] - RAD, g_i[2] + blockDim.z, g_l[0], g_l[1], g_l[2])];
		s_in[flatten(s_i[0] - RAD, s_i[1] - RAD, s_i[2] - RAD, s_l[0], s_l[1], s_l[2])]
			= origin_DATA[flatten(g_i[0] - RAD, g_i[1] - RAD, g_i[2] - RAD, g_l[0], g_l[1], g_l[2])];
		s_in[flatten(s_i[0] - RAD, s_i[1] + blockDim.y, s_i[2] + blockDim.z, s_l[0], s_l[1], s_l[2])]
			= origin_DATA[flatten(g_i[0] - RAD, g_i[1] + blockDim.y, g_i[2] + blockDim.z, g_l[0], g_l[1], g_l[2])];
		s_in[flatten(s_i[0] - RAD, s_i[1] + blockDim.y, s_i[2] - RAD, s_l[0], s_l[1], s_l[2])]
			= origin_DATA[flatten(g_i[0] - RAD, g_i[1] + blockDim.y, g_i[2] - RAD, g_l[0], g_l[1], g_l[2])];
		s_in[flatten(s_i[0] + blockDim.x, s_i[1] - RAD, s_i[2] - RAD, s_l[0], s_l[1], s_l[2])]
			= origin_DATA[flatten(g_i[0] + blockDim.x, g_i[1] - RAD, g_i[2] - RAD, g_l[0], g_l[1], g_l[2])];
		s_in[flatten(s_i[0] + blockDim.x, s_i[1] + blockDim.y, s_i[2] - RAD, s_l[0], s_l[1], s_l[2])]
			= origin_DATA[flatten(g_i[0] + blockDim.x, g_i[1] + blockDim.y, g_i[2] - RAD, g_l[0], g_l[1], g_l[2])];
		s_in[flatten(s_i[0] + blockDim.x, s_i[1] + blockDim.y, s_i[2] + blockDim.z, s_l[0], s_l[1], s_l[2])]
			= origin_DATA[flatten(g_i[0] + blockDim.x, g_i[1] + blockDim.y, g_i[2] + blockDim.z, g_l[0], g_l[1], g_l[2])];
		s_in[flatten(s_i[0] + blockDim.x, s_i[1] - RAD, s_i[2] + blockDim.z, s_l[0], s_l[1], s_l[2])]
			= origin_DATA[flatten(g_i[0] + blockDim.x, g_i[1] - RAD, g_i[2] + blockDim.z, g_l[0], g_l[1], g_l[2])];
	}
}

inline
__device__ void min_26_nbd(int data,
	uchar* s_in,
	int* L_label,
	const int s_w,
	int *label,
	const int s_index,
	const int  slice_area,
	const int elpise)
{
	/*--------------current slice-----------------------*/
	//up
	if (diff(data, s_in[s_index - s_w]) <= elpise)
		*label = min((int)*label, (int)L_label[s_index - s_w]);
	//down
	if (diff(data, s_in[s_index + s_w]) <= elpise)
		*label = min((int)*label, (int)L_label[s_index + s_w]);
	//left
	if (diff(data, s_in[s_index - 1]) <= elpise)
		*label = min((int)*label, (int)L_label[s_index - 1]);
	//right
	if (diff(data, s_in[s_index + 1]) <= elpise)
		*label = min((int)*label, (int)L_label[s_index + 1]);
	//up left
	if (diff(data, s_in[s_index - s_w - 1]) <= elpise)
		*label = min((int)*label, (int)L_label[s_index - s_w - 1]);
	//up right
	if (diff(data, s_in[s_index - s_w + 1]) <= elpise)
		*label = min((int)*label, (int)L_label[s_index - s_w + 1]);
	//down left
	if (diff(data, s_in[s_index + s_w - 1]) <= elpise)
		*label = min((int)*label, (int)L_label[s_index + s_w - 1]);
	//down right
	if (diff(data, s_in[s_index + s_w + 1]) <= elpise)
		*label = min((int)*label, (int)L_label[s_index + s_w + 1]);
	/*--------------up slice-----------------------*/
	//cur index
	if (diff(data, s_in[s_index - slice_area]) <= elpise)
		*label = min((int)*label, (int)L_label[s_index - slice_area]);
	//up
	if (diff(data, s_in[s_index - slice_area - s_w]) <= elpise)
		*label = min((int)*label, (int)L_label[s_index - slice_area - s_w]);
	//down
	if (diff(data, s_in[s_index - slice_area + s_w]) <= elpise)
		*label = min((int)*label, (int)L_label[s_index - slice_area + s_w]);
	//left
	if (diff(data, s_in[s_index - slice_area - 1]) <= elpise)
		*label = min((int)*label, (int)L_label[s_index - slice_area - 1]);
	//right
	if (diff(data, s_in[s_index - slice_area + 1]) <= elpise)
		*label = min((int)*label, (int)L_label[s_index - slice_area + 1]);
	//up left
	if (diff(data, s_in[s_index - slice_area - s_w - 1]) <= elpise)
		*label = min((int)*label, (int)L_label[s_index - slice_area - s_w - 1]);
	//up right
	if (diff(data, s_in[s_index - slice_area - s_w + 1]) <= elpise)
		*label = min((int)*label, (int)L_label[s_index - slice_area - s_w + 1]);
	//down left
	if (diff(data, s_in[s_index - slice_area + s_w - 1]) <= elpise)
		*label = min((int)*label, (int)L_label[s_index - slice_area + s_w - 1]);
	//down right
	if (diff(data, s_in[s_index - slice_area + s_w + 1]) <= elpise)
		*label = min((int)*label, (int)L_label[s_index - slice_area + s_w + 1]);
	/*--------------down slice-----------------------*/
	//cur index
	if (diff(data, s_in[s_index + slice_area]) <= elpise)
		*label = min((int)*label, (int)L_label[s_index + slice_area]);
	//up
	if (diff(data, s_in[s_index + slice_area - s_w]) <= elpise)
		*label = min((int)*label, (int)L_label[s_index + slice_area - s_w]);
	//down
	if (diff(data, s_in[s_index + slice_area + s_w]) <= elpise)
		*label = min((int)*label, (int)L_label[s_index + slice_area + s_w]);
	//left
	if (diff(data, s_in[s_index + slice_area - 1]) <= elpise)
		*label = min((int)*label, (int)L_label[s_index + slice_area - 1]);
	//right
	if (diff(data, s_in[s_index + slice_area + 1]) <= elpise)
		*label = min((int)*label, (int)L_label[s_index + slice_area + 1]);
	//up left
	if (diff(data, s_in[s_index + slice_area - s_w - 1]) <= elpise)
		*label = min((int)*label, (int)L_label[s_index + slice_area - s_w - 1]);
	//up right
	if (diff(data, s_in[s_index + slice_area - s_w + 1]) <= elpise)
		*label = min((int)*label, (int)L_label[s_index + slice_area - s_w + 1]);
	//down left
	if (diff(data, s_in[s_index + slice_area + s_w - 1]) <= elpise)
		*label = min((int)*label, (int)L_label[s_index + slice_area + s_w - 1]);
	//down right
	if (diff(data, s_in[s_index + slice_area + s_w + 1]) <= elpise)
		*label = min((int)*label, (int)L_label[s_index + slice_area + s_w + 1]);
	/*----------------above------------------------*/
}

inline
__device__ void min_26_nbd_control_bound(
	const int g_l[3],
	const int g_in[3],
	int data,
	uchar* s_in,
	int* L_label,
	const int s_w,
	int *label,
	const int s_index,
	const int  slice_area,
	const int elpise)
{
	/*--------------current slice-----------------------*/
	if (g_in[1] > 0) {
		//up
		if (diff(data, s_in[s_index - s_w]) <= elpise)
			*label = min((int)*label, (int)L_label[s_index - s_w]);
		if (g_in[0] > 0)
		{
			//up left
			if (diff(data, s_in[s_index - s_w - 1]) <= elpise)
				*label = min((int)*label, (int)L_label[s_index - s_w - 1]);
		}
		if (g_in[0] < g_l[0] - 1)
		{
			//up right
			if (diff(data, s_in[s_index - s_w + 1]) <= elpise)
				*label = min((int)*label, (int)L_label[s_index - s_w + 1]);
		}

	}
	if (g_in[1] < g_l[1] - 1) {
		//down
		if (diff(data, s_in[s_index + s_w]) <= elpise)
			*label = min((int)*label, (int)L_label[s_index + s_w]);
		if (g_in[0] > 0)
		{
			//down left
			if (diff(data, s_in[s_index + s_w - 1]) <= elpise)
				*label = min((int)*label, (int)L_label[s_index + s_w - 1]);
		}
		if (g_in[0] < g_l[0] - 1)
		{
			//down right
			if (diff(data, s_in[s_index + s_w + 1]) <= elpise)
				*label = min((int)*label, (int)L_label[s_index + s_w + 1]);
		}
	}
	if (g_in[0] > 0) {
		//left
		if (diff(data, s_in[s_index - 1]) <= elpise)
			*label = min((int)*label, (int)L_label[s_index - 1]);
	}
	if (g_in[0] < g_l[0] - 1) {
		//right
		if (diff(data, s_in[s_index + 1]) <= elpise)
			*label = min((int)*label, (int)L_label[s_index + 1]);
	}
	
	
	/*--------------up slice-----------------------*/
	if (g_in[2] > 0)
	{
		//cur index
		if (diff(data, s_in[s_index - slice_area]) <= elpise)
			*label = min((int)*label, (int)L_label[s_index - slice_area]);
		if (g_in[1] > 0) 
		{
			//up
			if (diff(data, s_in[s_index - slice_area - s_w]) <= elpise)
				*label = min((int)*label, (int)L_label[s_index - slice_area - s_w]);
			if (g_in[0] > 0)
			{
				//up left
				if (diff(data, s_in[s_index - slice_area - s_w - 1]) <= elpise)
					*label = min((int)*label, (int)L_label[s_index - slice_area - s_w - 1]);
			}
			if (g_in[0] < g_l[0] - 1)
			{
				//up right
				if (diff(data, s_in[s_index - slice_area - s_w + 1]) <= elpise)
					*label = min((int)*label, (int)L_label[s_index - slice_area - s_w + 1]);
			}
		}
		if (g_in[1] < g_l[1] - 1) {
			//down
			if (diff(data, s_in[s_index - slice_area + s_w]) <= elpise)
				*label = min((int)*label, (int)L_label[s_index - slice_area + s_w]);
			if (g_in[0] > 0)
			{
				//down left
				if (diff(data, s_in[s_index - slice_area + s_w - 1]) <= elpise)
					*label = min((int)*label, (int)L_label[s_index - slice_area + s_w - 1]);
			}
			if (g_in[0] < g_l[0] - 1)
			{
				//down right
				if (diff(data, s_in[s_index - slice_area + s_w + 1]) <= elpise)
					*label = min((int)*label, (int)L_label[s_index - slice_area + s_w + 1]);
			}

		}
		if (g_in[0] > 0) {
			//left
			if (diff(data, s_in[s_index - slice_area - 1]) <= elpise)
				*label = min((int)*label, (int)L_label[s_index - slice_area - 1]);
		}
		if (g_in[0] < g_l[0] - 1) {
			//right
			if (diff(data, s_in[s_index - slice_area + 1]) <= elpise)
				*label = min((int)*label, (int)L_label[s_index - slice_area + 1]);
		}
		
	}
	
	/*--------------down slice-----------------------*/
	if (g_in[2] < g_l[2] - 1)
	{
		//cur index
		if (diff(data, s_in[s_index + slice_area]) <= elpise)
			*label = min((int)*label, (int)L_label[s_index + slice_area]);
		if (g_in[1] > 0)
		{
			//up
			if (diff(data, s_in[s_index + slice_area - s_w]) <= elpise)
				*label = min((int)*label, (int)L_label[s_index + slice_area - s_w]);
			if (g_in[0] > 0)
			{
				//up left
				if (diff(data, s_in[s_index + slice_area - s_w - 1]) <= elpise)
					*label = min((int)*label, (int)L_label[s_index + slice_area - s_w - 1]);
			}
			if (g_in[0] < g_l[0] - 1)
			{
				//up right
				if (diff(data, s_in[s_index + slice_area - s_w + 1]) <= elpise)
					*label = min((int)*label, (int)L_label[s_index + slice_area - s_w + 1]);
			}
		}
		if (g_in[1] < g_l[1] - 1)
		{
			//down
			if (diff(data, s_in[s_index + slice_area + s_w]) <= elpise)
				*label = min((int)*label, (int)L_label[s_index + slice_area + s_w]);
			if (g_in[0] > 0)
			{
				//down left
				if (diff(data, s_in[s_index + slice_area + s_w - 1]) <= elpise)
					*label = min((int)*label, (int)L_label[s_index + slice_area + s_w - 1]);
			}
			if (g_in[0] < g_l[0] - 1)
			{
				//down right
				if (diff(data, s_in[s_index + slice_area + s_w + 1]) <= elpise)
					*label = min((int)*label, (int)L_label[s_index + slice_area + s_w + 1]);
			}
		}
		if (g_in[0] > 0)
		{
			//left
			if (diff(data, s_in[s_index + slice_area - 1]) <= elpise)
				*label = min((int)*label, (int)L_label[s_index + slice_area - 1]);
		}
		if (g_in[0] < g_l[0] - 1)
		{
			//right
			if (diff(data, s_in[s_index + slice_area + 1]) <= elpise)
				*label = min((int)*label, (int)L_label[s_index + slice_area + 1]);
		}
			
	}
	/*----------------above------------------------*/
}


//3d
__global__
void scanning26(unsigned char* origin_DATA,
	int* L_label,
	int* R_label,
	int w,
	int h,
	int z,
	bool *check)
{
	const int elpise = 1E-4;
	//1000
	__shared__ uchar s_in[(TX + 2)*(TY + 2)*(TZ + 2)];
	__shared__ int s_lb[(TX + 2)*(TY + 2)*(TZ + 2)];

	//global index
	
	const int col = blockIdx.x*blockDim.x + threadIdx.x;
	const int row = blockIdx.y*blockDim.y + threadIdx.y;
	const int sli = blockIdx.z*blockDim.z + threadIdx.z;
	if ((col >= w) || (row >= h) || (sli >= z))	return;
	const int id = flatten(col, row, sli, w, h, z);
	int label = w*h*z;

	const int g_local[3] = { w ,h,z };
	const int g_index[3] = { col ,row,sli };


	//local width and height
	const int s_w = blockDim.x + 2 * RAD;
	const int s_h = blockDim.y + 2 * RAD;
	const int s_z = blockDim.z + 2 * RAD;
	const int slice_area = s_w*s_h;



	//local index
	const int s_col = threadIdx.x + RAD;
	const int s_row = threadIdx.y + RAD;
	const int s_sli = threadIdx.z + RAD;
	const int s_index = flatten(s_col, s_row, s_sli, s_w, s_h, s_z);

	const int s_local[3] = { s_w ,s_h,s_z };
	const int s_ind[3] = { s_col ,s_row,s_sli };
	//global data to share block data
	s_in[s_index] = origin_DATA[id];
	s_lb[s_index] = L_label[id];

	//host data to share
	Host2share_26_C(threadIdx, blockDim, origin_DATA, s_in, s_local, s_ind, g_local, g_index);
	Host2share_26(threadIdx, blockDim, L_label, s_lb, s_local, s_ind, g_local, g_index);

	__syncthreads();
	
	
	//current data
	int data = s_in[s_index];
	__syncthreads();
	//find neighbor min label
	min_26_nbd(data, s_in, s_lb, s_w, &label, s_index, slice_area, elpise);
	__syncthreads();


	if (label < s_lb[s_index]) {
		//atomicMin(&R[L[id]], label);
		R_label[s_lb[s_index]] = label;				//修改的是
		*check = true;
	}
}


__global__ void analysis26(int* L_label,
	int* R_label,
	int w,
	int h,
	int z)

{
	//there is not using share memory
	//global index
	const int col = blockIdx.x*blockDim.x + threadIdx.x;
	const int row = blockIdx.y*blockDim.y + threadIdx.y;
	const int sli = blockIdx.z*blockDim.z + threadIdx.z;
	if ((col >= w) || (row >= h) || (sli >= z))	return;
	const int id = flatten(col, row, sli, w, h, z);

	int label = L_label[id];
	int ref;
	if (label == id) {
		//找到它的局部最小值因为当右边R矩阵记录了领域最小，那么它就会记录上一个最小的最小
		//因此我一直迭代，就找到了局部最小值
		do{ 
			label = R_label[ref = label]; 
		}while (ref ^ label);
		R_label[id] = label;
	}
}

inline
int divUp(int a, int b)
{
	return (a + b - 1) / b;
}

__global__ void labeling26(int* L_label,
	int* R_label,
	int w,
	int h,
	int z)
{
	//global index
	const int col = blockIdx.x*blockDim.x + threadIdx.x;
	const int row = blockIdx.y*blockDim.y + threadIdx.y;
	const int sli = blockIdx.z*blockDim.z + threadIdx.z;
	if ((col >= w) || (row >= h) || (sli >= z))	return;
	const int id = flatten(col, row, sli, w, h, z);

	//此时L[id]的标签还是L[id]，就还是当前最小
	L_label[id] = R_label[R_label[L_label[id]]];
}



int * CCL::cuda_ccl(unsigned char * image,
	int degree_of_connectivity,
	int threshold,
	const int WIDTH,
	const int HEIGHT,
	const int SLICE)
{
	const int Elem = WIDTH*HEIGHT*SLICE;
	cudaError_t Error;
	int * result = (int *)malloc(Elem * sizeof(int));
	uchar *origin_data;
	int *L_label, *R_label;
	Error = cudaMalloc(&origin_data, sizeof(int) * Elem);
	if (Error != cudaSuccess)
		cout << "origin_data cudaMalloc error" << endl;
	Error = cudaMalloc(&L_label, sizeof(int) * Elem);
	if (Error != cudaSuccess)
		cout << "L_label cudaMalloc error" << endl;
	Error = cudaMalloc(&R_label, sizeof(int) * Elem);
	if (Error != cudaSuccess)
		cout << "R_label cudaMalloc error" << endl;

	Error = cudaMemcpy(origin_data, image, sizeof(unsigned char) * Elem, cudaMemcpyHostToDevice);
	if (Error != cudaSuccess)
		cout << "cudaMemcpy error" << endl;

	bool* md;
	Error = cudaMalloc((void**)&md, sizeof(bool));
	if (Error != cudaSuccess)
		cout << "cudaMemcpy error" << endl;
	int blocks;
	int grids;
	dim3 block(TX, TY,TZ);	
	dim3 grid(divUp(WIDTH, TX), divUp(HEIGHT, TY), divUp(SLICE, TZ));
	
	init_CCL_26 << <grid, block >> > (L_label, R_label, WIDTH, HEIGHT, SLICE);
	cudaDeviceSynchronize();
	auto err = cudaGetLastError();
	if (err != cudaSuccess)
		cout << "error" << endl;
	
	const size_t smSz = (TX + 2 * RAD)*(TY + 2 * RAD) *(TZ + 2 * RAD) * sizeof(int);
	for (;;) {
		bool m = false;
		Error = cudaMemcpy(md, &m, sizeof(bool), cudaMemcpyHostToDevice);
		if (Error != cudaSuccess) {
			cout << "cudaMemcpy error" << endl;
		}
		scanning26 << <grid, block >> > (origin_data, L_label, R_label, WIDTH, HEIGHT, SLICE, md);
		err = cudaGetLastError();
		if (err != cudaSuccess)
			cout << "error" << endl;
		cudaMemcpy(&m, md, sizeof(bool), cudaMemcpyDeviceToHost);
		if (Error != cudaSuccess) {
			cout << "cudaMemcpy error" << endl;
		}
		if (m) {
			analysis26 << <grid, block >> >(L_label, R_label, WIDTH, HEIGHT, SLICE);
			err = cudaGetLastError();
			if (err != cudaSuccess)
				cout << "error" << endl;
			//cudaThreadSynchronize();
			labeling26 << <grid, block >> >(L_label, R_label, WIDTH, HEIGHT, SLICE);
			err = cudaGetLastError();
			if (err != cudaSuccess)
				cout << "error" << endl;
		}
		else break;
	}

	cudaMemcpy(result, L_label, Elem * sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaFree(L_label);
	cudaFree(R_label);
	cudaFree(origin_data);
	

	return result;
}
