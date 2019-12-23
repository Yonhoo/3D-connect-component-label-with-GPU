#include "kernel.h"
#include <iostream>
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/opencv.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h" 
#include "itkImageSeriesReader.h"
#include "itkImageSliceConstIteratorWithIndex.h"
#include "itkGrayscaleErodeImageFilter.h"
#include "itkFlatStructuringElement.h"
#include "itkImageLinearIteratorWithIndex.h"
#include "itkGDCMImageIOFactory.h"
#include "itkGrayscaleDilateImageFilter.h"
#include <thread>

using namespace std;


using PixelType = signed int;
constexpr unsigned int Dimension = 3;

using ImageType = itk::Image< PixelType, Dimension >;
using ReaderType = itk::ImageSeriesReader< ImageType >;

using SliceIteratorType = itk::ImageSliceConstIteratorWithIndex<ImageType>;
using ImageIOType = itk::GDCMImageIO;
using NamesGeneratorType = itk::GDCMSeriesFileNames;



struct Args {

	int *out_data;						//itk data
	uchar *in_data;						//opencv data
	const size_t in_slice_data;			//opencv data one slice bytes
	const size_t out_slice_data;		//itk slice data bytes
	const size_t paddedRowBytes;		//copy data bytes

};

struct Args_2d {

	uchar *out_data;					//opencv 3D data
	int *in_data;						//opencv 2D data
	const size_t in_slice_data;			//opencv data  width bytes
	const size_t out_slice_data;		//opencv data  width bytes
	const size_t paddedRowBytes;		//copy data bytes

};

struct Args_text {

	uchar *out_data;					//opencv 3D data
	uchar *in_data;						//opencv 2D data
	const size_t in_slice_data;			//opencv data  width bytes
	const size_t out_slice_data;		//opencv data  width bytes
	const size_t paddedRowBytes;		//copy data bytes

};

void thread_copy(void *args)
{
	struct Args *temp = (struct Args *)args;
	memcpy(temp->in_data + temp->in_slice_data,
		temp->out_data + temp->out_slice_data,
		temp->paddedRowBytes);
	return ;
}

void thread_copy_2d(void *args)
{
	struct Args_2d *temp = (struct Args_2d *)args;
	memcpy(temp->in_data + temp->in_slice_data,
		temp->out_data + temp->out_slice_data,
		temp->paddedRowBytes);
	return;
}

void check_data(ImageType::Pointer image,cv::Mat * MATRIX_)
{
	int my_row = 0;
	int my_col = 0;
	int my_slice = 0;
	SliceIteratorType my_it(image, image->GetRequestedRegion());
	my_it.SetFirstDirection(0);		//axis row
	my_it.SetSecondDirection(1);			//axis col
	my_it.GoToBegin();
	int y = 0;
	while (!my_it.IsAtEnd())
	{
		my_row = 0;
		while (!my_it.IsAtEndOfSlice() && my_row < 512)
		{

			my_col = 0;
			auto* my_sli = MATRIX_->ptr<int>(my_slice, my_row, 0);
			while (!my_it.IsAtEndOfLine() && my_col < 512)
			{
				if (my_sli[my_col] != my_it.Get()) {
					cout << "error" << endl;
					y = 1;
					break;
				}++my_it;
				++my_col;
			}
			if (y == 1)	break;
			++my_row;
			my_it.NextLine();
		}
		my_it.NextSlice();
		if (y == 1)	break;
		++my_slice;
	}
	if (y == 1)	cout << "copy data error" << endl;
	else cout<<"copy data right" << endl;
}

std::shared_ptr<cv::Mat> CCL::Load_itk_series(const char *path)
{
	int my_slice = 0;
	int my_row = 0;
	int my_col = 0;
	
	ImageIOType::Pointer gdcmIO = ImageIOType::New();
	NamesGeneratorType::Pointer namesGenerator = NamesGeneratorType::New();

	namesGenerator->SetInputDirectory(path);
	const ReaderType::FileNamesContainer & filenames = namesGenerator->GetInputFileNames();

	ReaderType::Pointer reader = ReaderType::New();
	reader->SetImageIO(gdcmIO);
	reader->SetFileNames(filenames);

	reader->Update();
	

	//access image 
	ImageType::Pointer image = reader->GetOutput();
	
	ImageType::RegionType haha = image->GetLargestPossibleRegion();
	//3-dim image size
	ImageType::SizeType __size = haha.GetSize();
	if (__size[0] != 512 && __size[1] != 512)
	{
		throw out_of_range("input size must be 512*512 !!!");
	}

	//CV data
	int CT_size[] = { __size[2],512,512, };
	std::shared_ptr<cv::Mat> MATRIX_ = make_shared<cv::Mat>(3, CT_size, CV_32SC1, cv::Scalar(0));
	
	//mutiple thread
	thread *th = new thread[__size[2]];

	size_t paddedRowBytes = sizeof(int) * 512 * 512;
	for (int i = 0; i < __size[2]; i++)
	{
		Args args = { image->GetBufferPointer(),
					MATRIX_->data,
					i*MATRIX_->step[0],
					i* 512 * 512,
					paddedRowBytes
		};
		th[i]=thread(thread_copy, &args);
	}
	for (int i = 0; i < __size[2]; i++)
	{
		th[i].join();
	}
	delete[]th;
	
	//change the data from -1024 to 0 
	cv::Mat t1emp(3, CT_size, CV_32SC1, cv::Scalar(1024));
	/*int *pp = (int*)MATRIX_->dataend;
	int* p = (int*)MATRIX_->data;
	for (; p < pp; p++)(*p) += 1024;*/
	*MATRIX_ = *MATRIX_ + t1emp;
	return MATRIX_;


}

void  load_Series_dicom(const char *file_path)
{
	int my_slice = 0;
	int my_row = 0;
	int my_col = 0;
	
	ImageIOType::Pointer gdcmIO = ImageIOType::New();
	NamesGeneratorType::Pointer namesGenerator = NamesGeneratorType::New();

	namesGenerator->SetInputDirectory(file_path);
	const ReaderType::FileNamesContainer & filenames = namesGenerator->GetInputFileNames();

	ReaderType::Pointer reader = ReaderType::New();
	reader->SetImageIO(gdcmIO);
	reader->SetFileNames(filenames);

	reader->Update();


	//access image 
	//经过查看源码，发现这里应该是数据对象的指针；
	ImageType::Pointer image = reader->GetOutput();

	//only convert 2D image
	//cv::Mat img=itk::OpenCVImageBridge::ITKImageToCVMat<ImageType>(image)


	//pixel value
	ImageType::RegionType haha = image->GetLargestPossibleRegion();
	//3-dim image size
	ImageType::SizeType __size = haha.GetSize();
	if (__size[0] != 512 && __size[1] != 512)
	{
		throw out_of_range("input size must be 512*512 !!!");
	}
	/*----------def size------------------*/
	int CT_size[] = { __size[2],512,512, };
	std::shared_ptr<cv::Mat> MATRIX_ = make_shared<cv::Mat>(3, CT_size, CV_32SC1, cv::Scalar(0));

	SliceIteratorType my_it(image, image->GetRequestedRegion());
	my_it.SetFirstDirection(0);		//axis row
	my_it.SetSecondDirection(1);			//axis col
	my_it.GoToBegin();

	while (!my_it.IsAtEnd())
	{
		my_row = 0;
		while (!my_it.IsAtEndOfSlice() && my_row < 512)
		{

			my_col = 0;
			auto* my_sli = MATRIX_->ptr<int>(my_slice, my_row, 0);
			while (!my_it.IsAtEndOfLine() && my_col < 512)
			{
				my_sli[my_col] = my_it.Get();
				++my_it;
				++my_col;
			}
			++my_row;
			my_it.NextLine();
		}
		my_it.NextSlice();

		++my_slice;
	}
	//change the data from -1024 to 0 
	/*int *pp = (int*)MATRIX_->dataend;
	int* p = (int*)MATRIX_->data;
	for (; p < pp; p++)(*p) += 1024;*/
	return ;


}


std::shared_ptr<cv::Mat> CCL::binary_matrix(
	const cv::Mat &matr,
	signed int isolevel)
{
	int slice = matr.size[0];
	int height = matr.size[1];
	int width = matr.size[2];

	int sz[] = { slice, height,width };

	std::shared_ptr<cv::Mat> bin_martix(new cv::Mat(3, sz, CV_8UC1, cv::Scalar(0)));

	for (auto cut = 0; cut < slice; ++cut)
	{

		for (auto row = 0; row < height; ++row)
		{
			auto my_sli = matr.ptr<signed int>(cut, row, 0);
			for (auto col = 0; col < width; ++col)
			{
				if (my_sli[col] > isolevel) {
					(*bin_martix->ptr<bool>(cut, row, col)) = 1;

				}
			}
		}

	}

	return bin_martix;
}



std::shared_ptr<cv::Mat>  CCL::binary_mat(int *slice_num)
{
	
	std::shared_ptr<cv::Mat> m = Load_itk_series("C:\\00101092\\thin");
	
	std::shared_ptr<cv::Mat> b_m = binary_matrix(*m, 2000);
	
	*slice_num = b_m->size[0];
	
	return b_m;
		
}



unordered_map<int, int> CCL::GPU_conn_26(std::shared_ptr<cv::Mat> b_m,const int slice_num, unordered_map<int, int> &conn_map)
{
	int *in_data=cuda_ccl(b_m->data, 26, 0, b_m->size[1], b_m->size[2], b_m->size[0]);
	

	int temp = 0;
	for (int i = 0; i < slice_num; i++)
	{

		for (int r = 0; r < 512; r++)
		{
			for (int c = 0; c < 512; c++)
			{
				temp = in_data[i * 512 * 512 + r * 512 + c];
				if (temp)	conn_map[temp]++;
			}

		}

	}
	return conn_map;
	
}



void mark_26_conn(
	int label,
	queue<cv::Vec3i> &quad,
	cv::Mat &__matrix,
	cv::Mat &label_matrix,
	cv::Vec3i center,
	vector<cv::Vec3i> &list_connect)
{
	int plane = center[2];
	int row = center[0];
	int col = center[1];
	if (plane - 1 >= 0) {
		auto  _1pxiel = __matrix.ptr<bool>(plane - 1, row, col);
		auto  _1pxiel_L = label_matrix.ptr<int>(plane - 1, row, col);
		if ((*_1pxiel) == 1 && (*_1pxiel_L) == 0) {
			(*_1pxiel_L) = label;
			cv::Vec3i _1p(row, col, plane - 1);
			quad.push(_1p);
			list_connect.push_back(_1p);
		}
	}
	if (plane + 1 < __matrix.size[0]) {
		auto  _3pxiel = __matrix.ptr<bool>(plane + 1, row, col);
		auto  _3pxiel_L = label_matrix.ptr<int>(plane + 1, row, col);
		if ((*_3pxiel) == 1 && (*_3pxiel_L) == 0) {
			(*_3pxiel_L) = label;
			cv::Vec3i _3p(row, col, plane + 1);
			quad.push(_3p);
			list_connect.push_back(_3p);
		}
	}

	if (row - 1 >= 0)
	{
		auto  _4pxiel = __matrix.ptr<bool>(plane, row - 1, col);
		auto  _4pxiel_L = label_matrix.ptr<int>(plane, row - 1, col);
		if ((*_4pxiel) == 1 && (*_4pxiel_L) == 0) {
			(*_4pxiel_L) = label;
			cv::Vec3i _4p(row - 1, col, plane);
			quad.push(_4p);
			list_connect.push_back(_4p);
		}
		if (plane - 1 >= 0)		//��һ��
		{
			auto _0pxiel = __matrix.ptr<bool>(plane - 1, row - 1, col);
			auto _0pxiel_L = label_matrix.ptr<int>(plane - 1, row - 1, col);
			if ((*_0pxiel) == 1 && (*_0pxiel_L) == 0) {
				(*_0pxiel_L) = label;
				cv::Vec3i _0p(row - 1, col, plane - 1);
				quad.push(_0p);
				list_connect.push_back(_0p);
			}
			if (col - 1 >= 0) {
				auto _18pxiel = __matrix.ptr<bool>(plane - 1, row - 1, col - 1);	//�޸ĵ���ԭ����ı�ǩ
				auto _18pxiel_L = label_matrix.ptr<int>(plane - 1, row - 1, col - 1);
				if ((*_18pxiel) == 1 && (*_18pxiel_L) == 0) {
					(*_18pxiel_L) = label;
					cv::Vec3i _18p(row - 1, col - 1, plane - 1);
					quad.push(_18p);
					list_connect.push_back(_18p);
				}
			}
			if (col + 1 < __matrix.size[2]) {
				auto _19pxiel = __matrix.ptr<bool>(plane - 1, row - 1, col + 1);
				auto _19pxiel_L = label_matrix.ptr<int>(plane - 1, row - 1, col + 1);
				if ((*_19pxiel) == 1 && (*_19pxiel_L) == 0) {
					(*_19pxiel_L) = label;
					cv::Vec3i _19p(row - 1, col + 1, plane - 1);
					quad.push(_19p);
					list_connect.push_back(_19p);
				}
			}

		}
		if (plane + 1 < __matrix.size[0])
		{
			auto _2pxiel = __matrix.ptr<bool>(plane + 1, row - 1, col);
			auto _2pxiel_L = label_matrix.ptr<int>(plane + 1, row - 1, col);
			if ((*_2pxiel) == 1 && (*_2pxiel_L) == 0) {
				(*_2pxiel_L) = label;
				cv::Vec3i _2p(row - 1, col, plane + 1);
				quad.push(_2p);
				list_connect.push_back(_2p);
			}
			if (col - 1 >= 0) {
				auto _20pxiel = __matrix.ptr<bool>(plane + 1, row - 1, col - 1);
				auto _20pxiel_L = label_matrix.ptr<int>(plane + 1, row - 1, col - 1);
				if ((*_20pxiel) == 1 && (*_20pxiel_L) == 0) {
					(*_20pxiel_L) = label;
					cv::Vec3i _20p(row - 1, col - 1, plane + 1);
					quad.push(_20p);
					list_connect.push_back(_20p);
				}
			}
			if (col + 1 < __matrix.size[2]) {
				auto _21pxiel = __matrix.ptr<bool>(plane + 1, row - 1, col + 1);
				auto _21pxiel_L = label_matrix.ptr<int>(plane + 1, row - 1, col + 1);
				if ((*_21pxiel) == 1 && (*_21pxiel_L) == 0) {
					(*_21pxiel_L) = label;
					cv::Vec3i _21p(row - 1, col + 1, plane + 1);
					quad.push(_21p);
					list_connect.push_back(_21p);
				}
			}

		}

		if (col - 1 >= 0)
		{
			auto _5pxiel = __matrix.ptr<bool>(plane, row - 1, col - 1);
			auto _5pxiel_L = label_matrix.ptr<int>(plane, row - 1, col - 1);
			if ((*_5pxiel) == 1 && (*_5pxiel_L) == 0) {
				(*_5pxiel_L) = label;
				cv::Vec3i _5p(row - 1, col - 1, plane);
				quad.push(_5p);
				list_connect.push_back(_5p);
			}
		}
		if (col + 1 < __matrix.size[2])
		{
			auto _6pxiel = __matrix.ptr<bool>(plane, row - 1, col + 1);
			auto _6pxiel_L = label_matrix.ptr<int>(plane, row - 1, col + 1);
			if ((*_6pxiel) == 1 && (*_6pxiel_L) == 0) {
				(*_6pxiel_L) = label;
				cv::Vec3i _6p(row - 1, col + 1, plane);
				quad.push(_6p);
				list_connect.push_back(_6p);
			}
		}
	}

	if (row + 1 < __matrix.size[1])
	{
		if (plane - 1 >= 0)		//��һ��
		{
			auto _7pxiel = __matrix.ptr<bool>(plane - 1, row + 1, col);
			auto _7pxiel_L = label_matrix.ptr<int>(plane - 1, row + 1, col);
			if ((*_7pxiel) == 1 && (*_7pxiel_L) == 0) {
				(*_7pxiel_L) = label;
				cv::Vec3i _7p(row + 1, col, plane - 1);
				quad.push(_7p);
				list_connect.push_back(_7p);
			}
			if (col - 1 >= 0) {
				auto _22pxiel = __matrix.ptr<bool>(plane - 1, row + 1, col - 1);
				auto _22pxiel_L = label_matrix.ptr<int>(plane - 1, row + 1, col - 1);
				if ((*_22pxiel) == 1 && (*_22pxiel_L) == 0) {
					(*_22pxiel_L) = label;
					cv::Vec3i _22p(row + 1, col - 1, plane - 1);
					quad.push(_22p);
					list_connect.push_back(_22p);
				}
			}
			if (col + 1 < __matrix.size[2]) {
				auto _23pxiel = __matrix.ptr<bool>(plane - 1, row + 1, col + 1);
				auto _23pxiel_L = label_matrix.ptr<int>(plane - 1, row + 1, col + 1);
				if ((*_23pxiel) == 1 && (*_23pxiel_L) == 0) {
					(*_23pxiel_L) = label;
					cv::Vec3i _23p(row + 1, col + 1, plane - 1);
					quad.push(_23p);
					list_connect.push_back(_23p);
				}
			}
		}
		if (plane + 1 < __matrix.size[0])
		{
			auto _8pxiel = __matrix.ptr<bool>(plane + 1, row + 1, col);
			auto _8pxiel_L = label_matrix.ptr<int>(plane + 1, row + 1, col);
			if ((*_8pxiel) == 1 && (*_8pxiel_L) == 0) {
				(*_8pxiel_L) = label;
				cv::Vec3i _8p(row + 1, col, plane + 1);
				quad.push(_8p);
				list_connect.push_back(_8p);
			}
			if (col - 1 >= 0) {
				auto _24pxiel = __matrix.ptr<bool>(plane + 1, row + 1, col - 1);
				auto _24pxiel_L = label_matrix.ptr<int>(plane + 1, row + 1, col - 1);
				if ((*_24pxiel) == 1 && (*_24pxiel_L) == 0) {
					(*_24pxiel_L) = label;
					cv::Vec3i _24p(row + 1, col - 1, plane + 1);
					quad.push(_24p);
					list_connect.push_back(_24p);
				}
			}
			if (col + 1 < __matrix.size[2]) {
				auto _25pxiel = __matrix.ptr<bool>(plane + 1, row + 1, col + 1);
				auto _25pxiel_L = label_matrix.ptr<int>(plane + 1, row + 1, col + 1);
				if ((*_25pxiel) == 1 && (*_25pxiel_L) == 0) {
					(*_25pxiel_L) = label;
					cv::Vec3i _25p(row + 1, col + 1, plane + 1);
					quad.push(_25p);
					list_connect.push_back(_25p);
				}
			}
		}
		auto _9pxiel = __matrix.ptr<bool>(plane, row + 1, col);
		auto _9pxiel_L = label_matrix.ptr<int>(plane, row + 1, col);
		if ((*_9pxiel) == 1 && (*_9pxiel_L) == 0) {
			(*_9pxiel_L) = label;
			cv::Vec3i _9p(row + 1, col, plane);
			quad.push(_9p);
			list_connect.push_back(_9p);
		}
		if (col - 1 >= 0)
		{
			auto _10pxiel = __matrix.ptr<bool>(plane, row + 1, col - 1);
			auto _10pxiel_L = label_matrix.ptr<int>(plane, row + 1, col - 1);
			if ((*_10pxiel) == 1 && (*_10pxiel_L) == 0) {
				(*_10pxiel_L) = label;
				cv::Vec3i _10p(row + 1, col - 1, plane);
				quad.push(_10p);
				list_connect.push_back(_10p);

			}
		}
		if (col + 1 < __matrix.size[2])
		{
			auto _11pxiel = __matrix.ptr<bool>(plane, row + 1, col + 1);
			auto _11pxiel_L = label_matrix.ptr<int>(plane, row + 1, col + 1);
			if ((*_11pxiel) == 1 && (*_11pxiel_L) == 0) {
				(*_11pxiel_L) = label;
				cv::Vec3i _11p(row + 1, col + 1, plane);
				quad.push(_11p);
				list_connect.push_back(_11p);

			}
		}
	}

	if (col - 1 >= 0)
	{
		if (plane - 1 >= 0)		//��һ��
		{
			auto _12pxiel = __matrix.ptr<bool>(plane - 1, row, col - 1);
			auto _12pxiel_L = label_matrix.ptr<int>(plane - 1, row, col - 1);
			if ((*_12pxiel) == 1 && (*_12pxiel_L) == 0) {
				(*_12pxiel_L) = label;
				cv::Vec3i _12p(row, col - 1, plane - 1);
				quad.push(_12p);
				list_connect.push_back(_12p);
			}
		}
		if (plane + 1 < __matrix.size[0])
		{
			auto _13pxiel = __matrix.ptr<bool>(plane + 1, row, col - 1);
			auto _13pxiel_L = label_matrix.ptr<int>(plane + 1, row, col - 1);
			if ((*_13pxiel) == 1 && (*_13pxiel_L) == 0) {
				(*_13pxiel_L) = label;
				cv::Vec3i _13p(row, col - 1, plane + 1);
				quad.push(_13p);
				list_connect.push_back(_13p);
			}
		}
		auto _14pxiel = __matrix.ptr<bool>(plane, row, col - 1);
		auto _14pxiel_L = label_matrix.ptr<int>(plane, row, col - 1);
		if ((*_14pxiel) == 1 && (*_14pxiel_L) == 0) {
			(*_14pxiel_L) = label;
			cv::Vec3i _14p(row, col - 1, plane);
			quad.push(_14p);
			list_connect.push_back(_14p);
		}
	}

	if (col + 1 < __matrix.size[2])
	{
		if (plane - 1 >= 0)		//��һ��
		{
			auto _15pxiel = __matrix.ptr<bool>(plane - 1, row, col + 1);
			auto _15pxiel_L = label_matrix.ptr<int>(plane - 1, row, col + 1);
			if ((*_15pxiel) == 1 && (*_15pxiel_L) == 0) {
				(*_15pxiel_L) = label;
				cv::Vec3i _15p(row, col + 1, plane - 1);
				quad.push(_15p);
				list_connect.push_back(_15p);

			}
		}
		if (plane + 1 < __matrix.size[0])
		{
			auto _16pxiel = __matrix.ptr<bool>(plane + 1, row, col + 1);
			auto _16pxiel_L = label_matrix.ptr<int>(plane + 1, row, col + 1);
			if ((*_16pxiel) == 1 && (*_16pxiel_L) == 0) {
				(*_16pxiel_L) = label;
				cv::Vec3i _16p(row, col + 1, plane + 1);
				quad.push(_16p);
				list_connect.push_back(_16p);

			}
		}
		auto _17pxiel = __matrix.ptr<bool>(plane, row, col + 1);
		auto _17pxiel_L = label_matrix.ptr<int>(plane, row, col + 1);
		if ((*_17pxiel) == 1 && (*_17pxiel_L) == 0) {
			(*_17pxiel_L) = label;
			cv::Vec3i _17p(row, col + 1, plane);
			quad.push(_17p);
			list_connect.push_back(_17p);
		}
	}
}


void Find_Connect_3D(
	cv::Mat &__matrix,
	cv::Mat &label_matrix,
	vector<vector<cv::Vec3i>> &map_connect,
	int connec_num)
{
	queue<cv::Vec3i> quad;
	int _slice = __matrix.size[0];
	int height = __matrix.size[1];
	int width = __matrix.size[2];
	int Label = 1;
	for (auto cut = 0; cut < _slice; ++cut)
	{
		for (auto row = 0; row < height; ++row)
		{
			auto my_sli = __matrix.ptr<bool>(cut, row, 0);
			auto my_lab = label_matrix.ptr<int>(cut, row, 0);
			for (auto col = 0; col < width; ++col)
			{
				auto &current = my_sli[col];
				auto &cur_lab = my_lab[col];
				cv::Vec3i cur(row, col, cut);
				if (cur_lab == 0 && current == 1)
				{

					cur_lab = Label;
					quad.push(cur);
					vector<cv::Vec3i> *coord = new vector<cv::Vec3i>;
					coord->push_back(cur);
					while (!quad.empty())
					{
						cv::Vec3i curr_center = quad.front();
						quad.pop();
						mark_26_conn(Label, quad, __matrix, label_matrix, curr_center, *coord);

					}
					map_connect.push_back(*coord);

					Label++;
				}
			}
		}
	}


}


void CCL::CPU_conn_26(std::shared_ptr<cv::Mat> mat_data, vector<vector<cv::Vec3i>> &result)
{
	int _slice = mat_data->size[0];
	int height = mat_data->size[1];
	int width = mat_data->size[2];
	int sz[] = { _slice,height,width };
	cv::Mat *L_mat = new cv::Mat(3, sz, CV_8UC1, cv::Scalar(0));
	Find_Connect_3D(*mat_data, *L_mat, result, 26);
}


void CCL::conn_check(unordered_map<int, int> &gpu_conn, unordered_map<int, int> &cpu_conn)
{
	cout << "GPU size    "<<gpu_conn.size() << endl;
	for (auto it:gpu_conn)
	{
		if (it.second > 100)	cout << it.second << endl;
	}
	cout << "CPU size    " << cpu_conn.size() << endl;
	for (auto it : cpu_conn)
	{
		if (it.second > 100)	cout << it.second << endl;
	}
}


void CCL::git_3d_conn(uchar* label_data, unordered_map<int, int> &cpu_conn,int size[3])
{
	int *label_ = cc3d::connected_components3d_26<uchar, int>(label_data, size[0], size[1], size[2], size[1]*size[2]*size[0]);

	for (int s = 0; s < size[0]; s++)
	{
		for (int r = 0; r < size[1]; r++)
		{
			for (int c = 0; c < size[2]; c++)
			{
				if (*(label_ + s* size[1] * size[2] + r* size[2] + c)) {
					cpu_conn[*(label_ + s* size[1] * size[2] + r* size[2] + c)]++;
				}
			}
		}
	}
}




