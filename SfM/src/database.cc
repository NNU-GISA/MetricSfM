// ObjectSfM - Object Based Structure-from-Motion.
// Copyright (C) 2018  Ohio State University, CEGE, GDA group
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef MAX_
#define MAX_(a,b) ( ((a)>(b)) ? (a):(b) )
#endif // !MAX

#include "Database.h"

#include <fstream> 
#include <filesystem>
#include <string>
#include <omp.h>

#include "utils/exif_reader.h"
#include "utils/basic_funcs.h"
#include "utils/gist.hpp"

#include "feature/feature_extractor_vl_sift.h"
#include "feature/feature_extractor_opencv.h"
#include "feature/feature_extractor_cuda_sift.h"

#include "imagebase.h"
#include "utils/converter_utm_latlon.h"
#include "utils/local_orientation.h"
#include "utils/lsd.h"
#include "utils/find_polynomial_roots_companion_matrix.h"

namespace objectsfm {

Database::Database()
{
	image_formats_ = {"jpg", "png", "bmp", "tiff", "JPG"};
}

Database::~Database()
{
}

bool Database::FeatureExtraction()
{
	// search all the images in the input fold
	SearchImagePaths();
	if (!image_paths_.size())
	{
		return false;
	}

	int num_imgs = image_paths_.size();

	std::vector<int> idx_missing_feature;
	if (CheckFeatureIndexExist())
	{
		idx_missing_feature = CheckMissingFeatureFile();
	}
	else
	{
		idx_missing_feature.resize(num_imgs, 0);
		for (size_t i = 0; i < num_imgs; i++)
		{
			idx_missing_feature[i] = i;
		}
	}
	if (idx_missing_feature.size())
	{
		std::string file_feature = output_fold_ + "//feature_index.txt";
		std::ofstream ofs(file_feature, std::ios::app);

		for (size_t i = 0; i < idx_missing_feature.size(); i++)
		{
			int idx = idx_missing_feature[i];
			ExtractImageFeatures(idx);
			WriteoutImageFeature(idx);

			ReleaseImageFeatures(idx);
			ofs << idx << std::endl;
		}
		ofs.close();

		// write gist feature out
		if (options.extract_gist)
		{
			WriteoutGistFeature();
			for (size_t i = 0; i < num_imgs_; i++) {
				std::vector<float>().swap(gist_descriptors_[i]);
			}
		}
	}

	// done
	std::cout << "---finish feature extraction" << std::endl;

	return true;
}

void Database::SearchImagePaths()
{
	for (size_t i = 0; i < image_formats_.size(); i++)
	{
		std::vector<cv::String> image_paths_temp;
		cv::glob(input_fold_+"/*."+ image_formats_[i], image_paths_temp, false);

		for (size_t j = 0; j < image_paths_temp.size(); j++)
		{
			image_paths_.emplace_back(std::string(image_paths_temp[j].c_str()));
		}
	}

	num_imgs_ = image_paths_.size();
	image_infos_.resize(num_imgs_);
	keypoints_.resize(num_imgs_);
	descriptors_.resize(num_imgs_);
	if (options.extract_gist)
	{
		gist_descriptors_.resize(num_imgs_);
	}
	words_fbow_.resize(num_imgs_);
	words_vector_.resize(num_imgs_);
	words_id_.resize(num_imgs_);
	for (size_t i = 0; i < num_imgs_; i++)
	{
		words_fbow_[i] = NULL;
		words_vector_[i] = NULL;
	}
}

bool Database::CheckFeatureIndexExist()
{
	std::string path = output_fold_ + "//feature_index.txt";
	std::ifstream inf(path);
	if (!inf.good())
	{
		return false;
	}

	return true;
}

std::vector<int> Database::CheckMissingFeatureFile()
{
	std::string path = output_fold_ + "//feature_index.txt";
	std::ifstream infile(path);

	std::vector<int> index(num_imgs_, 0);
	int idx = -1;
	while (!infile.eof())
	{
		infile >> idx;
		if (idx >= 0)
		{
			index[idx] = 1;
		}
	}

	std::vector<int> missing_idx;
	for (size_t i = 0; i < num_imgs_; i++)
	{
		if (!index[i])
		{
			missing_idx.push_back(i);
		}
	}

	return missing_idx;
}

bool Database::ExtractImageInfo(int idx)
{
	std::string path_ = image_paths_[idx];
	if (path_.empty())
	{
		std::cout << "Empty path!" << std::endl;
		return false;
	}

	// read in the exif
	std::tr2::sys::path path_full(path_);
	std::string filename_ = (path_full.filename()).string();
	std::string extension_ = (path_full.extension()).string();

	if (extension_ == ".jpg" || extension_ == ".JPG")
	{
		easyexif::EXIFInfo result;
		int iserror = result.readExif(path_);
		if (!iserror)
		{
			// focal length in pixel
			image_infos_[idx]->f_mm = (float)result.FocalLength;
			image_infos_[idx]->f_pixel = (float)result.FocalLength*(float)result.LensInfo.FocalPlaneXResolution;
			image_infos_[idx]->gps_latitude = (float)result.GeoLocation.Latitude;
			image_infos_[idx]->gps_attitude = (float)result.GeoLocation.Altitude;
			image_infos_[idx]->gps_longitude = (float)result.GeoLocation.Longitude;
			image_infos_[idx]->cam_maker = result.Make;
			image_infos_[idx]->cam_model = result.Model;
		}
	}

	return true;
}

bool Database::LoadAllImageInfo()
{

	return true;
}

void Database::ExtractImageFeatures(int idx)
{
	// read in the image exif information
	image_infos_[idx] = new ImageInfo;
	ExtractImageInfo(idx);

	// read in image data
	cv::Mat img = cv::imread(image_paths_[idx], 0);
	float img_size = img.cols*img.rows;
	int pitch = 128;
	float ratio = 0.0;
	cv::Size resize;
	if (options.resize_image / img_size >= 0.8)
	{
		resize.width = (img.cols / pitch + 1) * pitch;
	}
	else
	{
		resize.width = 1664;
	}
	ratio = float(resize.width) / img.cols;
	resize.height = int(ratio*img.rows);

	cv::resize(img, img, resize);
	
	image_infos_[idx]->cols = img.cols;
	image_infos_[idx]->rows = img.rows;
	//image_infos_[idx]->f_pixel *= ratio;
	image_infos_[idx]->f_pixel = 0.0;
	image_infos_[idx]->zoom_ratio = ratio;

	// extract feature
	keypoints_[idx] = new ListKeyPoint;
	descriptors_[idx] = new cv::Mat;
	if (options.feature_type == "SIFT")
	{
		FeatureExtractorOpenCV::Run(img, "SIFT", keypoints_[idx], descriptors_[idx]);
	}
	else if (options.feature_type == "SURF")
	{
		FeatureExtractorOpenCV::Run(img, "SURF", keypoints_[idx], descriptors_[idx]);
	}
	else if (options.feature_type == "VLSIFT")
	{
		VLSiftExtractor vlsift;
		vlsift.Run(img, keypoints_[idx], descriptors_[idx]);
	}
	else
	{
		CUDASiftExtractor::Run(img, keypoints_[idx], descriptors_[idx]);
	}

	if (idx % 5 == 0)
	{
		std::cout << "---Extracing feature " << idx << "/" << image_paths_.size() << " points:" << keypoints_[idx]->pts.size() << std::endl;
	}

	if (options.extract_gist)
	{
		cv::Mat img_rgb = cv::imread(image_paths_[idx]);

		Gist *gister = new Gist();
		Gist::Params *gist_paras = new Gist::Params();
		vector<int> vec_s32_or(4, 8);   // 4 scales, 8 orientations
		gist_paras->vecIntOrientation = vec_s32_or;
		Gist::GistDescriptor *gist_descriptor = gister->compute(img, gist_paras);
		gist_descriptors_[idx] = vector<float>(gist_descriptor->pfGistDescriptor, gist_descriptor->pfGistDescriptor + gist_descriptor->s32Length);

		delete gist_paras;
		delete gister;
	}
}


bool Database::ReadinImageFeatures(int idx)
{
	//if (keypoints_[idx] != NULL && keypoints_[idx]->pts.size())
	//{
	//	return true;
	//}

	std::string path = output_fold_ + "//" + std::to_string(idx) + "_feature";
	std::ifstream ifs(path, std::ios::binary);

	if (!ifs.is_open())
	{
		return false;
	}

	//
	image_infos_[idx] = new ImageInfo;
	keypoints_[idx] = new ListKeyPoint;
	descriptors_[idx] = new cv::Mat;

	// read in image info
	ifs.read((char*)(&image_infos_[idx]->rows), sizeof(int));
	ifs.read((char*)(&image_infos_[idx]->cols), sizeof(int));
	ifs.read((char*)(&image_infos_[idx]->zoom_ratio), sizeof(float));
	ifs.read((char*)(&image_infos_[idx]->f_mm), sizeof(float));
	ifs.read((char*)(&image_infos_[idx]->f_pixel), sizeof(float));
	ifs.read((char*)(&image_infos_[idx]->gps_latitude), sizeof(float));
	ifs.read((char*)(&image_infos_[idx]->gps_longitude), sizeof(float));

	int size_cam_maker = 0;
	ifs.read((char*)(&size_cam_maker), sizeof(int));
	char *data_cam_maker = new char[size_cam_maker+1];
	ifs.read(data_cam_maker, size_cam_maker * sizeof(char));
	data_cam_maker[size_cam_maker] = '\0';
	image_infos_[idx]->cam_maker = data_cam_maker;

	int size_cam_model = 0;
	ifs.read((char*)(&size_cam_model), sizeof(int));
	char *data_cam_model = new char[size_cam_model+1];
	ifs.read(data_cam_model, size_cam_model * sizeof(char));
	data_cam_model[size_cam_model] = '\0';
	image_infos_[idx]->cam_model = data_cam_model;

	// read in key points
	int num_pts;
	ifs.read((char*)(&num_pts), sizeof(int));
	float* ptr_pts = new float[2*num_pts];
	ifs.read((char*)ptr_pts, 2 * sizeof(float)*num_pts);
	keypoints_[idx]->pts.clear();
	keypoints_[idx]->pts.resize(num_pts);
	float* ptr_temp = ptr_pts;
	for (size_t i = 0; i < num_pts; i++)
	{
		keypoints_[idx]->pts[i].pt.x = ptr_temp[0];
		keypoints_[idx]->pts[i].pt.y = ptr_temp[1];
		ptr_temp += 2;
	}
	delete[] ptr_pts;

	// read in descriptor
	int rows, cols, type;
	ifs.read((char*)(&rows), sizeof(int));
	ifs.read((char*)(&cols), sizeof(int));
	ifs.read((char*)(&type), sizeof(int));
	descriptors_[idx]->release();
	descriptors_[idx]->create(rows, cols, type);
	ifs.read((char*)(descriptors_[idx]->data), descriptors_[idx]->elemSize() * descriptors_[idx]->total());

	ifs.close();

	return true;
}

bool Database::ReadinImageFeaturesUndistorted(int idx, CameraModel *cam_model)
{
	std::string path = output_fold_ + "//" + std::to_string(idx) + "_feature";
	std::ifstream ifs(path, std::ios::binary);

	if (!ifs.is_open()) {
		return false;
	}

	//
	image_infos_[idx] = new ImageInfo;
	keypoints_[idx] = new ListKeyPoint;
	descriptors_[idx] = new cv::Mat;

	// read in image info
	ifs.read((char*)(&image_infos_[idx]->rows), sizeof(int));
	ifs.read((char*)(&image_infos_[idx]->cols), sizeof(int));
	ifs.read((char*)(&image_infos_[idx]->zoom_ratio), sizeof(float));
	ifs.read((char*)(&image_infos_[idx]->f_mm), sizeof(float));
	ifs.read((char*)(&image_infos_[idx]->f_pixel), sizeof(float));
	ifs.read((char*)(&image_infos_[idx]->gps_latitude), sizeof(float));
	ifs.read((char*)(&image_infos_[idx]->gps_longitude), sizeof(float));

	int size_cam_maker = 0;
	ifs.read((char*)(&size_cam_maker), sizeof(int));
	char *data_cam_maker = new char[size_cam_maker + 1];
	ifs.read(data_cam_maker, size_cam_maker * sizeof(char));
	data_cam_maker[size_cam_maker] = '\0';
	image_infos_[idx]->cam_maker = data_cam_maker;

	int size_cam_model = 0;
	ifs.read((char*)(&size_cam_model), sizeof(int));
	char *data_cam_model = new char[size_cam_model + 1];
	ifs.read(data_cam_model, size_cam_model * sizeof(char));
	data_cam_model[size_cam_model] = '\0';
	image_infos_[idx]->cam_model = data_cam_model;

	
	// distortion 
	double k1 = cam_model->k1_;
	double k2 = cam_model->k2_;
	double f = cam_model->f_;
	Eigen::VectorXd polynomial(6);
	polynomial[0] = pow(k2, 2);
	polynomial[1] = 2 * k1 * k2;
	polynomial[2] = 2 * k2 + pow(k1, 2);
	polynomial[3] = 2 * k1;
	polynomial[4] = 1.0;

	if (0)
	{
		// test the calibration parameters
		cv::Mat image = cv::imread(image_paths_[idx]);
		int n = image.channels();
		float ratio = image_infos_[idx]->zoom_ratio;
		cv::resize(image, image, cv::Size(image.cols*ratio, image.rows*ratio));
		cv::Mat img(2 * image.rows, 2 * image.cols, CV_8UC3, cv::Scalar(0, 0, 0));
		uchar* ptr = image.data;
		uchar* ptr_new = img.data;
		int loc_pre = 0;
		for (size_t i = 0; i < image.rows; i++)
		{
			cout << i << endl;
			for (size_t j = 0; j < image.cols; j++)
			{
				double u = j - image.cols / 2.0;
				double v = i - image.rows / 2.0;

				Eigen::VectorXd real(5), imaginary(5);
				polynomial[5] = -(u*u + v * v) / (f*f);
				FindPolynomialRootsCompanionMatrix(polynomial, &real, &imaginary);
				double r2 = 0;
				for (size_t m = 0; m < 5; m++) {
					if (imaginary[m] == 0) {
						r2 = real[m];
					}
				}
				if (r2 != 0) {
					double d = 1 + k1 * r2 + k2 * r2 * r2;
					u /= d;
					v /= d;
				}

				int x = u + img.cols / 2;
				int y = v + img.rows / 2;
				if (x >= 0 && x<img.cols && y >= 0 && y<img.rows)
				{
					int loc = 3 * (y*img.cols + x);
					ptr_new += loc - loc_pre;
					ptr_new[0] = ptr[0];
					ptr_new[1] = ptr[1];
					ptr_new[2] = ptr[2];
					loc_pre = loc;
				}
				ptr += 3;
			}
		}

		cv::imwrite("F:\\rect_img.bmp", img);
	}

	// read in key points
	int num_pts;
	ifs.read((char*)(&num_pts), sizeof(int));
	float* ptr_pts = new float[2 * num_pts];
	ifs.read((char*)ptr_pts, 2 * sizeof(float)*num_pts);
	keypoints_[idx]->pts.clear();
	keypoints_[idx]->pts.resize(num_pts);
	float* ptr_temp = ptr_pts;
	for (size_t i = 0; i < num_pts; i++)
	{
		double u = ptr_temp[0];
		double v = ptr_temp[1];

		// undistortion
		/* u = f * d * x
		   v = f * d * y
		   d = 1 + k1 * r2 + k2 * r4
		   so, u2+v2 = f2*d2*(x*x+ y*y) = f2*d2*r2
		             = f2*(1 + k1 * r2 + k2 * r4)2*r2
					 = f2*(1 + k1 * r2 + k2 * r4)2*r2
		*/
		Eigen::VectorXd real(5), imaginary(5);
		polynomial[5] = -(u*u + v * v) / (f*f);
		FindPolynomialRootsCompanionMatrix(polynomial, &real, &imaginary);
		double r2 = 0;
		for (size_t j = 0; j < 5; j++)
		{
			if (imaginary[j] == 0) {
				r2 = real[j];
			}
		}
		if (r2 != 0) {
			double d = 1 + k1 * r2 + k2 * r2 * r2;
			keypoints_[idx]->pts[i].pt.x = u / d;
			keypoints_[idx]->pts[i].pt.y = v / d;
		}
		else {
			keypoints_[idx]->pts[i].pt.x = u;
			keypoints_[idx]->pts[i].pt.y = v;
		}
		ptr_temp += 2;
	}
	delete[] ptr_pts;

	// read in descriptor
	int rows, cols, type;
	ifs.read((char*)(&rows), sizeof(int));
	ifs.read((char*)(&cols), sizeof(int));
	ifs.read((char*)(&type), sizeof(int));
	descriptors_[idx]->release();
	descriptors_[idx]->create(rows, cols, type);
	ifs.read((char*)(descriptors_[idx]->data), descriptors_[idx]->elemSize() * descriptors_[idx]->total());

	ifs.close();

	return true;
}

bool Database::ReadinImageKeyPoints(int idx)
{
	if (keypoints_[idx] != NULL && keypoints_[idx]->pts.size())
	{
		return true;
	}

	std::string path = output_fold_ + "//" + std::to_string(idx) + "_feature";
	std::ifstream ifs(path, std::ios::binary);

	if (!ifs.is_open())
	{
		return false;
	}

	// read in image info
	ImageInfo img_inf_temp;
	ifs.read((char*)(&img_inf_temp.rows), sizeof(int));
	ifs.read((char*)(&img_inf_temp.cols), sizeof(int));
	ifs.read((char*)(&img_inf_temp.zoom_ratio), sizeof(float));
	ifs.read((char*)(&img_inf_temp.f_mm), sizeof(float));
	ifs.read((char*)(&img_inf_temp.f_pixel), sizeof(float));
	ifs.read((char*)(&img_inf_temp.gps_latitude), sizeof(float));
	ifs.read((char*)(&img_inf_temp.gps_longitude), sizeof(float));

	int size_cam_maker = 0;
	ifs.read((char*)(&size_cam_maker), sizeof(int));
	char *data_cam_maker = new char[size_cam_maker + 1];
	ifs.read(data_cam_maker, size_cam_maker * sizeof(char));
	data_cam_maker[size_cam_maker] = '\0';

	int size_cam_model = 0;
	ifs.read((char*)(&size_cam_model), sizeof(int));
	char *data_cam_model = new char[size_cam_model + 1];
	ifs.read(data_cam_model, size_cam_model * sizeof(char));
	data_cam_model[size_cam_model] = '\0';

	// read in key points
	keypoints_[idx] = new ListKeyPoint;
	int num_pts;
	ifs.read((char*)(&num_pts), sizeof(int));
	float* ptr_pts = new float[2 * num_pts];
	ifs.read((char*)ptr_pts, 2 * sizeof(float)*num_pts);
	keypoints_[idx]->pts.clear();
	keypoints_[idx]->pts.resize(num_pts);
	float* ptr_temp = ptr_pts;
	for (size_t i = 0; i < num_pts; i++)
	{
		keypoints_[idx]->pts[i].pt.x = ptr_temp[0];
		keypoints_[idx]->pts[i].pt.y = ptr_temp[1];
		ptr_temp += 2;
	}
	delete[] ptr_pts;

	ifs.close();

	return true;
}

bool Database::ReleaseImageKeyPoints(int idx)
{
	std::vector<cv::KeyPoint>().swap(keypoints_[idx]->pts);
	return true;
}

bool Database::WriteoutImageFeature(int idx)
{
	std::string path = output_fold_ + "//" + std::to_string(idx) + "_feature";
	std::ofstream ofs(path, std::ios::binary);

	if (!ofs.is_open())
	{
		return false;
	}

	// write out image information
	ofs.write((const char*)(&image_infos_[idx]->rows), sizeof(int));
	ofs.write((const char*)(&image_infos_[idx]->cols), sizeof(int));
	ofs.write(reinterpret_cast<const char*>(&image_infos_[idx]->zoom_ratio), sizeof(float));
	ofs.write(reinterpret_cast<const char*>(&image_infos_[idx]->f_mm), sizeof(float));
	ofs.write(reinterpret_cast<const char*>(&image_infos_[idx]->f_pixel), sizeof(float));
	ofs.write(reinterpret_cast<const char*>(&image_infos_[idx]->gps_latitude), sizeof(float));
	ofs.write(reinterpret_cast<const char*>(&image_infos_[idx]->gps_longitude), sizeof(float));

	int size_cam_maker = image_infos_[idx]->cam_maker.length();
	ofs.write((const char*)(&size_cam_maker), sizeof(int));
	ofs.write(image_infos_[idx]->cam_maker.data(), size_cam_maker*sizeof(char));

	int size_cam_model = image_infos_[idx]->cam_model.length();
	ofs.write((const char*)(&size_cam_model), sizeof(int));
	ofs.write(image_infos_[idx]->cam_model.data(), size_cam_model * sizeof(char));

	// write out key points
	int num_pts = keypoints_[idx]->pts.size();
	ofs.write((const char*)(&num_pts), sizeof(int));
	float* ptr_pts = new float[2 * num_pts];
	float* ptr_idx = ptr_pts;
	for (size_t i = 0; i < num_pts; i++)    // points are centralized
	{
		ptr_idx[0] = keypoints_[idx]->pts[i].pt.x - image_infos_[idx]->cols / 2.0;
		ptr_idx[1] = keypoints_[idx]->pts[i].pt.y - image_infos_[idx]->rows / 2.0;
		ptr_idx += 2;
	}
	ofs.write((const char*)ptr_pts, 2 * sizeof(float)*num_pts);
	delete[] ptr_pts;

	// write out descriptor
	int type = descriptors_[idx]->type();
	ofs.write((const char*)(&descriptors_[idx]->rows), sizeof(int));
	ofs.write((const char*)(&descriptors_[idx]->cols), sizeof(int));
	ofs.write((const char*)(&type), sizeof(int));
	ofs.write((const char*)(descriptors_[idx]->data), descriptors_[idx]->elemSize() * descriptors_[idx]->total());

	ofs.close();

	return true;
}

bool Database::WriteoutImageFeature(int idx, std::string path)
{
	std::ofstream ofs(path, std::ios::binary);

	if (!ofs.is_open())
	{
		return false;
	}

	// write out image information
	ofs.write((const char*)(&image_infos_[idx]->rows), sizeof(int));
	ofs.write((const char*)(&image_infos_[idx]->cols), sizeof(int));
	ofs.write(reinterpret_cast<const char*>(&image_infos_[idx]->zoom_ratio), sizeof(float));
	ofs.write(reinterpret_cast<const char*>(&image_infos_[idx]->f_mm), sizeof(float));
	ofs.write(reinterpret_cast<const char*>(&image_infos_[idx]->f_pixel), sizeof(float));
	ofs.write(reinterpret_cast<const char*>(&image_infos_[idx]->gps_latitude), sizeof(float));
	ofs.write(reinterpret_cast<const char*>(&image_infos_[idx]->gps_longitude), sizeof(float));

	int size_cam_maker = image_infos_[idx]->cam_maker.length();
	ofs.write((const char*)(&size_cam_maker), sizeof(int));
	ofs.write(image_infos_[idx]->cam_maker.data(), size_cam_maker * sizeof(char));

	int size_cam_model = image_infos_[idx]->cam_model.length();
	ofs.write((const char*)(&size_cam_model), sizeof(int));
	ofs.write(image_infos_[idx]->cam_model.data(), size_cam_model * sizeof(char));

	// write out key points
	int num_pts = keypoints_[idx]->pts.size();
	ofs.write((const char*)(&num_pts), sizeof(int));
	float* ptr_pts = new float[2 * num_pts];
	float* ptr_idx = ptr_pts;
	for (size_t i = 0; i < num_pts; i++)    // points are centralized
	{
		ptr_idx[0] = keypoints_[idx]->pts[i].pt.x - image_infos_[idx]->cols / 2.0;
		ptr_idx[1] = keypoints_[idx]->pts[i].pt.y - image_infos_[idx]->rows / 2.0;
		ptr_idx += 2;
	}
	ofs.write((const char*)ptr_pts, 2 * sizeof(float)*num_pts);
	delete[] ptr_pts;

	// write out descriptor
	int type = descriptors_[idx]->type();
	ofs.write((const char*)(&descriptors_[idx]->rows), sizeof(int));
	ofs.write((const char*)(&descriptors_[idx]->cols), sizeof(int));
	ofs.write((const char*)(&type), sizeof(int));
	ofs.write((const char*)(descriptors_[idx]->data), descriptors_[idx]->elemSize() * descriptors_[idx]->total());

	ofs.close();

	return true;
}

void Database::ReleaseImageFeatures(int idx)
{
	delete image_infos_[idx];
	std::vector<cv::KeyPoint>().swap(keypoints_[idx]->pts);
	(*descriptors_[idx]).release();
}

bool Database::WriteoutGistFeature()
{
	std::string path = output_fold_ + "//" + "gist_feature";
	std::ofstream ofs(path, std::ios::binary);

	std::cout << gist_descriptors_[0].size() << std::endl;
	int dim_gist = gist_descriptors_[0].size();
	ofs.write((const char*)(&dim_gist), sizeof(int));

	for (size_t idx = 0; idx < num_imgs_; idx++)
	{
		float* ptr_gist = new float[dim_gist];
		for (size_t i = 0; i < dim_gist; i++)
		{
			ptr_gist[i] = gist_descriptors_[idx][i];
		}
		ofs.write((const char*)ptr_gist, dim_gist * sizeof(float));
		delete[] ptr_gist;
	}
	ofs.close();

	return true;
}

bool Database::ReadinGistFeature()
{
	std::string path = output_fold_ + "//" + "gist_feature";
	std::ifstream ifs(path, std::ios::binary);

	int dim_gist = 0;
	ifs.read((char*)(&dim_gist), sizeof(int));

	for (size_t idx = 0; idx < num_imgs_; idx++)
	{
		float* ptr_gist = new float[dim_gist];
		ifs.read((char*)ptr_gist, dim_gist * sizeof(float));

		gist_descriptors_[idx].resize(dim_gist);
		for (size_t i = 0; i < dim_gist; i++)
		{
			gist_descriptors_[idx][i] = ptr_gist[i];
		}

		delete[] ptr_gist;
	}
	ifs.close();

	return true;
}


// Vocabulary

void Database::BuildVocabularyTree()
{
	int num_imgs = image_paths_.size();
	int sample_step = MAX_(num_imgs / options.num_image_voc, 1);

	// sample a subset to generate the vocabulary for efficiency
	std::vector<cv::Mat> features;
	for (size_t i = 0; i < num_imgs; i += sample_step)
	{
		int idx = i;
		ReadinImageFeatures(idx);
		features.push_back(descriptors_[idx]->clone());
		ReleaseImageFeatures(idx);
	}

	// generate the vocabulary
	int nThreads = 3;
	auto t_start = std::chrono::high_resolution_clock::now();
	voc_creator_.create(voc_, features, options.feature_type, fbow::VocabularyCreator::Params(options.fbow_k, options.fbow_l, nThreads));
	auto t_end = std::chrono::high_resolution_clock::now();

	std::cout << "---num vocabularies=" << voc_.size() << std::endl;
}

void Database::ReadinVocabularyTree()
{
	std::string voc_path = output_fold_ + "//voctree";

	std::ifstream ifs(voc_path, std::ios::binary);

	if (!ifs.is_open())
	{
		return;
	}

	ifs.read((char*)(&voc_._params._aligment), sizeof(uint32_t));
	ifs.read((char*)(&voc_._params._nblocks), sizeof(uint32_t));
	ifs.read((char*)(&voc_._params._desc_size_bytes_wp), sizeof(uint64_t));
	ifs.read((char*)(&voc_._params._block_size_bytes_wp), sizeof(uint64_t));
	ifs.read((char*)(&voc_._params._feature_off_start), sizeof(uint64_t));
	ifs.read((char*)(&voc_._params._child_off_start), sizeof(uint64_t));
	ifs.read((char*)(&voc_._params._total_size), sizeof(uint64_t));
	ifs.read((char*)(&voc_._params._desc_type), sizeof(int));
	ifs.read((char*)(&voc_._params._desc_size), sizeof(int));
	ifs.read((char*)(&voc_._params._m_k), sizeof(uint32_t));

	if (voc_._data != 0) free(voc_._data);
	voc_._data = (char*)malloc(voc_._params._total_size);
	ifs.read((char*)(voc_._data), voc_._params._total_size);

	ifs.close();
}

void Database::WriteoutVocabularyTree()
{
	std::string voc_path = output_fold_ + "//voctree";

	std::ofstream ofs(voc_path, std::ios::binary);

	if (!ofs.is_open())
	{
		return;
	}

	ofs.write((const char*)(&voc_._params._aligment), sizeof(uint32_t));
	ofs.write((const char*)(&voc_._params._nblocks), sizeof(uint32_t));
	ofs.write((const char*)(&voc_._params._desc_size_bytes_wp), sizeof(uint64_t));
	ofs.write((const char*)(&voc_._params._block_size_bytes_wp), sizeof(uint64_t));
	ofs.write((const char*)(&voc_._params._feature_off_start), sizeof(uint64_t));
	ofs.write((const char*)(&voc_._params._child_off_start), sizeof(uint64_t));
	ofs.write((const char*)(&voc_._params._total_size), sizeof(uint64_t));
	ofs.write((const char*)(&voc_._params._desc_type), sizeof(int));
	ofs.write((const char*)(&voc_._params._desc_size), sizeof(int));
	ofs.write((const char*)(&voc_._params._m_k), sizeof(uint32_t));

	ofs.write((const char*)(voc_._data), voc_._params._total_size);

	ofs.close();
}


bool Database::CheckVocabularyTreeExist()
{
	std::string file_voc = output_fold_ + "//voctree";
	std::ifstream infile(file_voc, std::ios::binary);
	if (!infile.is_open())
	{
		return false;
	}
	return true;
	infile.close();
}

/* Words*/

void Database::BuildWords()
{
	// find the images without words
	std::vector<int> idx_missing;
	if (CheckWordsIndexExist())
	{
		idx_missing = CheckMissingWordsFile();
	}
	else
	{
		idx_missing.resize(num_imgs_,0);
		for (size_t i = 0; i < num_imgs_; i++)
		{
			idx_missing[i] = i;
		}
	}
	max_words_id_ = 0;

	// generate missing words
	if (idx_missing.size())
	{
		std::string file_words = output_fold_ + "//words_index.txt";
		std::ofstream ofs(file_words, std::ios::app);

		// vocabulary tree
		if (CheckVocabularyTreeExist())
		{
			ReadinVocabularyTree();
		}
		else
		{
			//std::cout << "---Fatal Error: No voc-tree found" << std::endl;
			//return;
			BuildVocabularyTree();
			WriteoutVocabularyTree();
		}
		std::cout << "---Number of blocks " << voc_.size() << std::endl;

		// generate words
		for (int i = 0; i < idx_missing.size(); i++)
		{
			if (i % 5 == 0)
			{
				std::cout << "---Generating word " << i << "/" << idx_missing.size() << std::endl;
			}

			int idx = idx_missing[i];
			ReadinImageFeatures(idx);
			if (keypoints_[idx]->pts.size() > 300)
			{
				GenerateWordsForImage(idx);
			}
			WriteoutWordsForImage(idx);

			ReleaseImageFeatures(idx);
			ReleaseWordsForImage(idx);

			ofs << idx << std::endl;
		}
		ofs.close();
	}
}

bool Database::CheckWordsIndexExist()
{
	std::string file_words = output_fold_ + "//words_index.txt";
	std::ifstream infile(file_words);
	if (!infile.is_open())
	{
		return false;
	}
	return true;
	infile.close();
}

std::vector<int> Database::CheckMissingWordsFile()
{
	std::string file_words = output_fold_ + "//words_index.txt";
	std::ifstream infile(file_words);

	std::vector<int> index(num_imgs_, 0);
	int idx = -1;
	while (!infile.eof())
	{
		infile >> idx;
		if (idx >= 0)
		{
			index[idx] = 1;
		}
	}

	std::vector<int> missing_idx;
	for (size_t i = 0; i < num_imgs_; i++)
	{
		if (!index[i])
		{
			missing_idx.push_back(i);
		}
	}

	return missing_idx;
}

void Database::GenerateWordsForImage(int idx)
{
	if (!voc_.size())
	{
		std::cout << "---No vocabulary tree available!" << std::endl;
		return;
	}

	words_fbow_[idx] = new fbow::fBow;
	(*words_fbow_[idx]) = voc_.transform(*descriptors_[idx], words_id_[idx]);

	int max_words_id_cur = 0;
	math::get_max(words_id_[idx], words_id_[idx].size(), max_words_id_cur);
	max_words_id_ = MAX(max_words_id_cur, max_words_id_);
}

void Database::ReadinWordsForImage(int idx)
{
	std::string path = output_fold_ + "//" + std::to_string(idx) + "_words";
	std::ifstream ifs(path, std::ios::binary);

	// read in words
	int num_words = 0;
	ifs.read((char*)(&num_words), sizeof(int));
	int* ptr_first = new int[num_words];
	float* ptr_second = new float[num_words];
	ifs.read((char*)ptr_first, sizeof(int)*num_words);
	ifs.read((char*)ptr_second, sizeof(float)*num_words);

	words_vector_[idx] = new std::map<uint32_t, float>;
	for (size_t i = 0; i < num_words; i++)
	{
		words_vector_[idx]->insert(std::pair<uint32_t, float>(ptr_first[i], ptr_second[i]));
	}
	delete[] ptr_first;
	delete[] ptr_second;

	// read in feature-word ids
	int num_pts = 0;
	ifs.read((char*)(&max_words_id_), sizeof(int));
	ifs.read((char*)(&num_pts), sizeof(int));
	int* ptr_ids = new int[num_pts];
	ifs.read((char*)ptr_ids, sizeof(int)*num_pts);
	words_id_[idx].resize(num_pts);
	for (size_t i = 0; i < num_pts; i++)
	{
		words_id_[idx][i] = *ptr_ids++;
	}

	ifs.close();
}

void Database::WriteoutWordsForImage(int idx)
{
	std::string path = output_fold_ + "//" + std::to_string(idx) + "_words";
	std::ofstream ofs(path, std::ios::binary);

	// write out words
	if (words_fbow_[idx] != NULL)
	{
		int num_words = words_fbow_[idx]->size();
		int* ptr_first = new int[num_words];
		float* ptr_second = new float[num_words];
		int* ptr_first_temp = ptr_first;
		float* ptr_second_temp = ptr_second;
		for (auto e : *words_fbow_[idx])
		{
			*ptr_first_temp++ = e.first;
			*ptr_second_temp++ = e.second;
		}

		ofs.write((const char*)(&num_words), sizeof(int));
		ofs.write((const char*)ptr_first, sizeof(int)*num_words);
		ofs.write((const char*)ptr_second, sizeof(float)*num_words);
		delete[] ptr_first;
		delete[] ptr_second;

		// write out feature-word ids
		int num_pts = keypoints_[idx]->pts.size();
		int* ptr_id = new int[num_pts];
		for (size_t i = 0; i < num_pts; i++)
		{
			ptr_id[i] = words_id_[idx][i];
		}

		ofs.write((const char*)(&max_words_id_), sizeof(int));
		ofs.write((const char*)(&num_pts), sizeof(int));
		ofs.write((const char*)ptr_id, sizeof(int)*num_pts);
		delete[] ptr_id;
	}
	
	ofs.close();
}

void Database::WriteoutWordsForImage(int idx, std::string path)
{
	std::ofstream ofs(path, std::ios::binary);

	// write out words
	if (words_vector_[idx]->size() > 0)
	{
		int num_words = words_vector_[idx]->size();
		int* ptr_first = new int[num_words];
		float* ptr_second = new float[num_words];
		int* ptr_first_temp = ptr_first;
		float* ptr_second_temp = ptr_second;
		for (auto e : *words_vector_[idx])
		{
			*ptr_first_temp++ = e.first;
			*ptr_second_temp++ = e.second;
		}

		ofs.write((const char*)(&num_words), sizeof(int));
		ofs.write((const char*)ptr_first, sizeof(int)*num_words);
		ofs.write((const char*)ptr_second, sizeof(float)*num_words);
		delete[] ptr_first;
		delete[] ptr_second;

		// write out feature-word ids
		int num_pts = keypoints_[idx]->pts.size();
		int* ptr_id = new int[num_pts];
		for (size_t i = 0; i < num_pts; i++)
		{
			ptr_id[i] = words_id_[idx][i];
		}

		ofs.write((const char*)(&max_words_id_), sizeof(int));
		ofs.write((const char*)(&num_pts), sizeof(int));
		ofs.write((const char*)ptr_id, sizeof(int)*num_pts);
		delete[] ptr_id;
	}

	ofs.close();
}

void Database::ReleaseWordsForImage(int idx)
{
	if (words_fbow_[idx] != NULL)
	{
		words_fbow_[idx]->clear();
	}
	if (words_vector_[idx] != NULL)
	{
		words_vector_[idx]->clear();
	}
	if (!words_id_[idx].empty())
	{
		words_id_[idx].clear();
	}
}

void Database::ReadinDSMInfo()
{
	// read in tfw information
	std::string path_tfw = fold_dsm_ + "\\dsm.tfw";
	ifstream reader(path_tfw);
	double temp1, temp2;
	reader >> gsd_x_ >> temp1;
	reader >> temp2 >> gsd_y_;
	reader >> ori_x_ >> ori_y_;
	reader.close();

	img_pixel_scale_ = (abs(gsd_x_) + abs(gsd_y_)) / 2.0;
}

void Database::ReadinDSMImage()
{
	// read in image data
	std::string path_img = fold_dsm_ + "\\dsm.tif";
	img_data_ = new float[dsm_rows_*dsm_cols_*bands_];
	ReadImageFile(path_img.c_str(), (unsigned char *)img_data_, dsm_cols_, dsm_rows_, bands_);
}

void Database::ReadinDSMImage(float lat, float lon, float radius)
{
	ReadinDSMInfo();

	// read in the whole image
	ReadinDSMImage();

	// convert the lat and lon into x y
	double x, y;
	LLtoUTM(ellipsoid_id_, lat, lon, y, x, (char*)zone_id_.data());

	/*cv::Mat_<double> R1 = (cv::Mat_<double>(3, 3) << 0.71718, - 0.69689, 0.00000,
		0.00000, 0.00000, - 1.00000,
		0.69689, 0.71718, 0.00000);
	cv::Mat_<double> t1= (cv::Mat_<double>(3, 1) << 3476904.70746, 1.70000, -4583177.04679);
	cv::Mat_<double> c1 = -R1.inv() * t1;

	x = c1(0);
	y = c1(1);

	cv::Mat_<double> R2 = (cv::Mat_<double>(3, 3) << 0.77351, - 0.63378, 0.00000,
		0.00000, - 0.00000, - 1.00000,
		0.63378, 0.77351, 0.00000);
	cv::Mat_<double> t2 = (cv::Mat_<double>(3, 1) << 3077119.59504, 1.70000, - 4860636.42764);
	cv::Mat_<double> c2 = -R2.inv() * t2;

	cv::Mat_<double> R3 = (cv::Mat_<double>(3, 3) << 0.88696, - 0.46184, 0.00000,
		- 0.00000, 0.00000, - 1.00000,
		0.46184, 0.88696, 0.00000);
	cv::Mat_<double> t3 = (cv::Mat_<double>(3, 1) << 2015857.27017, 1.70000, - 5388027.91920);
	cv::Mat_<double> c3 = -R3.inv() * t3;

	cv::Mat_<double> R4 = (cv::Mat_<double>(3, 3) << 0.98078, 0.19514, 0.00000,
		- 0.00000, - 0.00000, - 1.00000,
		- 0.19514, 0.98078, 0.00000);
	cv::Mat_<double> t4 = (cv::Mat_<double>(3, 1) << -1801189.13783, 1.70000, - 5463544.81529);
	cv::Mat_<double> c4 = -R4.inv() * t4;
	std::cout << c1 << std::endl;
	std::cout << c2 << std::endl;
	std::cout << c3 << std::endl;
	std::cout << c4 << std::endl;*/

	// crop the dsm image
	int xmin = MAX((x - radius - ori_x_) / gsd_x_, 0);
	int xmax = MIN((x + radius - ori_x_) / gsd_x_, dsm_cols_);
	int ymin = MAX((y + radius - ori_y_) / gsd_y_, 0);
	int ymax = MIN((y - radius - ori_y_) / gsd_y_, dsm_rows_);

	img_cropped_ = cv::Mat(xmax - xmin, ymax - ymin, CV_32FC1, cv::Scalar(0.0));
	float* ptr_roi = (float*)img_cropped_.data;
	for (size_t i = ymin; i < ymax; i++)
	{
		float* ptr_img = img_data_ + i * dsm_cols_ + xmin;
		for (size_t j = xmin; j < xmax; j++)
		{
			*ptr_roi = *ptr_img;
			ptr_roi++;
			ptr_img++;
		}
	}

	//double xs = x - radius;
	//double ys = y - radius;
	//cv::circle(img_cropped_, cv::Point((c1(0) - xs) / gsd_x_, (c1(1) - ys) / -gsd_y_), 5, cv::Scalar(255), 3);
	//cv::circle(img_cropped_, cv::Point((c2(0) - xs) / gsd_x_, (c2(1) - ys) / -gsd_y_), 5, cv::Scalar(255), 3);
	//cv::circle(img_cropped_, cv::Point((c3(0) - xs) / gsd_x_, (c3(1) - ys) / -gsd_y_), 5, cv::Scalar(255), 3);
	//cv::circle(img_cropped_, cv::Point((c4(0) - xs) / gsd_x_, (c4(1) - ys) / -gsd_y_), 5, cv::Scalar(255), 3);
	//cv::imwrite("F:\\img_cropped_.jpg", img_cropped_);

	//
	delete[] img_data_;
}

void Database::ExtractSites()
{
	int rows = img_cropped_.rows;
	int cols = img_cropped_.cols;

	// sort all the data points
	float avg = 0.0;
	int count = 0;
	float* ptr = (float*)img_cropped_.data;
	for (size_t i = 0; i < rows; i++)
	{
		for (size_t j = 0; j < cols; j++)
		{
			if (*ptr == *ptr)
			{
				avg += *ptr;
				count++;
			}
			ptr++;
		}
	}
	avg /= count;

	std::vector<std::pair<int, float>> ground_hyps;
	ptr = (float*)img_cropped_.data;
	for (size_t i = 0; i < rows; i++)
	{
		for (size_t j = 0; j < cols; j++)
		{
			if (*ptr > avg)
			{
				ground_hyps.push_back(std::pair<int, float>(i*cols + j, *ptr));
			}
			ptr++;
		}
	}
	std::sort(ground_hyps.begin(), ground_hyps.end(), [](const std::pair<int, float> &lhs, const std::pair<int, float> &rhs) { return lhs.second < rhs.second; });

	// do region growing
	int x_offset[8] = { -1, -1, -1, 0, 1, 1, 1, 0 };
	int y_offset[8] = { 1, 0, -1, -1, -1, 0, 1, 1 };
	int loc_offset[8];
	for (size_t i = 0; i < 8; i++)
	{
		loc_offset[i] = y_offset[i] * cols + x_offset[i];
	}

	non_building_mask_ = cv::Mat(rows, cols, CV_8UC1, cv::Scalar(0));
	for (size_t i = 0; i <ground_hyps.size() / 20; i++)
	{
		int loc_seed = ground_hyps[i].first;
		int y_seed = loc_seed / cols;
		int x_seed = loc_seed % cols;
		if (non_building_mask_.at<uchar>(y_seed, x_seed))
		{
			continue;
		}

		std::vector<cv::Point> cluster;
		cluster.push_back(cv::Point(x_seed, y_seed));
		int count = 0;
		while (count < cluster.size())
		{
			int x0 = cluster[count].x;
			int y0 = cluster[count].y;
			if (x0 < 1 || x0>cols - 2 || y0<1 || y0>rows - 2)
			{
				count++;
				continue;
			}

			float *ptr = (float*)img_cropped_.data + y0 * cols + x0;
			float h0 = *ptr;
			for (size_t j = 0; j < 8; j++)
			{
				int x = x0 + x_offset[j];
				int y = y0 + y_offset[j];
				if (non_building_mask_.at<uchar>(y, x))
				{
					continue;
				}

				float hj = *(ptr + loc_offset[j]);
				if (hj == hj & abs(hj - h0)<0.2)
				{
					cluster.push_back(cv::Point(x, y));
					non_building_mask_.at<uchar>(y, x) = 1;
				}
			}
			count++;
		}

		if (cluster.size() > 1000)
		{
			int step = super_pixel_size_;
			for (size_t j = 0; j < cluster.size(); j += step)
			{
				int loc = cluster[j].y*cols + cluster[j].x;
				float x = cluster[j].x;
				float y = cluster[j].y;
				float z = img_cropped_.at<float>(y,x);
				site_pts_.push_back(cv::Point3f(x,y,z));
			}
		}
	}
}

void Database::ExtractSiteVisiblePts()
{
	int cols = img_cropped_.cols;
	int rows = img_cropped_.rows;

	// edge detection
	cv::Mat img_cropped_8U;
	img_cropped_.convertTo(img_cropped_8U, CV_8UC1);
	cv::Canny(img_cropped_8U, edge_map_, 5.0, 2.0*5.0, 3, false);
	cv::imwrite("F:\\edge_map_.jpg", 255*edge_map_);

	// extract the visible pts of each site
	int n_angles = 180;
	float d_angle = 2.0 * CV_PI / n_angles;

	site_visible_pts_.resize(site_pts_.size());
	for (size_t i = 0; i < site_pts_.size(); i++)
	{
		int x_site = int(site_pts_[i].x);
		if (x_site >= cols) x_site = cols - 1;
		int y_site = int(site_pts_[i].y);
		if (y_site >= cols) y_site = rows - 1;
		float h_site = site_pts_[i].z;

		std::vector<cv::Point3f> visible_pts_i;
		for (size_t j = 0; j < n_angles; j++)
		{
			float angle = j * d_angle;
			float dx = sin(angle);
			float dy = -cos(angle);

			float x = site_pts_[i].x, y = site_pts_[i].y;
			int x_former = 0, y_former = 0;
			int x_cur = 0, y_cur = 0;

			float angle_ratio_max = -100000.0;
			while (true)
			{
				x += dx;
				y += dy;
				x_cur = int(x);
				y_cur = int(y);

				if (x_cur < 0 || x_cur >= cols || y_cur < 0 || y_cur >= rows) break;
				if (x_cur == x_former && y_cur == y_former) continue;
				if (!edge_map_.at<uchar>(y_cur, x_cur)) continue;

				float h_cur = img_cropped_.at<float>(y_cur, x_cur);
				//float dis = img_pixel_scale_ * std::sqrt((x - site.x)*(x - site.x) + (y - site.y)*(y - site.y));
				float dis = std::sqrt((x - site_pts_[i].x)*(x - site_pts_[i].x) + (y - site_pts_[i].y)*(y - site_pts_[i].y));
				float angle_ratio_cur = (h_cur - h_site) / dis;
				if (angle_ratio_cur > angle_ratio_max)
				{
					angle_ratio_max = angle_ratio_cur;
					visible_pts_i.push_back(cv::Point3f(x_cur, y_cur, h_cur - h_site));
				}
			}
		}

		site_visible_pts_[i] = visible_pts_i;
	}

	// calculate the gradient orientation map of the image
	cv::Mat gx_map(rows, cols, CV_32FC1, Scalar(0));
	cv::Mat gy_map(rows, cols, CV_32FC1, Scalar(0));
	cv::Sobel(img_cropped_, gx_map, CV_32FC1, 1, 0, 5, 1, 0, cv::BORDER_REPLICATE);
	cv::Sobel(img_cropped_, gy_map, CV_32FC1, 0, 1, 5, 1, 0, cv::BORDER_REPLICATE);

	// calculate the site-pt and pt's orientation angle deviation
	float PI2 = 2 * CV_PI;
	int num_sites = site_visible_pts_.size();
	site_pts_angles_.resize(num_sites);
	for (size_t i = 0; i < num_sites; i++)
	{
		std::cout << i << std::endl;
		int num_pts = site_visible_pts_[i].size();
		site_pts_angles_[i].resize(num_pts);

		for (size_t j = 0; j < num_pts; j++)
		{
			// the site-pt angle
			float dx1 = site_pts_[i].x - site_visible_pts_[i][j].x;
			float dy1 = site_pts_[i].y - site_visible_pts_[i][j].y;
			float angle1 = atan2(dy1, dx1);

			// the pt orientation
			int x = site_visible_pts_[i][j].x;
			int y = site_visible_pts_[i][j].y;
			float dx2 = - gy_map.at<float>(y, x);
			float dy2 =   gx_map.at<float>(y, x);
			float angle2_1 = atan2(dy2, dx2);
			float angle2_2 = atan2(-dy2, -dx2);

			float angle_dev1 = angle2_1 - angle1;
			if (angle_dev1 < 0) angle_dev1 += 2 * CV_PI;
			float angle_dev2 = angle2_2 - angle1;
			if (angle_dev2 < 0) angle_dev2 += 2 * CV_PI;

			site_pts_angles_[i][j] = MIN(angle_dev1, angle_dev2);
		}
	}
}

void Database::ExtractSiteVisibleLines()
{
	int cols = img_cropped_.cols;
	int rows = img_cropped_.rows;

	// line detection
	cv::Mat img_cropped_8U;
	img_cropped_.convertTo(img_cropped_8U, CV_8UC1);
	
	image_double image = new_image_double(img_cropped_8U.cols, img_cropped_8U.rows);
	int img_size = img_cropped_8U.cols * img_cropped_8U.rows;
	double* ptr1 = image->data;
	uchar* ptr2 = img_cropped_8U.data;
	for (size_t i = 0; i < img_size; i++)
	{
		*ptr1++ = *ptr2++;
	}
		
	ntuple_list lines_lsd = lsd(image, 1.0);
	free_image_double(image);
	int dim = lines_lsd->dim;

	// get the line attitude
	int win_size = 5;
	std::vector<std::vector<float>> lines;
	for (size_t i = 0; i < lines_lsd->size; i++)
	{
		float xs = lines_lsd->values[i*dim + 0];
		float ys = lines_lsd->values[i*dim + 1];
		float xe = lines_lsd->values[i*dim + 2];
		float ye = lines_lsd->values[i*dim + 3];
		int len = sqrt((xs - xe)*(xs - xe) + (ys - ye)*(ys - ye));
		if (len < 12) continue;
		float dx = (xe - xs) / len;
		float dy = (ye - ys) / len;

		// 
		std::vector<float> h_neighs;
		for (size_t j = 0; j < len; j++)
		{
			int x0 = int(xs + j * dx);
			int y0 = int(ys + j * dy);

			for (int m = -win_size; m < win_size; m++)
			{
				for (int n = -win_size; n < win_size; n++)
				{
					int x = x0 + m;
					int y = y0 + n;
					if (x < 0 || x >= cols || y < 0 || y >= rows) continue;

					float h = img_cropped_.at<float>(y, x);
					if (h != 0) h_neighs.push_back(h);
				}
			}
		}
		std::sort(h_neighs.begin(), h_neighs.end());

		float h_avg = 0.0;
		for (size_t j = 1; j < h_neighs.size()/10; j++)
		{
			h_avg += h_neighs[h_neighs.size() - j];
		}
		h_avg /= (h_neighs.size() / 10);

		//
		std::vector<float> line_i(5);
		line_i[4] = h_avg;  
		line_i[0] = xs; line_i[1] = ys;
		line_i[2] = xe; line_i[3] = ye;
		lines.push_back(line_i);
	}

	// find visible lines for each site
	cv::Mat mask1(rows, cols, CV_32FC1, cv::Scalar(0));
	cv::Mat mask2(rows, cols, CV_32FC1, cv::Scalar(0));
	cv::Mat mask3(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));
	for (size_t i = 0; i < lines.size(); i++)
	{
		cv::line(mask1, cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), cv::Scalar(lines[i][4]), 2);
		cv::line(mask2, cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), cv::Scalar(i), 2);

		int r = rand() % 255;
		int g = rand() % 255;
		int b = rand() % 255;
		cv::line(mask3, cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), cv::Scalar(r, g, b), 2);
		std::string linename = std::to_string(i);
		cv::Point mid_pt((lines[i][0] + lines[i][2]) / 2, (lines[i][1] + lines[i][3]) / 2);
		cv::putText(mask3, linename, mid_pt, 1, 1, cv::Scalar(r, g, b), 1);
	}
	cv::imwrite("F:\\img_line3.jpg", mask3);
	cv::imwrite("F:\\img_line2.jpg", 255*mask2);


	int angle_num = 180;
	float angle_step = 2 * CV_PI / angle_num;
	int num_sites = site_pts_.size();
	site_visible_lines_.resize(num_sites);
	for (size_t i = 0; i < site_pts_.size(); i++)
	{
		if (i == 1442)
		{
			int aa = 0;
		}
		int x_site = int(site_pts_[i].x);
		if (x_site >= cols) x_site = cols - 1;
		int y_site = int(site_pts_[i].y);
		if (y_site >= cols) y_site = rows - 1;
		float h_site = site_pts_[i].z;

		std::vector<int> visible_lines_id_i;
		for (size_t j = 0; j < angle_num; j++)
		{
			float angle = j * angle_step;
			float dx = sin(angle);
			float dy = -cos(angle);

			float x = site_pts_[i].x, y = site_pts_[i].y;
			int x_former = 0, y_former = 0;
			int x_cur = 0, y_cur = 0;

			float angle_ratio_max = -100000.0;
			while (true)
			{
				x += dx;
				y += dy;
				x_cur = int(x);
				y_cur = int(y);

				if (x_cur < 0 || x_cur >= cols || y_cur < 0 || y_cur >= rows) break;
				if (x_cur == x_former && y_cur == y_former) continue;
				float h_cur = mask1.at<float>(y_cur, x_cur);
				if (h_cur == 0) continue;

				float dis = std::sqrt((x - site_pts_[i].x)*(x - site_pts_[i].x) + (y - site_pts_[i].y)*(y - site_pts_[i].y));
				float angle_ratio_cur = (h_cur - h_site) / dis;
				if (angle_ratio_cur > angle_ratio_max)
				{
					angle_ratio_max = angle_ratio_cur;
					visible_lines_id_i.push_back(mask2.at<float>(y_cur, x_cur));
				}
			}
		}

		std::sort(visible_lines_id_i.begin(), visible_lines_id_i.end());
		int id_former = visible_lines_id_i[0];
		int id_count = 1;
		std::vector<int> lines_temp;
		for (size_t j = 0; j < visible_lines_id_i.size(); j++)
		{
			if (visible_lines_id_i[j] == id_former) 
				id_count++;
			else
			{
				if (id_count >= 5)
				{
					for (size_t m = 0; m < 5; m++)
					{
						site_visible_lines_[i].push_back(lines[id_former][m]);
						lines_temp.push_back(id_former);
					}
				}
				id_former = visible_lines_id_i[j];
				id_count = 1;
			}
		}
		if (id_count >= 5)
		{
			for (size_t m = 0; m < 5; m++)
			{
				site_visible_lines_[i].push_back(lines[id_former][m]);
				lines_temp.push_back(id_former);
			}
		}
	}
}

void Database::WriteoutVisibleLines()
{
	std::string path_vlines = fold_dsm_ + "\\vlines.txt";
	ofstream of(path_vlines);

	of << site_visible_lines_.size() <<std::endl;
	for (size_t i = 0; i < site_visible_lines_.size(); i++)
	{
		of << site_visible_lines_[i].size() << " ";
		for (size_t j = 0; j < site_visible_lines_[i].size(); j++)
		{
			of << site_visible_lines_[i][j] << " ";
		}
		of << std::endl;
	}
	of.close();
}

void Database::ReadinVisibleLines()
{
	std::string path_vlines = fold_dsm_ + "\\vlines.txt";
	ifstream reader(path_vlines);

	int num_site = 0;
	reader >> num_site;
	site_visible_lines_.resize(num_site);
	for (size_t i = 0; i < num_site; i++)
	{
		int num_lines = 0;
		reader >> num_lines;
		site_visible_lines_[i].resize(num_lines);
		for (size_t j = 0; j < site_visible_lines_[i].size(); j++)
		{
			reader >> site_visible_lines_[i][j];
		}
	}
	reader.close();
}

}  // namespace objectsfm
