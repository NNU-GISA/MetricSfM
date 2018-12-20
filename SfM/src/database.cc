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
#include "utils/find_polynomial_roots_companion_matrix.h"

#include "feature/feature_extractor_vl_sift.h"
#include "feature/feature_extractor_opencv.h"
#include "feature/feature_extractor_cuda_sift.h"


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

	//if (options.extract_gist)
	//{
	//	cv::Mat img_rgb = cv::imread(image_paths_[idx]);

	//	Gist *gister = new Gist();
	//	Gist::Params *gist_paras = new Gist::Params();
	//	vector<int> vec_s32_or(4, 8);   // 4 scales, 8 orientations
	//	gist_paras->vecIntOrientation = vec_s32_or;
	//	Gist::GistDescriptor *gist_descriptor = gister->compute(img, gist_paras);
	//	gist_descriptors_[idx] = vector<float>(gist_descriptor->pfGistDescriptor, gist_descriptor->pfGistDescriptor + gist_descriptor->s32Length);

	//	delete gist_paras;
	//	delete gister;
	//}
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

}  // namespace objectsfm
