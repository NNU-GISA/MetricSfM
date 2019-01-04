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

#ifndef MIN_
#define MIN_(a,b) ( ((a)<(b)) ? (a):(b) )
#endif // !MIN_

#define ASSERT_MSG(expr, msg) \
	if (!(expr)) { \
		std::cerr << msg << std::endl; \
		std::exit(EXIT_FAILURE); \
	} \

#include "dense_reconstruction.h"

#include <fstream>
#include <filesystem>
#include <iomanip>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include "libsgm.h"
#include "elas.h"
#include "utils/basic_funcs.h"


namespace objectsfm {

	DenseReconstruction::DenseReconstruction()
	{
	}

	DenseReconstruction::~DenseReconstruction()
	{
	}

	void DenseReconstruction::Run(std::string fold)
	{
		fold_img = fold + "\\rgb";

		// step1: read in the sfm result
		std::string sfm_file = fold + "\\sfm_sure.txt";
		ReadinPoseFile(sfm_file);

		// step2: generate depth image 
		//fold_output = fold + "\\dense_sgm_result";
		//SGMDense();
		fold_output = fold + "\\dense_elas_result";
		ELASDense();
	}

	void DenseReconstruction::ReadinPoseFile(std::string sfm_file)
	{
		// in the same format as sure
		std::ifstream ff(sfm_file);
		std::string temp;
		for (size_t i = 0; i < 8; i++) {
			std::getline(ff, temp);
		}

		std::string name;
		int w, h;
		double k1, k2, k3, p1, p2;
		while (!ff.eof())
		{
			cv::Mat_<double> K(3, 3);
			cv::Mat_<double> R(3, 3);
			cv::Mat_<double> t(3, 1);

			ff >> name >> w >> h;
			for (size_t i = 0; i < 3; i++) {
				for (size_t j = 0; j < 3; j++) {
					ff >> K(i, j);
				}
			}
			ff >> k1 >> k2 >> k3 >> p1 >> p2;

			for (size_t i = 0; i < 3; i++) {
				ff >> t(i, 0);
			}

			for (size_t i = 0; i < 3; i++) {
				for (size_t j = 0; j < 3; j++) {
					ff >> R(i, j);
				}
			}

			names.push_back(name);
			Ks.push_back(cv::Mat(K));
			Rs.push_back(cv::Mat(R));
			ts.push_back(cv::Mat(t));
		}
	}

	void DenseReconstruction::SGMDense()
	{
		if (!std::experimental::filesystem::exists(fold_output)) {
			std::experimental::filesystem::create_directory(fold_output);
		}

		Rs_new.resize(Rs.size());
		ts_new.resize(ts.size());

		// apply sgm on each image pair
		for (size_t i = 0; i < names.size() - 1; i++)
		{
			std::cout << i << std::endl;
			std::string file_left = fold_img + "\\" + names[i + 1];
			cv::Mat left = cv::imread(file_left, 0);

			std::string file_right = fold_img + "\\" + names[i];
			cv::Mat right = cv::imread(file_right, 0);

			// generate epipolar image
			cv::Mat Rleft_, Rright_;
			EpipolarRectification(left, Ks[i + 1], Rs[i + 1], ts[i + 1], right, Ks[i], Rs[i], ts[i], true, Rleft_, Rright_);

			// get disparity via sgm
			ASSERT_MSG(left.size() == right.size() && left.type() == right.type(), "input images must be same size and type.");
			ASSERT_MSG(left.type() == CV_8U || left.type() == CV_16U, "input image format must be CV_8U or CV_16U.");
			ASSERT_MSG(options_.disp_size == 64 || options_.disp_size == 128, "disparity size must be 64 or 128.");

			int bits = 0;
			switch (left.type()) {
			case CV_8UC1: bits = 8; break;
			case CV_16UC1: bits = 16; break;
			default:
				std::cerr << "invalid input image color format" << left.type() << std::endl;
				std::exit(EXIT_FAILURE);
			}

			//cv::resize(left, left, cv::Size(left.cols / 2, left.rows / 2));
			//cv::resize(right, right, cv::Size(right.cols / 2, right.rows / 2));

			sgm::StereoSGM::Parameters para(10, 120, options_.uniqueness, false);
			sgm::StereoSGM ssgm(left.cols, left.rows, options_.disp_size, bits, 8, sgm::EXECUTE_INOUT_HOST2HOST, para);

			cv::Mat disparity(cv::Size(left.cols, left.rows), CV_8UC1);
			ssgm.execute(left.data, right.data, disparity.data);

			// disparity to depth
			double baseline = cv::norm(ts[i + 1] - ts[i]);
			double focal = Ks[i + 1].at<double>(0, 0);

			cv::Mat depth(cv::Size(left.cols, left.rows), CV_8UC1, cv::Scalar(0));
			uchar* ptr_depth = depth.data;
			uchar* ptr_disparity = disparity.data;
			for (size_t m = 0; m < depth.rows; m++)
			{
				for (size_t n = 0; n < depth.cols; n++)
				{
					double disparity_temp = *ptr_disparity;
					if (disparity_temp != 0) {
						double depth_temp = 200 * baseline * focal / disparity_temp;
						if (depth_temp > 255 || depth_temp < 0) {
							depth_temp = 0;
						}
						*ptr_depth = depth_temp;
					}
					ptr_disparity++; ptr_depth++;
				}
			}
			std::string file_depth = fold_output + "\\" + names[i + 1];
			cv::imwrite(file_depth, depth);

			// pose of epipolar image
			Rs_new[i + 1] = Rs[i + 1] * Rleft_;
			cv::Mat c = -Rs[i + 1].inv() * ts[i + 1];
			ts_new[i + 1] = -Rs_new[i + 1] * c;
		}
	}

	void DenseReconstruction::ELASDense()
	{
		if (!std::experimental::filesystem::exists(fold_output)) {
			std::experimental::filesystem::create_directory(fold_output);
		}

		std::string fold_txt = fold_output + "\\txt";
		if (!std::experimental::filesystem::exists(fold_txt)) {
			std::experimental::filesystem::create_directory(fold_txt);
		}

		std::string fold_visualize = fold_output + "\\visualize";
		if (!std::experimental::filesystem::exists(fold_visualize)) {
			std::experimental::filesystem::create_directory(fold_visualize);
		}

		std::string fold_depth = fold_output + "\\depth";
		if (!std::experimental::filesystem::exists(fold_depth)) {
			std::experimental::filesystem::create_directory(fold_depth);
		}

		// apply elas on each image pair
		int count_output = 0;
		for (size_t i = 0; i < names.size() - 1; i++)
		{
			std::cout << i << std::endl;
			std::string file_left = fold_img + "\\" + names[i + 1];
			cv::Mat left = cv::imread(file_left);

			std::string file_right = fold_img + "\\" + names[i];
			cv::Mat right = cv::imread(file_right);

			// generate epipolar image
			cv::Mat Rleft_, Rright_;
			EpipolarRectification(left, Ks[i + 1], Rs[i + 1], ts[i + 1], right, Ks[i], Rs[i], ts[i], false, Rleft_, Rright_);
			cv::Mat left_rgb = left.clone();
			cv::cvtColor(left, left, CV_RGB2GRAY);
			cv::cvtColor(right, right, CV_RGB2GRAY);

			// elas
			const int32_t dims[3] = { left.cols,left.rows,left.cols };
			float* d_left = (float*)malloc(left.cols*left.rows * sizeof(float));
			float* d_right = (float*)malloc(left.cols*left.rows * sizeof(float));

			Elas::parameters param(Elas::ROBOTICS);
			param.disp_min = 0;
			param.disp_max = 400;

			Elas elas(param);
			elas.process(left.data, right.data, d_left, d_right, dims);

			// convert the left desprity into depth
			// distance = focal_length * baseline distance / disparity
			cv::Mat c_left  = -Rs[i + 1].inv() * ts[i + 1];
			cv::Mat c_right = -Rs[i].inv() * ts[i];
			double base_line = cv::norm(c_left - c_right);
			double f_left = Ks[i + 1].at<double>(0, 0);

			int size = left.cols * left.rows;
			cv::Mat depth_left(left.rows, left.cols, CV_16UC1, cv::Scalar(0));
			float* ptr_left = d_left;
			unsigned short* ptr_depth = (unsigned short*)depth_left.data;
			for (int j = 0; j < size; j++)
			{
				double disparity_j = *ptr_left;
				if (disparity_j > 0) 
				{
					double depth_j = 20 * f_left * base_line / disparity_j;
					*ptr_depth = MIN_(depth_j, 600);
				}
				ptr_left++; ptr_depth++;
			}
			std::cout << "base_line " << base_line << " f_left " << f_left << std::endl;

			// save out
			std::string name = std::to_string(count_output);
			name = std::string(8 - name.length(), '0') + name;

			cv::imwrite(fold_visualize + "\\" + name + ".jpg", left_rgb);
			cv::imwrite(fold_depth + "\\" + name + ".png", depth_left);

			cv::Mat R_epi = Rs[i + 1] * Rleft_;
			cv::Mat c_epi = -Rs[i + 1].inv() * ts[i + 1];
			cv::Mat t_epi = -R_epi * c_epi;
			cv::Mat M_epi = (cv::Mat_<double>(3, 4) << R_epi.at<double>(0, 0), R_epi.at<double>(0, 1), R_epi.at<double>(0, 2), t_epi.at<double>(0, 0),
				                                       R_epi.at<double>(1, 0), R_epi.at<double>(1, 1), R_epi.at<double>(1, 2), t_epi.at<double>(1, 0),
				                                       R_epi.at<double>(2, 0), R_epi.at<double>(2, 1), R_epi.at<double>(2, 2), t_epi.at<double>(2, 0));
			cv::Mat P_epi = Ks[i + 1] * M_epi;
			std::ofstream ff(fold_txt + "\\" + name + ".txt");
			ff << std::fixed << std::setprecision(8);
			//ff << "CONTOUR" << std::endl;
			for (size_t m = 0; m < 3; m++) {
				for (size_t n = 0; n < 3; n++) {
					ff << R_epi.at<double>(m, n) << " ";
				}
				ff << std::endl;
			}
			ff << t_epi.at<double>(0, 0) << " " << t_epi.at<double>(1, 0) << " " << t_epi.at<double>(2, 0) << std::endl;
			ff.close();

			free(d_left);
			free(d_right);

			count_output++;
		}
	}

	void DenseReconstruction::EpipolarRectification(cv::Mat &img1, cv::Mat K1, cv::Mat R1, cv::Mat t1,
		cv::Mat &img2, cv::Mat K2, cv::Mat R2, cv::Mat t2,
		bool write_out, cv::Mat &R1_, cv::Mat &R2_)
	{
		cols = img1.cols;
		rows = img2.rows;
		cv::Size img_size(cols, rows);

		//
		cv::Mat R21 = R2 * R1.inv();
		cv::Mat t21 = -R21 * t1 + t2;
		cv::Mat dist(5, 1, CV_64FC1, cv::Scalar(0));

		//
		cv::Mat P1_, P2_, Q_;
		cv::stereoRectify(K1, dist, K2, dist, img_size, R21, t21, R1_, R2_, P1_, P2_, Q_, cv::CALIB_ZERO_DISPARITY, -1, img_size);

		cv::Mat mapx1, mapy1, mapx2, mapy2;
		cv::initUndistortRectifyMap(K1, dist, R1_, P1_, img_size, CV_32FC1, mapx1, mapy1);
		cv::initUndistortRectifyMap(K2, dist, R2_, P2_, img_size, CV_32FC1, mapx2, mapy2);

		cv::Mat img_rectified1, img_rectified2;
		cv::remap(img1, img_rectified1, mapx1, mapy1, cv::INTER_LINEAR);
		cv::remap(img2, img_rectified2, mapx2, mapy2, cv::INTER_LINEAR);
		img1 = img_rectified1;
		img2 = img_rectified2;

		if (write_out)
		{
			cv::imwrite("F:\\img_rectified1.jpg", img_rectified1);
			cv::imwrite("F:\\img_rectified2.jpg", img_rectified2);
		}
	}

	void DenseReconstruction::SavePoseFile(std::string sfm_file)
	{
		//
		FILE * fp;
		fp = fopen(sfm_file.c_str(), "w+");
		fprintf(fp, "%s\n", "fileName imageWidth imageHeight");
		fprintf(fp, "%s\n", "camera matrix K [3x3]");
		fprintf(fp, "%s\n", "radial distortion [3x1]");
		fprintf(fp, "%s\n", "tangential distortion [2x1]");
		fprintf(fp, "%s\n", "camera position t [3x1]");
		fprintf(fp, "%s\n", "camera rotation R [3x3]");
		fprintf(fp, "%s\n\n", "camera model P = K [R|-Rt] X");


		for (size_t i = 1; i < Rs_new.size(); i++)
		{
			fprintf(fp, "%s %d %d\n", names[i], cols, rows);
			fprintf(fp, "%.8lf %.8lf %.8lf\n", Ks[i].at<double>(0, 0), 0, Ks[i].at<double>(0, 2));
			fprintf(fp, "%.8lf %.8lf %.8lf\n", 0, Ks[i].at<double>(1, 1), Ks[i].at<double>(1, 2));
			fprintf(fp, "%d %d %d\n", 0, 0, 1);
			fprintf(fp, "%.8lf %.8lf %.8lf\n", 0, 0, 0);
			fprintf(fp, "%.8lf %.8lf\n", 0, 0);
			fprintf(fp, "%.8lf %.8lf %.8lf\n", ts_new[i].at<double>(0, 0), ts_new[i].at<double>(1, 0), ts_new[i].at<double>(2, 0));
			fprintf(fp, "%.8lf %.8lf %.8lf\n", Rs_new[i].at<double>(0, 0), Rs_new[i].at<double>(0, 1), Rs_new[i].at<double>(0, 2));
			fprintf(fp, "%.8lf %.8lf %.8lf\n", Rs_new[i].at<double>(1, 0), Rs_new[i].at<double>(1, 1), Rs_new[i].at<double>(1, 2));
			fprintf(fp, "%.8lf %.8lf %.8lf\n", Rs_new[i].at<double>(2, 0), Rs_new[i].at<double>(2, 1), Rs_new[i].at<double>(2, 2));
		}

		fclose(fp);
	}

	void DenseReconstruction::Depth2Points(std::string fold)
	{
		double scale = 20.0;
		double focal = 900.95001200;

		std::string fold_point = fold + "\\pointcloud";
		if (!std::experimental::filesystem::exists(fold_point)) {
			std::experimental::filesystem::create_directory(fold_point);
		}

		// 
		std::string fold_rgb = fold + "\\visualize";
		std::string fold_depth = fold + "\\depth";
		std::string fold_txt = fold + "\\txt";

		std::vector<cv::String> img_paths;
		cv::glob(fold_rgb + "/*.jpg", img_paths, false);
		for (size_t i = 0; i < img_paths.size(); i++)
		{
			std::string path_i = std::string(img_paths[i].c_str());
			int idxs = path_i.find_last_of("\\");
			int idxe = path_i.find_last_of(".");
			std::string name_i = path_i.substr(idxs + 1, idxe - idxs - 1);

			std::string path_rgb = fold_rgb + "\\" + name_i + ".jpg";
			std::string path_depth = fold_depth + "\\" + name_i + ".png";
			std::string path_txt = fold_txt + "\\" + name_i + ".txt";

			cv::Mat depth = cv::imread(path_depth, -1);
			//cv::Mat rgb = cv::imread(path_rgb);

			std::ifstream ff(path_txt);
			cv::Mat R = cv::Mat(3, 3, CV_64FC1);
			cv::Mat t = cv::Mat(3, 1, CV_64FC1);
			for (size_t m = 0; m < 3; m++)
			{
				for (size_t n = 0; n < 3; n++)
				{
					double v = 0;
					ff >> v;
					*((double*)R.data + m * 3 + n) = v;
				}
			}
			for (size_t m = 0; m < 3; m++)
			{
				double v = 0;
				ff >> v;
				*((double*)t.data + m) = v;
			}
			ff.close();

			//
			double px = depth.cols / 2;
			double py = depth.rows / 2;
			short* ptr_depth = (short*)depth.data;
			//uchar* ptr_rgb = rgb.data;
			std::vector<cv::Point3d> pts;
			std::vector<cv::Point3d> colors;
			for (size_t m = 0; m < depth.rows; m++)
			{
				for (size_t n = 0; n < depth.cols; n++)
				{
					if (*ptr_depth != 0)
					{
						double z = *ptr_depth / scale;
						double x = z * (n - px) / focal;
						double y = z * (m - py) / focal;
						cv::Mat Xc = (cv::Mat_<double>(3, 1) << x, y, z);
						cv::Mat Xw = R.inv()*(Xc - t);

						pts.push_back(cv::Point3d(Xw.at<double>(0, 0), Xw.at<double>(1, 0), Xw.at<double>(2, 0)));
						//colors.push_back(cv::Point3d(ptr_rgb[0], ptr_rgb[1], ptr_rgb[2]));
						colors.push_back(cv::Point3d(0, 0, 0));
					}
					ptr_depth++;
					//ptr_rgb++;
				}
			}

			// write out
			std::string path_point = fold_point + "\\" + name_i + ".txt";
			std::ofstream of(path_point);
			for (size_t m = 0; m < pts.size(); m += 2)
			{
				of << pts[m].x << " " << pts[m].y << " " << pts[m].z << " ";
				of << colors[m].x << " " << colors[m].y << " " << colors[m].z << std::endl;
			}
			of.close();
		}
	}

}  // namespace objectsfm
