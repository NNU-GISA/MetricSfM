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

#include "slam_gps.h"

#include <fstream>
#include <filesystem>
#include <iomanip>

#include <Eigen/Core>
#include "ceres/ceres.h"
#include <opencv2/opencv.hpp>

#include "utils/basic_funcs.h"
#include "utils/converter_utm_latlon.h"
#include "utils/ellipsoid_utm_info.h"
#include "utils/reprojection_error_pose_cam_xyz.h"
#include "utils/reprojection_error_pose_xyz.h"
#include "utils/gps_error_pose.h"


namespace objectsfm {

	SLAMGPS::SLAMGPS()
	{
		rows_ = 750;
		cols_ = 1000;
	}

	SLAMGPS::~SLAMGPS()
	{
	}

	void SLAMGPS::Run(std::string file_slam, std::string file_gps, std::string file_rgb)
	{
		// read in slam information
		ReadinSLAM(file_slam);

		// read in gps
		std::map<int, cv::Point2d> gps_info;
		ReadinGPS(file_gps, gps_info);

		// accosiation
		AssociateCameraGPS(file_rgb, gps_info);

		GrawGPS("F:\\pos_gps.bmp");
		GrawSLAM("F:\\pos_slam1.bmp");

		// do adjustment
		FullBundleAdjustment();

		// write out
		std::string outfile = "F:\\cam_pts.txt";
		WriteCameraPointsOut(outfile);
		GrawSLAM("F:\\pos_slam2.bmp");
	}

	void SLAMGPS::ReadinSLAM(std::string file_slam)
	{
		std::ifstream ff(file_slam);

		// cameras
		int n_frames = 0;
		ff >> n_frames;
		cams_.resize(n_frames);
		int id;
		double tt;
		Eigen::Vector3d t;
		Eigen::Matrix3d R;
		std::map<int, int> cam_index;
		for (size_t i = 0; i < n_frames; i++)
		{
			cams_[i] = new Camera();
			ff >> id >> tt >> fx >> fy >> cx >> cy;
			for (size_t m = 0; m < 3; m++) {
				ff >> t(m);
			}
			for (size_t m = 0; m < 3; m++) {
				for (size_t n = 0; n < 3; n++)
				{
					ff >> R(m,n);
				}
			}
			cams_[i]->SetRTPose(R, t);
			cams_[i]->SetID(id);

			cam_index.insert(std::pair<int, int>(id, i));
		}

		// points
		int n_pts = 0;
		ff >> n_pts;
		pts_.resize(n_pts);
		double x, y, z;
		for (size_t i = 0; i < n_pts; i++)
		{
			pts_[i] = new Point3D();
 			ff >> x >> y >> z;
			pts_[i]->data[0] = x;
			pts_[i]->data[1] = y;
			pts_[i]->data[2] = z;

			int n_obs = 0;
			ff >> n_obs;
			int id_cam;
			double px, py;
			for (size_t j = 0; j < n_obs; j++)
			{
				ff >> id_cam >> px >> py;

				std::map<int, int >::iterator iter_i_c = cam_index.find(id_cam);
				if (iter_i_c == cam_index.end())
				{
					continue;
				}
				int idx_cam = iter_i_c->second;

				pts_[i]->AddObservation(cams_[idx_cam], px - cx, py - cy, j);
			}
		}

		// camera models
		cam_models_.resize(1);
		cam_models_[0] = new CameraModel(cam_models_.size(), rows_, cols_, (fx+fy)/2.0, 0.0, "lu", "lu");
		cam_models_[0]->SetIntrisicParas((fx + fy) / 2.0, cx, cy);
		for (size_t i = 0; i < cams_.size(); i++)
		{
			cams_[i]->AssociateCamereModel(cam_models_[0]);
		}
	}

	void SLAMGPS::ReadinGPS(std::string file_gps, std::map<int, cv::Point2d> &gps_info )
	{
		int ellipsoid_id = 23; // WGS-84
		std::string zone_id = "17N";

		std::ifstream ff(file_gps);
		int id;
		double lat, lon, alt;
		double x, y;
		while (!ff.eof())
		{
			ff >> id >> lat >> lon >> alt;
			LLtoUTM(ellipsoid_id, lat, lon, y, x, (char*)zone_id.c_str());
			gps_info.insert(std::pair<int, cv::Point2d>(id, cv::Point2d(x, y)));
		}
		ff.close();
	}

	void SLAMGPS::AssociateCameraGPS(std::string file_rgb, std::map<int, cv::Point2d>& gps_info)
	{
		std::ifstream ff(file_rgb);
		std::string s0;
		std::getline(ff, s0);
		std::getline(ff, s0);
		std::getline(ff, s0);


		std::vector<cv::Point2d> gps_all_frame;
		while (!ff.eof())
		{
			std::string s;
			std::getline(ff, s);
			size_t loc1 = s.find_last_of('/');
			size_t loc2 = s.find_last_of('.');
			std::string name = s.substr(loc1 + 1, loc2 - loc1 - 1);
			int id = std::stoi(name);

			std::map<int, cv::Point2d>::iterator iter = gps_info.find(id);
			if (iter == gps_info.end())
			{
				continue;
			}
			gps_all_frame.push_back(iter->second);
		}

		// gps of keyframes
		cams_gps_.resize(cams_.size());
		for (size_t i = 0; i < cams_.size(); i++)
		{
			int id_frame = cams_[i]->id_;
			cams_gps_[i] = gps_all_frame[id_frame];
		}
	}

	void SLAMGPS::FullBundleAdjustment()
	{
		ceres::Problem problem;
		ceres::Solver::Options options;
		ceres::Solver::Summary summary;

		options.max_num_iterations = 100;
		options.minimizer_progress_to_stdout = true;
		options.num_threads = 1;
		options.linear_solver_type = ceres::DENSE_SCHUR;

		// add reprojection error
		int count1 = 0;
		for (size_t i = 0; i < pts_.size(); i++)
		{
			std::map<int, Camera*>::iterator iter_cams = pts_[i]->cams_.begin();
			std::map<int, Eigen::Vector2d>::iterator iter_pts = pts_[i]->pts2d_.begin();
			while (iter_cams != pts_[i]->cams_.end())
			{
				ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
				ceres::CostFunction* cost_function;

				//cost_function = ReprojectionErrorPoseCamXYZ::Create(iter_pts->second(0), iter_pts->second(1), pts_[i]->weight);
				//problem.AddResidualBlock(cost_function, loss_function,
				//	iter_cams->second->data, iter_cams->second->cam_model_->data, pts_[i]->data);

				cost_function = ReprojectionErrorPoseXYZ::Create(iter_pts->second(0), iter_pts->second(1), iter_cams->second->cam_model_->data, pts_[i]->weight);
				problem.AddResidualBlock(cost_function, loss_function, iter_cams->second->data, pts_[i]->data);

				iter_cams++;
				iter_pts++;

				count1++;
			}
		}

		// add gps-topology error
		int count2 = 0;
		if (1)
		{
			int step = 5;
			double weight = 100;
			for (int i = 30; i < cams_.size(); i++)
			{
				int id1 = i;
				int k = (i - 6) / 3;
				for (int j = 0; j < k; j++)
				{
					int id2 = j;
					int id3 = k + j;

					cv::Point2d v12 = cams_gps_[id2] - cams_gps_[id1];
					double l12 = sqrt(v12.x*v12.x + v12.y*v12.y);

					cv::Point2d v13 = cams_gps_[id3] - cams_gps_[id1];
					double l13 = sqrt(v13.x*v13.x + v13.y*v13.y);

					cv::Point2d v23 = cams_gps_[id3] - cams_gps_[id2];
					double l23 = sqrt(v23.x*v23.x + v23.y*v23.y);

					if (l12 == 0 || l13 == 0 || l23 == 0)
					{
						continue;
					}

					double angle1 = acos((v12.x*v13.x + v12.y*v13.y) / l12 / l13);
					double angle2 = acos((-v12.x*v23.x - v12.y*v23.y) / l12 / l23);
					double angle3 = acos((v13.x*v23.x + v13.y*v23.y) / l13 / l23);

					ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
					ceres::CostFunction* cost_function;
					cost_function = GPSErrorPose::Create(angle1, angle2, angle3, weight);
					problem.AddResidualBlock(cost_function, loss_function, cams_[id1]->data, cams_[id2]->data, cams_[id3]->data);

					count2++;
				}
			}
		}
		std::cout << "count1 " << count1 << " count2 " << count2 << std::endl;

		ceres::Solve(options, &problem, &summary);
		std::cout << summary.FullReport() << "\n";

		// update parameters
		for (size_t i = 0; i < cams_.size(); i++)
		{
			cams_[i]->UpdatePoseFromData();
		}
	}

	void SLAMGPS::SaveModel()
	{
		std::string fold = "";
		if (!std::experimental::filesystem::exists(fold))
		{
			std::experimental::filesystem::create_directory(fold);
		}

		// write out point cloud
		std::string path_ptcloud = fold + "//pts_cams.txt";
		WriteCameraPointsOut(path_ptcloud);

		// write out points
		std::string path_pt3d = fold + "//pts.txt";
		std::ofstream ofs_pt(path_pt3d);
		ofs_pt << pts_.size() << std::endl;
		ofs_pt.precision(20);
		for (size_t i = 0; i < pts_.size(); i++)
		{
			ofs_pt << pts_[i]->data[0] << " " << pts_[i]->data[1] << " " << pts_[i]->data[2] << std::endl;
		}
		ofs_pt.close();

		// write out cameras
		std::string path_cam = fold + "//cams.txt";
		std::ofstream ofs_cam(path_cam);
		ofs_cam << cams_.size() << std::endl;
		ofs_cam.precision(20);
		for (size_t i = 0; i < cams_.size(); i++)
		{
			ofs_cam << cams_[i]->id_img_ << " ";
			ofs_cam << cams_[i]->data[0] << " " << cams_[i]->data[1] << " " << cams_[i]->data[2] << " ";    // angle-axis
			ofs_cam << cams_[i]->pos_ac_.c(0) << " " << cams_[i]->pos_ac_.c(1) << " " << cams_[i]->pos_ac_.c(2) << " ";  // camera center
			ofs_cam << cams_[i]->cam_model_->data[0] << " " << cams_[i]->cam_model_->data[1] << " " << cams_[i]->cam_model_->data[2] << std::endl;    // f, k1, k2
		}
		ofs_cam.close();

		// write out points of each camera
		for (size_t i = 0; i < cams_.size(); i++)
		{
			std::string path_cam_pts = fold + "//point_cam" + std::to_string(i) + ".txt";
			std::ofstream ofs_cam_pts(path_cam_pts);
			ofs_cam_pts.precision(20);

			for (auto iter = cams_[i]->pts_.begin(); iter != cams_[i]->pts_.end(); iter++)
			{
				ofs_cam_pts << iter->second->data[0] << " " << iter->second->data[1] << " " << iter->second->data[2] << std::endl;
			}
			ofs_cam_pts.close();
		}

		// clearn current results
		for (size_t i = 0; i < pts_.size(); i++)
		{
			delete pts_[i];
		}
		pts_.clear();

		for (size_t i = 0; i < cams_.size(); i++)
		{
			delete cams_[i];
		}
		cams_.clear();

		for (size_t i = 0; i < cam_models_.size(); i++)
		{
			delete cam_models_[i];
		}
		cam_models_.clear();
	}

	void SLAMGPS::WriteCameraPointsOut(std::string path)
	{
		std::ofstream ofs(path);
		if (!ofs.is_open())
		{
			return;
		}
		ofs << std::fixed << std::setprecision(8);

		float span = 100;
		for (size_t i = 0; i < pts_.size(); i++)
		{
			if (abs(pts_[i]->data[0]) > span || abs(pts_[i]->data[1]) > span || abs(pts_[i]->data[2]) > span)
			{
				continue;
			}

			ofs << pts_[i]->data[0] << " " << pts_[i]->data[1] << " " << pts_[i]->data[2] 
				<< " " << 0 << " " << 0 << " " << 0 << std::endl;
		}

		double scale = (cams_[1]->pos_ac_.c- cams_[0]->pos_ac_.c).squaredNorm() / 10;
		//double scale = 0.01;
		for (size_t i = 0; i < cams_.size(); i++)
		{
			std::vector<Eigen::Vector3d> axis(3);
			axis[0] = cams_[i]->pos_rt_.R.inverse() * Eigen::Vector3d(1, 0, 0);
			axis[1] = cams_[i]->pos_rt_.R.inverse() * Eigen::Vector3d(0, 1, 0);
			axis[2] = cams_[i]->pos_rt_.R.inverse() * Eigen::Vector3d(0, 0, 1);

			std::vector<Eigen::Vector3d> cam_pts;
			GenerateCamera3D(cams_[i]->pos_ac_.c, axis, cams_[i]->cam_model_->f_,
				cams_[i]->cam_model_->w_, cams_[i]->cam_model_->h_, scale, cam_pts);

			for (size_t j = 0; j < cam_pts.size(); j++)
			{
				if (abs(cam_pts[j](0)) > span || abs(cam_pts[j](1)) > span || abs(cam_pts[j](2)) > span)
				{
					continue;
				}

				ofs << cam_pts[j](0) << " " << cam_pts[j](1) << " " << cam_pts[j](2)
					<< " " << 255 << " " << 0 << " " << 0 << std::endl;
			}
		}
		ofs.close();
	}

	void SLAMGPS::GrawGPS(std::string path)
	{
		double xmin = 1000000000000;
		double xmax = -xmin;
		double ymin = xmin;
		double ymax = -ymin;
		for (size_t i = 0; i < cams_gps_.size(); i++)
		{
			double x = cams_gps_[i].x;
			double y = cams_gps_[i].y;
			if (x < xmin) xmin = x;
			if (x > xmax) xmax = x;
			if (y < ymin) ymin = y;
			if (y > ymax) ymax = y;
		}

		double xspan = xmax - xmin;
		double yspan = ymax - ymin;
		double span = MAX(xspan, yspan);
		double step = (span + 1) / 2000;
		int h = yspan / step;
		int w = xspan / step;
		cv::Mat img(h, w, CV_8UC3, cv::Scalar(255, 255, 255));
	
		for (size_t i = 0; i < cams_gps_.size(); i++)
		{
			double x = cams_gps_[i].x;
			double y = cams_gps_[i].y;
			int xx = (x - xmin) / step;
			int yy = (y - ymin) / step;
			cv::circle(img, cv::Point(xx, yy), 5, cv::Scalar(0, 0, 255));
			cv::putText(img, std::to_string(i), cv::Point(xx, yy), 1, 1, cv::Scalar(255, 0, 0));
		}
		cv::imwrite(path, img);
	}

	void SLAMGPS::GrawSLAM(std::string path)
	{
		double xmin = 1000000000000;
		double xmax = -xmin;
		double ymin = xmin;
		double ymax = -ymin;
		for (size_t i = 0; i < cams_.size(); i++)
		{
			double x = cams_[i]->pos_ac_.c(0);
			double y = cams_[i]->pos_ac_.c(2);
			if (x < xmin) xmin = x;
			if (x > xmax) xmax = x;
			if (y < ymin) ymin = y;
			if (y > ymax) ymax = y;
		}

		double xspan = xmax - xmin;
		double yspan = ymax - ymin;
		double span = MAX(xspan, yspan);
		double step = (span + 0.1) / 2000;
		int h = yspan / step;
		int w = xspan / step;
		cv::Mat img(h, w, CV_8UC3, cv::Scalar(255, 255, 255));

		for (size_t i = 0; i < cams_gps_.size(); i++)
		{
			double x = cams_[i]->pos_ac_.c(0);
			double y = cams_[i]->pos_ac_.c(2);
			int xx = (x - xmin) / step;
			int yy = (y - ymin) / step;
			cv::circle(img, cv::Point(xx, yy), 5, cv::Scalar(0, 0, 255));
			cv::putText(img, std::to_string(i), cv::Point(xx, yy), 1, 1, cv::Scalar(255, 0, 0));
		}
		cv::imwrite(path, img);
	}

}  // namespace objectsfm
