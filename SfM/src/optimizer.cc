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

#include "optimizer.h"

#include "camera.h"
#include "structure.h"
#include "utils/basic_funcs.h"
#include "utils/reprojection_error_pose_cam_xyz.h"
#include "utils/reprojection_error_pose_xyz.h"
#include "utils/reprojection_error_pose_cam.h"
#include "utils/reprojection_error_pose.h"
#include "utils/reprojection_error_xyz.h"

namespace objectsfm {


BundleAdjuster::BundleAdjuster(std::vector<Camera*> cams, std::vector<CameraModel*> cam_models, std::vector<Point3D*> pts)
{
	cams_ = cams;
	cam_models_ = cam_models;
	pts_ = pts;
}

BundleAdjuster::~BundleAdjuster()
{
}

void BundleAdjuster::SetOptions(BundleAdjustOptions options)
{
	options_.max_num_iterations = options.max_num_iterations;
	options_.minimizer_progress_to_stdout = options.minimizer_progress_to_stdout;
	options_.num_threads = options.num_threads;
	options_.linear_solver_type = ceres::DENSE_SCHUR;
}

void BundleAdjuster::RunOptimizetion(bool is_initial_run, double weight)
{
	// for initial run, normalize the distribution of these 3D points
	if (is_initial_run)
	{
		Normalize();
		Perturb();
	}

	// build the problem
	int num_pts = pts_.size();
	int count2 = 0, count3 = 0, count4 = 0;
	for (size_t i = 0; i < num_pts; i++)
	{
		if (pts_[i]->is_bad_estimated_)
		{
			continue;
		}

		if (pts_[i]->cams_.size() == 2) 
		{ 
			pts_[i]->weight = 1.0; 
			count2++;
		}
		if (pts_[i]->cams_.size() >= 3)
		{
			pts_[i]->weight = weight;
			count3++;
		}

		std::map<int, Camera*>::iterator iter_cams = pts_[i]->cams_.begin();
		std::map<int, Eigen::Vector2d>::iterator iter_pts = pts_[i]->pts2d_.begin();
		while (iter_cams != pts_[i]->cams_.end())
		{
			ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
			ceres::CostFunction* cost_function;
			if (pts_[i]->is_mutable_ && iter_cams->second->is_mutable_)
			{
				if (iter_cams->second->cam_model_->is_mutable_)
				{
					cost_function = ReprojectionErrorPoseCamXYZ::Create(iter_pts->second(0), iter_pts->second(1), pts_[i]->weight);
					problem_.AddResidualBlock(cost_function, loss_function,
						iter_cams->second->data, iter_cams->second->cam_model_->data, pts_[i]->data);
				}
				else
				{
					cost_function = ReprojectionErrorPoseXYZ::Create(iter_pts->second(0), iter_pts->second(1), 
						iter_cams->second->cam_model_->data, pts_[i]->weight);
					problem_.AddResidualBlock(cost_function, loss_function, iter_cams->second->data, pts_[i]->data);
				}
				
			}
			else if (pts_[i]->is_mutable_ && !iter_cams->second->is_mutable_)
			{
				cost_function = ReprojectionErrorXYZ::Create(iter_pts->second(0), iter_pts->second(1),
					iter_cams->second->data, iter_cams->second->cam_model_->data, pts_[i]->weight);

				problem_.AddResidualBlock(cost_function, loss_function, pts_[i]->data);
			}
			else if (!pts_[i]->is_mutable_ && iter_cams->second->is_mutable_)
			{
				if (iter_cams->second->cam_model_->is_mutable_)
				{
					cost_function = ReprojectionErrorPoseCam::Create(iter_pts->second(0), iter_pts->second(1), pts_[i]->data, pts_[i]->weight);

					problem_.AddResidualBlock(cost_function, loss_function,
						iter_cams->second->data, iter_cams->second->cam_model_->data);
				}
				else
				{
					cost_function = ReprojectionErrorPose::Create(iter_pts->second(0), iter_pts->second(1), 
						pts_[i]->data, iter_cams->second->cam_model_->data, pts_[i]->weight);

					problem_.AddResidualBlock(cost_function, loss_function, iter_cams->second->data);
				}
			}
			iter_cams++;
			iter_pts++;
		}
	}
	//std::cout << "--------------count2 " << count2 << " count3 " << count3 << " count4 " << count4 << std::endl;

	// solve the problem
	ceres::Solve(options_, &problem_, &summary_);
	//std::cout << summary_.FullReport() << "\n";
}

void BundleAdjuster::FindOutliersPoints()
{

}

void BundleAdjuster::UpdateParameters()
{
	for (size_t i = 0; i < cams_.size(); i++)
	{
		cams_[i]->UpdatePoseFromData();
	}

	for (size_t i = 0; i < cam_models_.size(); i++)
	{
		cam_models_[i]->UpdataModelFromData();
	}
}

void BundleAdjuster::Normalize()
{
	int num_pts = pts_.size();

	// Compute the marginal median of the geometry
	std::vector<double> mid_pt(3,0);
	for (int i = 0; i < num_pts; ++i)
	{
		mid_pt[0] += pts_[i]->data[0];
		mid_pt[1] += pts_[i]->data[1];
		mid_pt[2] += pts_[i]->data[2];
	}
	mid_pt[0] /= num_pts;
	mid_pt[1] /= num_pts;
	mid_pt[2] /= num_pts;

	double median_absolute_deviation = 0;
	for (int i = 0; i < num_pts; ++i)
	{
		double L1dis = abs(pts_[i]->data[0] - mid_pt[0])
			+ abs(pts_[i]->data[1] - mid_pt[1])
			+ abs(pts_[i]->data[2] - mid_pt[2]);
		median_absolute_deviation += L1dis;
	}
	median_absolute_deviation /= num_pts;

	// Scale so that the median absolute deviation of the resulting reconstruction is 100.
	double scale = 100.0 / median_absolute_deviation;
	for (int i = 0; i < num_pts; ++i)
	{
		pts_[i]->data[0] = scale * (pts_[i]->data[0] - mid_pt[0]);
		pts_[i]->data[1] = scale * (pts_[i]->data[1] - mid_pt[1]);
		pts_[i]->data[2] = scale * (pts_[i]->data[2] - mid_pt[2]);
	}

	Eigen::Vector3d mid_pt_eigen(mid_pt[0], mid_pt[1], mid_pt[2]);
	for (int i = 0; i < cams_.size(); ++i)
	{
		cams_[i]->SetACPose(cams_[i]->pos_ac_.a, scale*(cams_[i]->pos_ac_.c - mid_pt_eigen));
	}
}

void BundleAdjuster::Perturb()
{
	double rotation_sigma = 0.1;
	double translation_sigma = 0.5;
	double point_sigma = 0.5;

	// perturb the 3d points
	for (size_t i = 0; i < pts_.size(); i++)
	{
		pts_[i]->data[0] += math::randnormal<double>()*point_sigma;
		pts_[i]->data[1] += math::randnormal<double>()*point_sigma;
		pts_[i]->data[2] += math::randnormal<double>()*point_sigma;
	}

	// perturb the camera
	for (size_t i = 0; i < cams_.size(); i++)
	{
		if (rotation_sigma > 0.0)
		{
			Eigen::Vector3d noise;
			noise[0] = math::randnormal<double>()*rotation_sigma;
			noise[1] = math::randnormal<double>()*rotation_sigma;
			noise[2] = math::randnormal<double>()*rotation_sigma;
			cams_[i]->SetACPose(cams_[i]->pos_ac_.a+noise, cams_[i]->pos_ac_.c);
		}

		if (translation_sigma > 0.0)
		{
			Eigen::Vector3d noise;
			noise[0] = math::randnormal<double>()*translation_sigma;
			noise[1] = math::randnormal<double>()*translation_sigma;
			noise[2] = math::randnormal<double>()*translation_sigma;
			cams_[i]->SetRTPose(cams_[i]->pos_rt_.R, cams_[i]->pos_rt_.t+noise);
		}
	}
}


}  // namespace objectsfm
