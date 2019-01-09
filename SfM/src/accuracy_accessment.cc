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

#include "accuracy_accessment.h"


namespace objectsfm {

	AccuracyAssessment::AccuracyAssessment()
	{
	}

	AccuracyAssessment::~AccuracyAssessment()
	{
	}

	void AccuracyAssessment::SetData(std::vector<CameraModel*>& cam_models, std::vector<Camera*>& cams, 
		std::vector<Point3D*>& pts)
	{
		cam_models_ = cam_models;
		cams_ = cams;
		pts_ = pts;
	}

	bool AccuracyAssessment::ErrorReprojectionPti(int i, double & e_avg, double & e_mse, int & n_obs)
	{
		Eigen::Vector4d pt_w_homo(pts_[i]->data[0], pts_[i]->data[1], pts_[i]->data[2], 1.0);

		std::map<int, Camera*>::iterator iter_cams = pts_[i]->cams_.begin();
		std::map<int, Eigen::Vector2d>::iterator iter_pts = pts_[i]->pts2d_.begin();
		std::vector<double> errors;
		while (iter_cams != pts_[i]->cams_.end())
		{
			Eigen::Vector3d pt_c = iter_cams->second->M * pt_w_homo; // convert into the camera coordinate system
			if (pt_c(2) > 0)
			{
				double x = pt_c(0) / pt_c(2);
				double y = pt_c(1) / pt_c(2);
				double r2 = x * x + y * y;
				double distortion = 1.0 + r2 * (iter_cams->second->cam_model_->k1_ + iter_cams->second->cam_model_->k2_  * r2);
				double u = iter_cams->second->cam_model_->f_ * distortion * x + iter_cams->second->cam_model_->dcx_; // the coordinate in pixel on the image
				double v = iter_cams->second->cam_model_->f_ * distortion * y + iter_cams->second->cam_model_->dcy_;
				double e = std::pow(u - iter_pts->second(0), 2) + std::pow(v - iter_pts->second(1), 2);
				errors.push_back(e);
			}
			iter_cams++;
			iter_pts++;
		}

		// 
		if (errors.size() <= 1) {
			return false;
		}

		e_avg = 0.0;
		e_mse = 0.0;
		for (size_t i = 0; i < errors.size(); i++) {
			e_avg += errors[i];
		}
		e_avg /= errors.size();

		for (size_t i = 0; i < errors.size(); i++) {
			e_mse += (errors[i] - e_avg)*(errors[i] - e_avg);
		}
		e_mse = std::sqrt(e_mse / (errors.size() - 1));

		n_obs = errors.size();

		return true;
	}

	void AccuracyAssessment::ErrorReprojectionPts(double & e_avg, double & e_mse, int & n_obs, std::vector<double> &errors)
	{
		errors.resize(pts_.size());

		e_avg = 0.0, e_mse = 0.0;
		n_obs = 0;
		int n_inliers = 0;
		for (size_t i = 0; i < pts_.size(); i++)
		{
			errors[i] = 1000.0;
			if (pts_[i]->is_bad_estimated_) {
				continue;
			}

			double e_avg_i = 0.0, e_mse_i = 0.0;
			int n_obs_i = 0;
			if (ErrorReprojectionPti(i, e_avg_i, e_mse_i, n_obs_i))
			{
				errors[i] = e_avg_i;
				e_avg += e_avg_i;
				e_mse += e_mse_i;
				n_obs += n_obs_i;
				n_inliers++;
			}
		}
		e_avg /= n_inliers;
		e_mse /= n_inliers;
		n_obs /= n_inliers;
	}


}  // namespace objectsfm
