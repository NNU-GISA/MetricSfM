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

#include "orientation/absolute_pose_via_p3p.h"

#include <iostream>
#include <Eigen/Dense>
#include <complex>
#include <algorithm>

#include "utils/polynomial.h"
#include "utils/basic_funcs.h"

#ifndef MIN_
#define MIN_(a,b) ( ((a)<(b)) ? (a):(b) )
#endif // !MIN

namespace objectsfm {

	bool AbsolutePoseP3P::P3PRANSAC( std::vector<Eigen::Vector3d> &pts_w, std::vector<Eigen::Vector2d> &pts_2d, double f, RTPose &pose )
	{
		// 
		int num = pts_w.size();

		std::vector<RTPose> pose_hyps;
		if (num<3)
		{
			return false;
		}
		else 
		{
			int iter_max = 0;
			if (num == 3)
			{
				iter_max = 1;
			}
			else
			{
				iter_max = 200;
			}

			RTPose pose_best;
			double error_best = 1000000000.0;
			int iter = 0;

			while (iter <= iter_max)
			{
				int N = 3;
				std::vector<int> random_index;
				math::RandVectorN(0, num, N, random_index);

				std::vector<Eigen::Vector3d> pts_w_partial(N);
				std::vector<Eigen::Vector2d> pts_2d_partial(N);
				for (int i = 0; i < N; ++i)
				{
					int id = random_index[i];
					pts_w_partial[i] = pts_w[id];
					pts_2d_partial[i] = pts_2d[id];
				}

				std::vector<RTPose> pose_hyps;
				if (!P3P(pts_w_partial, pts_2d_partial, f, pose_hyps))
				{
					continue;
				}

				for (size_t i = 0; i < pose_hyps.size(); i++)
				{
					double error_i = Error(pts_w, pts_2d, f, pose_hyps[i]);
					if (error_i < error_best)
					{
						error_best = error_i;
						pose_best = pose_hyps[i];
					}
				}
				iter++;
			}

			if (std::sqrt(error_best / num - 1) < 2.0)
			{
				pose = pose_best;
				return true;
			}
			else
			{
				return false;
			}
		}
	}

	bool AbsolutePoseP3P::P3P( std::vector<Eigen::Vector3d> &pts_w, std::vector<Eigen::Vector2d> &pts_2d, double f, std::vector<RTPose> &pose_hyps )
	{	
		Eigen::Vector3d normalized_image_points[3];
		// Store points_3d in world_points for ease of use. NOTE: we cannot use a
		// const ref or a Map because the world_points entries may be swapped later.
		Eigen::Vector3d world_points[3];
		for (int i = 0; i < 3; ++i) 
		{
			normalized_image_points[i] = (pts_2d[i]/f).homogeneous().normalized();
			world_points[i] = pts_w[i];
		}

		// If the points are collinear, there are no possible solutions.
		double kTolerance = 1e-6;
		Eigen::Vector3d world_1_0 = world_points[1] - world_points[0];
		Eigen::Vector3d world_2_0 = world_points[2] - world_points[0];
		if (world_1_0.cross(world_2_0).squaredNorm() < kTolerance) 
		{
			std::cout << "The 3 world points are collinear! No solution for absolute pose exits."<<std::endl;
			return false;
		}

		// Create intermediate camera frame such that the x axis is in the direction
		// of one of the normalized image points, and the origin is the same as the
		// absolute camera frame. This is a rotation defined as the transformation:
		// T = [tx, ty, tz] where tx = f0, tz = (f0 x f1) / ||f0 x f1||, and
		// ty = tx x tz and f0, f1, f2 are the normalized image points.
		Eigen::Matrix3d intermediate_camera_frame;
		intermediate_camera_frame.row(0) = normalized_image_points[0];
		intermediate_camera_frame.row(2) = normalized_image_points[0].cross(normalized_image_points[1]).normalized();
		intermediate_camera_frame.row(1) = intermediate_camera_frame.row(2).cross(intermediate_camera_frame.row(0));

		// Project the third world point into the intermediate camera frame.
		Eigen::Vector3d intermediate_image_point = intermediate_camera_frame * normalized_image_points[2];

		// Enforce that the intermediate_image_point is in front of the intermediate
		// camera frame. If the point is behind the camera frame, recalculate the
		// intermediate camera frame by swapping which feature we align the x axis to.
		if (intermediate_image_point[2] > 0) 
		{
			std::swap(normalized_image_points[0], normalized_image_points[1]);

			intermediate_camera_frame.row(0) = normalized_image_points[0];
			intermediate_camera_frame.row(2) = normalized_image_points[0]
			.cross(normalized_image_points[1]).normalized();
			intermediate_camera_frame.row(1) = intermediate_camera_frame.row(2).cross(intermediate_camera_frame.row(0));

			intermediate_image_point = intermediate_camera_frame * normalized_image_points[2];

			std::swap(world_points[0], world_points[1]);
			world_1_0 = world_points[1] - world_points[0];
			world_2_0 = world_points[2] - world_points[0];
		}

		// Create the intermediate world frame transformation that has the
		// origin at world_points[0] and the x-axis in the direction of
		// world_points[1]. This is defined by the transformation: N = [nx, ny, nz]
		// where nx = (p1 - p0) / ||p1 - p0||
		// nz = nx x (p2 - p0) / || nx x (p2 -p0) || and ny = nz x nx
		// Where p0, p1, p2 are the world points.
		Eigen::Matrix3d intermediate_world_frame;
		intermediate_world_frame.row(0) = world_1_0.normalized();
		intermediate_world_frame.row(2) = intermediate_world_frame.row(0).cross(world_2_0).normalized();
		intermediate_world_frame.row(1) = intermediate_world_frame.row(2).cross(intermediate_world_frame.row(0));

		// Transform world_point[2] to the intermediate world frame coordinates.
		Eigen::Vector3d intermediate_world_point = intermediate_world_frame * world_2_0;

		// Distance from world_points[1] to the intermediate world frame origin.
		double d_12 = world_1_0.norm();

		// Solve for the cos(theta) that will give us the transformation from
		// intermediate world frame to intermediate camera frame. We also get the
		// cot(alpha) for each solution necessary for back-substitution.
		double cos_theta[4];
		double cot_alphas[4];
		double b;
		const int num_solutions = SolvePlaneRotation( normalized_image_points, intermediate_image_point,
			intermediate_world_point, d_12, cos_theta, cot_alphas, &b);
		if (!num_solutions)
		{
			return false;
		}

		// Backsubstitution of each solution
		for (int i = 0; i < num_solutions; i++) 
		{
			RTPose pose_cur;
			Backsubstitute(intermediate_world_frame,
				intermediate_camera_frame,
				world_points[0],
				cos_theta[i],
				cot_alphas[i],
				d_12,
				b,
				&pose_cur.t,
				&pose_cur.R);

			pose_hyps.push_back(pose_cur);
		}

		return true;
	}

	int AbsolutePoseP3P::SolvePlaneRotation(const Eigen::Vector3d normalized_image_points[3], const Eigen::Vector3d & intermediate_image_point, const Eigen::Vector3d & intermediate_world_point, const double d_12, double cos_theta[4], double cot_alphas[4], double * b)
	{
		// Calculate these parameters ahead of time for reuse and
		// readability. Notation for these variables is consistent with the notation
		// from the paper.
		const double f_1 = intermediate_image_point[0] / intermediate_image_point[2];
		const double f_2 = intermediate_image_point[1] / intermediate_image_point[2];
		const double p_1 = intermediate_world_point[0];
		const double p_2 = intermediate_world_point[1];
		const double cos_beta = normalized_image_points[0].dot(normalized_image_points[1]);
		*b = 1.0 / (1.0 - cos_beta * cos_beta) - 1.0;

		if (cos_beta < 0) 
		{
			*b = -sqrt(*b);
		}
		else 
		{
			*b = sqrt(*b);
		}

		// Definition of temporary variables for readability in the coefficients
		// calculation.
		const double f_1_pw2 = f_1 * f_1;
		const double f_2_pw2 = f_2 * f_2;
		const double p_1_pw2 = p_1 * p_1;
		const double p_1_pw3 = p_1_pw2 * p_1;
		const double p_1_pw4 = p_1_pw3 * p_1;
		const double p_2_pw2 = p_2 * p_2;
		const double p_2_pw3 = p_2_pw2 * p_2;
		const double p_2_pw4 = p_2_pw3 * p_2;
		const double d_12_pw2 = d_12 * d_12;
		const double b_pw2 = (*b) * (*b);

		// Computation of coefficients of 4th degree polynomial.
		Eigen::VectorXd coefficients(5);
		coefficients(0) = -f_2_pw2 * p_2_pw4 - p_2_pw4 * f_1_pw2 - p_2_pw4;
		coefficients(1) =
			2.0 * p_2_pw3 * d_12 * (*b) + 2.0 * f_2_pw2 * p_2_pw3 * d_12 * (*b) -
			2.0 * f_2 * p_2_pw3 * f_1 * d_12;
		coefficients(2) =
			-f_2_pw2 * p_2_pw2 * p_1_pw2 - f_2_pw2 * p_2_pw2 * d_12_pw2 * b_pw2 -
			f_2_pw2 * p_2_pw2 * d_12_pw2 + f_2_pw2 * p_2_pw4 + p_2_pw4 * f_1_pw2 +
			2.0 * p_1 * p_2_pw2 * d_12 +
			2.0 * f_1 * f_2 * p_1 * p_2_pw2 * d_12 * (*b) -
			p_2_pw2 * p_1_pw2 * f_1_pw2 + 2.0 * p_1 * p_2_pw2 * f_2_pw2 * d_12 -
			p_2_pw2 * d_12_pw2 * b_pw2 - 2.0 * p_1_pw2 * p_2_pw2;
		coefficients(3) =
			2.0 * p_1_pw2 * p_2 * d_12 * (*b) + 2.0 * f_2 * p_2_pw3 * f_1 * d_12 -
			2.0 * f_2_pw2 * p_2_pw3 * d_12 * (*b) - 2.0 * p_1 * p_2 * d_12_pw2 * (*b);
		coefficients(4) =
			-2 * f_2 * p_2_pw2 * f_1 * p_1 * d_12 * (*b) +
			f_2_pw2 * p_2_pw2 * d_12_pw2 + 2.0 * p_1_pw3 * d_12 - p_1_pw2 * d_12_pw2 +
			f_2_pw2 * p_2_pw2 * p_1_pw2 - p_1_pw4 -
			2.0 * f_2_pw2 * p_2_pw2 * p_1 * d_12 + p_2_pw2 * f_1_pw2 * p_1_pw2 +
			f_2_pw2 * p_2_pw2 * d_12_pw2 * b_pw2;

		// Computation of roots.
		Eigen::VectorXd roots;
		FindPolynomialRoots(coefficients, &roots, NULL);

		// Calculate cot(alpha) needed for back-substitution.
		for (int i = 0; i < roots.size(); i++) 
		{
			cos_theta[i] = roots(i);
			cot_alphas[i] = (-f_1 * p_1 / f_2 - cos_theta[i] * p_2 + d_12 * (*b)) /
				(-f_1 * cos_theta[i] * p_2 / f_2 + p_1 - d_12);
		}

		return static_cast<int>(roots.size());
	}

	void AbsolutePoseP3P::Backsubstitute(const Eigen::Matrix3d & intermediate_world_frame, const Eigen::Matrix3d & intermediate_camera_frame, const Eigen::Vector3d & world_point_0, const double cos_theta, const double cot_alpha, const double d_12, const double b, Eigen::Vector3d * translation, Eigen::Matrix3d * rotation)
	{

		const double sin_theta = sqrt(1.0 - cos_theta * cos_theta);
		const double sin_alpha = sqrt(1.0 / (cot_alpha * cot_alpha + 1.0));
		double cos_alpha = sqrt(1.0 - sin_alpha * sin_alpha);

		if (cot_alpha < 0) {
			cos_alpha = -cos_alpha;
		}

		// Get the camera position in the intermediate world frame
		// coordinates. (Eq. 5 from the paper).
		const Eigen::Vector3d c_nu(
			d_12 * cos_alpha * (sin_alpha * b + cos_alpha),
			cos_theta * d_12 * sin_alpha * (sin_alpha * b + cos_alpha),
			sin_theta * d_12 * sin_alpha * (sin_alpha * b + cos_alpha));

		// Transform c_nu into world coordinates. Use a Map to put the solution
		// directly into the output.
		*translation = world_point_0 + intermediate_world_frame.transpose() * c_nu;

		// Construct the transformation from the intermediate world frame to the
		// intermediate camera frame.
		Eigen::Matrix3d intermediate_world_to_camera_rotation;
		intermediate_world_to_camera_rotation <<
			-cos_alpha, -sin_alpha * cos_theta, -sin_alpha * sin_theta,
			sin_alpha, -cos_alpha * cos_theta, -cos_alpha * sin_theta,
			0, -sin_theta, cos_theta;

		// Construct the rotation matrix.
		*rotation = (intermediate_world_frame.transpose() *
			intermediate_world_to_camera_rotation.transpose() *
			intermediate_camera_frame).transpose();

		// Adjust translation to account for rotation.
		*translation = -(*rotation) * (*translation);
	}

	// The reprojected point is computed as Xc = R * Xw + t
	double AbsolutePoseP3P::Error(const std::vector<Eigen::Vector3d>& pts_w, const std::vector<Eigen::Vector2d>& pts_2d, double f, RTPose pose)
	{
		// calculate the projection matrix P
		Eigen::Matrix<double, 3, 4> transformation_matrix;
		transformation_matrix.block<3, 3>(0, 0) = pose.R;
		transformation_matrix.col(3) = pose.t;
		Eigen::Matrix3d camera_matrix = Eigen::DiagonalMatrix<double, 3>(f, f, 1.0);
		Eigen::Matrix<double, 3, 4> projection_matrices = camera_matrix * transformation_matrix;

		// The reprojected point is computed as Xc = P * Xw 
		int num = pts_w.size();
		double error_total = 0.0;
		for (size_t i = 0; i < num; i++)
		{
			Eigen::Vector4d pt_w(pts_w[i](0), pts_w[i](1), pts_w[i](2), 1.0);
			Eigen::Vector3d pt_c = projection_matrices * pt_w;
			const Eigen::Vector2d pt_c_2d(pt_c(0) / pt_c(2), pt_c(1) / pt_c(2));
			error_total += (pt_c_2d - pts_2d[i]).squaredNorm();
		}

		return error_total;
	}

}  // namespace objectsfm
