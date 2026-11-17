#ifndef ROBOT_ROBOTS_OLAF_INEKF_HPP_
#define ROBOT_ROBOTS_OLAF_INEKF_HPP_

// Contact-aided invariant EKF (Hartley, Ghaffari, Eustice, Grizzle, IJRR 2020), right-invariant form of the open-source
// invariant-ekf library: the Olaf paper's state estimator [42]. C++ twin of arti_h1_olaf olaf/inekf.py (InEKF), which
// runs in the training loop for the actor observation: same equations, same order, double precision.
//
//   state X in SE_4(3): R (body -> world), v, p (body origin = IMU), d_L, d_R (sole contact points, world)
//   parameters: gyro bias b_g, accelerometer bias b_a;  error order (21): [phi_R, v, p, d_L, d_R, b_g, b_a]
//   propagate(gyro, accel, dt): strapdown IMU integration, covariance with the log-linear error dynamics
//   correct(s, contact): forward kinematics s_i = R^T (d_i - p) of each sole in contact; a sole that touches down is
//   (re)initialised at p + R s_i with the covariance of p; a sole in the air stays in the state unused.

#include <array>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Dense>

namespace robot {

class OlafInEKF {
public:
    static constexpr int DIM = 21;
    static constexpr int IR = 0, IV = 3, IP = 6, ID = 9, IBG = 15, IBA = 18;
    using Mat = Eigen::Matrix<double, DIM, DIM>;
    using Vec = Eigen::Matrix<double, DIM, 1>;

    struct Params {
        double gyro_noise{0.02}, accel_noise{2.0}, gyro_bias_noise{0.002}, accel_bias_noise{0.02};
        double contact_noise{0.05}, fk_noise{0.004};
        double init_rot{0.03}, init_vel{0.05}, init_pos{0.01}, init_gyro_bias{0.03}, init_accel_bias{0.2};
    };

    void configure(const Params& p) {
        Vec q = Vec::Zero();
        q.segment<3>(IR).setConstant(p.gyro_noise * p.gyro_noise);
        q.segment<3>(IV).setConstant(p.accel_noise * p.accel_noise);
        q.segment<6>(ID).setConstant(p.contact_noise * p.contact_noise);
        q.segment<3>(IBG).setConstant(p.gyro_bias_noise * p.gyro_bias_noise);
        q.segment<3>(IBA).setConstant(p.accel_bias_noise * p.accel_bias_noise);
        Qc_ = q.asDiagonal();
        fk_var_ = p.fk_noise * p.fk_noise;
        Vec p0 = Vec::Zero();
        p0.segment<3>(IR).setConstant(p.init_rot * p.init_rot);
        p0.segment<3>(IV).setConstant(p.init_vel * p.init_vel);
        p0.segment<3>(IP).setConstant(p.init_pos * p.init_pos);
        p0.segment<3>(IBG).setConstant(p.init_gyro_bias * p.init_gyro_bias);
        p0.segment<3>(IBA).setConstant(p.init_accel_bias * p.init_accel_bias);
        P0_ = p0.asDiagonal();
    }

    // Start at attitude R0, zero velocity / biases, the body origin above the soles s (body frame) with their mean
    // height at ground_z, x / y = 0, both soles in contact.
    void reset(const Eigen::Matrix3d& R0, const std::array<Eigen::Vector3d, 2>& s, double ground_z = 0.0) {
        R_ = R0;
        v_.setZero();
        p_.setZero();
        p_.z() = ground_z - 0.5 * ((R0 * s[0]).z() + (R0 * s[1]).z());
        bg_.setZero();
        ba_.setZero();
        P_ = P0_;
        augment(0, s[0]);
        augment(1, s[1]);
        contact_ = {true, true};
    }

    void propagate(const Eigen::Vector3d& gyro, const Eigen::Vector3d& accel, double dt) {
        const Eigen::Vector3d w = gyro - bg_;
        const Eigen::Vector3d a = accel - ba_;
        const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
        Mat A = Mat::Zero();
        A.block<3, 3>(IV, IR) = skew(g_);
        A.block<3, 3>(IP, IV) = I3;
        A.block<3, 3>(IR, IBG) = -R_;
        A.block<3, 3>(IV, IBG) = -skew(v_) * R_;
        A.block<3, 3>(IP, IBG) = -skew(p_) * R_;
        for (int i = 0; i < 2; ++i) A.block<3, 3>(ID + 3 * i, IBG) = -skew(d_[i]) * R_;
        A.block<3, 3>(IV, IBA) = -R_;
        const Mat Adt = A * dt;
        const Mat Phi = Mat::Identity() + Adt + 0.5 * Adt * Adt;
        Mat Ad = Mat::Zero();
        Ad.block<3, 3>(IR, IR) = R_;
        Ad.block<3, 3>(IV, IR) = skew(v_) * R_;
        Ad.block<3, 3>(IV, IV) = R_;
        Ad.block<3, 3>(IP, IR) = skew(p_) * R_;
        Ad.block<3, 3>(IP, IP) = R_;
        for (int i = 0; i < 2; ++i) {
            const int k = ID + 3 * i;
            Ad.block<3, 3>(k, IR) = skew(d_[i]) * R_;
            Ad.block<3, 3>(k, k) = R_;
        }
        Ad.block<6, 6>(IBG, IBG).setIdentity();
        const Mat Q = Phi * Ad * Qc_ * Ad.transpose() * Phi.transpose() * dt;
        const Mat P = Phi * P_ * Phi.transpose() + Q;
        P_ = 0.5 * (P + P.transpose());
        const Eigen::Vector3d acc_w = R_ * a + g_;
        p_ = p_ + v_ * dt + 0.5 * acc_w * dt * dt;
        v_ = v_ + acc_w * dt;
        R_ = R_ * exp_so3(w * dt);
    }

    void correct(const std::array<Eigen::Vector3d, 2>& s, const std::array<bool, 2>& contact) {
        std::array<bool, 2> fresh{};
        for (int i = 0; i < 2; ++i) {
            fresh[i] = contact[i] && !contact_[i];
            if (fresh[i]) augment(i, s[i]);
        }
        for (int i = 0; i < 2; ++i) {
            if (!contact[i] || fresh[i]) continue;
            const int k = ID + 3 * i;
            Eigen::Matrix<double, 3, DIM> H = Eigen::Matrix<double, 3, DIM>::Zero();
            H.block<3, 3>(0, IP) = -Eigen::Matrix3d::Identity();
            H.block<3, 3>(0, k) = Eigen::Matrix3d::Identity();
            const Eigen::Vector3d Z = R_ * s[i] + p_ - d_[i];
            const Eigen::Matrix<double, DIM, 3> PHt = P_ * H.transpose();
            const Eigen::Matrix3d S = H * PHt + fk_var_ * Eigen::Matrix3d::Identity();
            const Eigen::Matrix<double, DIM, 3> K = PHt * S.inverse();
            apply(K * Z);
            const Mat IKH = Mat::Identity() - K * H;
            const Mat P = IKH * P_ * IKH.transpose() + fk_var_ * K * K.transpose();
            P_ = 0.5 * (P + P.transpose());
        }
        contact_ = contact;
    }

    const Eigen::Matrix3d& R() const { return R_; }
    const Eigen::Vector3d& v() const { return v_; }
    const Eigen::Vector3d& p() const { return p_; }
    const Eigen::Vector3d& gyro_bias() const { return bg_; }
    const Eigen::Vector3d& accel_bias() const { return ba_; }
    const std::array<bool, 2>& contact() const { return contact_; }

private:
    static Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
        Eigen::Matrix3d m;
        m << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
        return m;
    }

    // Exp(phi) and the left Jacobian (small-angle series below 1e-6 rad), as olaf/inekf.py so3_exp_jl
    static void exp_jl(const Eigen::Vector3d& phi, Eigen::Matrix3d& E, Eigen::Matrix3d& J) {
        const double th = phi.norm();
        const Eigen::Matrix3d K = skew(phi), K2 = K * K;
        double a, b, c;
        if (th < 1e-6) {
            a = 1.0 - th * th / 6.0;
            b = 0.5 - th * th / 24.0;
            c = 1.0 / 6.0 - th * th / 120.0;
        } else {
            a = std::sin(th) / th;
            b = (1.0 - std::cos(th)) / (th * th);
            c = (th - std::sin(th)) / (th * th * th);
        }
        E = Eigen::Matrix3d::Identity() + a * K + b * K2;
        J = Eigen::Matrix3d::Identity() + b * K + c * K2;
    }

    static Eigen::Matrix3d exp_so3(const Eigen::Vector3d& phi) {
        Eigen::Matrix3d E, J;
        exp_jl(phi, E, J);
        return E;
    }

    void augment(int i, const Eigen::Vector3d& s) {
        d_[i] = p_ + R_ * s;
        Mat F = Mat::Identity();
        const int k = ID + 3 * i;
        F.block<3, DIM>(k, 0).setZero();
        F.block<3, 3>(k, IP).setIdentity();
        Mat P = F * P_ * F.transpose();
        P.block<3, 3>(k, k) += fk_var_ * Eigen::Matrix3d::Identity();
        P_ = P;
    }

    void apply(const Vec& delta) {
        Eigen::Matrix3d E, J;
        exp_jl(delta.segment<3>(IR), E, J);
        R_ = E * R_;
        v_ = E * v_ + J * delta.segment<3>(IV);
        p_ = E * p_ + J * delta.segment<3>(IP);
        for (int i = 0; i < 2; ++i) d_[i] = E * d_[i] + J * delta.segment<3>(ID + 3 * i);
        bg_ += delta.segment<3>(IBG);
        ba_ += delta.segment<3>(IBA);
    }

    Eigen::Matrix3d R_{Eigen::Matrix3d::Identity()};
    Eigen::Vector3d v_{Eigen::Vector3d::Zero()}, p_{Eigen::Vector3d::Zero()};
    std::array<Eigen::Vector3d, 2> d_{{Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()}};
    Eigen::Vector3d bg_{Eigen::Vector3d::Zero()}, ba_{Eigen::Vector3d::Zero()};
    Mat P_{Mat::Identity()}, P0_{Mat::Identity()}, Qc_{Mat::Zero()};
    double fk_var_{1e-4};
    std::array<bool, 2> contact_{{false, false}};
    const Eigen::Vector3d g_{0.0, 0.0, -9.81};
};

}  // namespace robot

#endif  // ROBOT_ROBOTS_OLAF_INEKF_HPP_
