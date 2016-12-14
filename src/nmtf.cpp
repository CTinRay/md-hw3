#include "nmtf.hpp"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>
#include <utility>

void nmtf2(double rate, double conv,
          int d1, int d2,
          const Eigen::MatrixXd x,
          const Eigen::MatrixXd mask,
          Eigen::MatrixXd&f,
          Eigen::MatrixXd&s,
          Eigen::MatrixXd&g) {
    double d = std::numeric_limits<double>::infinity();
    f.setRandom(x.rows(), d1);
    f.array() += 1;
    g.setRandom(x.cols(), d2);
    g.array() += 1;
    s.setRandom(d1, d2);

    std::cout << "f:" << f.row(0) << std::endl;
    std::cout << "g:" << g.row(0) << std::endl;

    std::vector<std::pair<int, int>>ones;
    for (auto i = 0u; i < mask.rows(); ++i) {
        for (auto j = 0u; j < mask.cols(); ++j) {
            if (mask(i, j) == 1) {
                ones.push_back(std::make_pair(i, j));
            }
        }
    }
    
    std::cout << "NMTF..." << std::endl;
    while (d > conv * conv) {
        d = 0;
        
        for (auto&p : ones) {
            int i = p.first;
            int j = p.second;
            double e = x(i, j) - f.row(i) * s * g.row(j).transpose();
            auto dfi = - e * s * g.row(j).transpose();
            auto dgi = - e * f.row(i) * s;
            auto ds  = - e * f.row(i).transpose() * g.row(j);
            d += dfi.norm() + dgi.norm() + ds.norm();
            f.row(i).array() -= dfi.array() * rate;
            g.row(j).array() -= dgi.array() * rate;
            s -= ds * rate;        
        }

        std::cout // << "s:" << s << std::endl
            << "d:" << d << std::endl
            << "f:" << f.row(0) << std::endl
            << "g:" << g.row(0) << std::endl
            << "loss: " << (x - f * s * g.transpose()).norm() << std::endl;
        
    }
}


void nmtf(double rate, double conv,
           const Eigen::MatrixXd x_orig,
           const Eigen::MatrixXd mask,
           Eigen::MatrixXd&f,
           Eigen::MatrixXd&s,
           Eigen::MatrixXd&g) {
    Eigen::MatrixXd x = x_orig;

    // double sum = 0;
    // int cnt = 0;
    // for (auto i = 0u; i < mask.rows(); ++i) {
    //     for (auto j = 0u; j < mask.cols(); ++j) {
    //         if (mask(i, j) == 1) {
    //             sum += x(i, j);
    //             cnt += 1;
    //         }
    //     }
    // }
    // double avg = sum / cnt;
    // for (auto i = 0u; i < mask.rows(); ++i) {
    //     for (auto j = 0u; j < mask.cols(); ++j) {
    //         if (mask(i, j) == 0) {
    //             x(i, j) = avg;                
    //         }
    //     }
    // }
    
    
    // f.setRandom(x.rows(), d1);
    // f.array() /= 10;
    // f.array() += 1;
    // g.setRandom(x.cols(), d2);
    // g.array() /= 10;
    // g.array() += 1;
    f.array() += 0.2;
    g.array() += 0.2;
    // s.setRandom(f.cols(), g.cols());
    s = f.transpose() * x * g;
    std::cout << "f:" << f.row(0) << std::endl;
    std::cout << "g:" << g.row(0) << std::endl;

    std::cout << "NMTF..." << std::endl;

    double d = std::numeric_limits<double>::infinity();
    while (d > conv * conv) {
        
        auto gprev = g;
        auto fprev = f;
        auto sprev = s;

        std::cout << "loss: " << (x - f * s * g.transpose()).squaredNorm() << std::endl;
        g.array() = g.array() * ((x.transpose() * f * s).array() /
                                 (g * (g.transpose() * x.transpose()) * (f * s)).array()).sqrt();

        std::cout << "loss: " << (x - f * s * g.transpose()).squaredNorm() << std::endl;        
        f.array() = f.array() * ((x * g * s.transpose()).array() /
                                 ((f * (f.transpose() * x)) * (g * s.transpose()) ).array()).sqrt();

        std::cout << "loss: " << (x - f * s * g.transpose()).squaredNorm() << std::endl;
        s.array() = s.array() * ((f.transpose() * x * g).array() /
                                 (((f.transpose() * f) * s) * (g.transpose() * g)).array()).sqrt();
        
        
        d = (fprev - f).squaredNorm() + (gprev - g).squaredNorm() + (sprev - s).squaredNorm();
        std::cout // << "s:" << s << std::endl
            << "d:" << d << std::endl
            << "f:" << f.row(0) << std::endl
            << "g:" << g.row(0) << std::endl
            << "loss: " << (x - f * s * g.transpose()).squaredNorm() << std::endl;        
    }

    for (auto i = 0u; i < f.rows(); ++i) {
        int x, y;
        f.row(i).maxCoeff(&x, &y);
        f.row(i).setZero();
        f(i, y) = 1;
    }

    for (auto i = 0u; i < g.rows(); ++i) {
        int x, y;
        g.row(i).maxCoeff(&x, &y);
        g.row(i).setZero();
        g(i, y) = 1;
    }

}
