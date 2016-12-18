#include "utils.hpp"
#include "nmtf.hpp"
#include <boost/program_options.hpp>
#include <iostream>
#include <random>
#include <vector>
#include <utility>
#include <cmath>
#include <fstream>

#define SQUARE(X) ((X) * (X))

struct Arguments {
    std::string source, target, sourceModel, cu, ci, all;
    double nmtfConv, nmtfRate, holdout;
    int nClustersU, nClustersI, rD1, rD2, nIters, sD1, sD2;
};


Arguments getArgs(int argc, char**argv) {
    Arguments args;
    try {
        namespace po = boost::program_options;

        po::positional_options_description positional;
        positional.add("source", 1);
        positional.add("all", 1);
        positional.add("cu", 1);
        positional.add("ci", 1);
        positional.add("target", 1);

        po::options_description desc("===== Codebook =====");
        desc.add_options()
            ("source", po::value<std::string>(&args.source) -> required(), "source.txt")
            ("all", po::value<std::string>(&args.all) -> required(), "all rating in source")
            ("cu", po::value<std::string>(&args.cu) -> default_value(""), "user cluster file cu.txt")
            ("ci", po::value<std::string>(&args.ci) -> default_value(""), "item cluster file ci.txt")
            ("target", po::value<std::string>(&args.target) -> required(), "test.txt")
            ("help", "Print help message.")
            ("nmtfRate", po::value<double>(&args.nmtfRate) -> default_value(0.0001), "learning rate for nmtf")
            ("holdout", po::value<double>(&args.holdout) -> default_value(0.1), "learning rate for nmtf")
            ("converge", po::value<double>(&args.nmtfConv) -> default_value(10), "converge criteria for nmtf")
            ("sD1", po::value<int>(&args.sD1) -> default_value(50000), "dimention of rating matrix")             
            ("sD2", po::value<int>(&args.sD2) -> default_value(5000), "dimention of rating matrix")             
            ("rD1", po::value<int>(&args.rD1) -> default_value(50000), "dimention of rating matrix")             
            ("rD2", po::value<int>(&args.rD2) -> default_value(5000), "dimention of rating matrix")             
            ("nClustersU", po::value<int>(&args.nClustersU) -> default_value(10), "number of user clusters")
            ("nClustersI", po::value<int>(&args.nClustersI) -> default_value(10), "number of item clusters")
            ("nIters", po::value<int>(&args.nIters) -> default_value(10), "number of iteration when construction");

        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv)
                  .options(desc)
                  .positional(positional)
                  .run(), vm);

        if (vm.count("help")) {
            std::cout << desc << std::endl;
            exit(0);
        }

        po::notify(vm);
    } catch (boost::program_options::error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        exit(-1);
    }

    return args;
}


void transferCodebook(int nIters,
                      const Eigen::MatrixXd& x,
                      const Eigen::MatrixXd& mask,
                      const Eigen::MatrixXd& codebook,
                      Eigen::MatrixXd& u,
                      Eigen::MatrixXd& v) {
    u = Eigen::MatrixXd::Zero(x.rows(), codebook.rows());
    v = Eigen::MatrixXd::Zero(x.cols(), codebook.cols());

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, v.cols() - 1);

    // randomly initialize v
    for (auto i = 0u; i < v.rows(); ++i) {
        auto j = dis(gen);
        v(i, j) = 1;
    }

    for (int t = 0; t < nIters; ++t) {
        Eigen::MatrixXd bvt = codebook * v.transpose();

#pragma omp parallel for
        for (int i = 0; i < x.rows(); ++i) {
            int j, k;
            Eigen::MatrixXd tmp = (-bvt).rowwise() + x.row(i);
            tmp.array().rowwise() *= mask.row(i).array();
            auto norms = tmp.rowwise().squaredNorm();
            norms.minCoeff(&j, &k);
            // std::cout << "j, k" << j << " " << k << std::endl;
            u.row(i).setZero();
            u(i, j) = 1;
        }

        Eigen::MatrixXd ub = u * codebook;

#pragma omp parallel for
        for (int i = 0; i < x.cols(); ++i) {
            int k, j;
            Eigen::MatrixXd tmp = (-ub).colwise() + x.col(i);
            tmp.array().colwise() *= mask.col(i).array();
            auto norms = tmp.colwise().squaredNorm();
            norms.minCoeff(&k, &j);
            // std::cout << "k, j" << k << " " << j << std::endl;
            v.row(i).setZero();
            v(i, j) = 1;
        }
        std::cout << "iter " << t << std::endl;

        auto lose = ((u * codebook * v.transpose()) - x).cwiseProduct(mask).squaredNorm();
        std::cout << "lost " << std::sqrt(lose) / mask.sum() << std::endl;
    }
}
    
int main(int argc, char**argv){
    Arguments args = getArgs(argc, argv);

    Eigen::MatrixXd sourceRate = Eigen::MatrixXd::Zero(args.sD1, args.sD2);
    Eigen::MatrixXd sourceMask = Eigen::MatrixXd::Zero(args.sD1, args.sD2);
    loadMatrix(args.source, sourceRate, sourceMask);
    Eigen::MatrixXd memU = Eigen::MatrixXd::Zero(args.sD1, args.nClustersU);
    Eigen::MatrixXd memI = Eigen::MatrixXd::Zero(args.sD2, args.nClustersI);
    loadMem(args.cu, memU);
    loadMem(args.ci, memI);

    
    loadFilled(args.all, sourceRate, sourceMask);
    // Eigen::MatrixXd sourceP, sourceQ;
    // loadModel(args.sourceModel, sourceP, sourceQ);
    // Eigen::MatrixXd sourceFilled = sourceP * sourceQ.transpose();
    // sourceRate.array() += ((1 - sourceMask.array()) * sourceFilled.array());

    Eigen::MatrixXd s;
    auto origU = memU;
    auto origI = memI;
    // nmtf(args.nmtfRate, args.nmtfConv, sourceRate, sourceMask, memU, s, memI);

    std::cout << "diff U:" << (memU - origU).squaredNorm() << std::endl
              << "diff I:" << (memI - origI).squaredNorm() << std::endl;

    // sourceRate.array() *= sourceMask.array();
    
    // constructing codebook
    Eigen::MatrixXd codebook = (memU.transpose() * sourceRate * memI)
        .cwiseQuotient(memU.transpose() * Eigen::MatrixXd::Ones(sourceRate.rows(), sourceRate.cols()) * memI);
    std::cout << "finish construting codebook" << std::endl;

    std::cout << codebook << std::endl;
    
    // transfer codebook
    Eigen::MatrixXd targetRate = Eigen::MatrixXd::Zero(args.rD1, args.rD2);
    Eigen::MatrixXd targetMask = Eigen::MatrixXd::Zero(args.rD1, args.rD2);
    loadMatrix(args.target, targetRate, targetMask);

    // sample holdout data    
    std::default_random_engine gen;
    std::vector<std::pair<int, int>>holdoutInds;
    std::vector<double>holdoutAnss;
    std::bernoulli_distribution bern(args.holdout);
    for (auto i = 0; i < targetRate.rows(); ++i) {
        for (auto j = 0; j < targetRate.cols(); ++j) {
            if (targetMask(i,j) == 1 && bern(gen)) {
                holdoutInds.push_back(std::make_pair(i, j));
                holdoutAnss.push_back(targetRate(i, j));                
                targetRate(i, j) = 0;
                targetMask(i, j) = 0;                    
            }
        }
    }
    std::cout << "finish sampling holdout data" << std::endl;

    
    transferCodebook(args.nIters, targetRate, targetMask, codebook, memU, memI);
    std::cout << "finish transfering codebook" << std::endl;

    // fill in target matrix       
    targetRate.array() += (1 - targetMask.array()) * (memU * codebook * memI.transpose()).array();

    double squareErr = 0;
    for (auto i = 0u; i < holdoutInds.size(); ++i) {
        int x = holdoutInds[i].first;
        int y = holdoutInds[i].second;
        std::cout << targetRate(x, y) << " " << holdoutAnss[i] << std::endl;
        squareErr += SQUARE(targetRate(x, y) - holdoutAnss[i]);
    }

    std::cout << holdoutInds.size() << "holdout samples." << std::endl
              << sqrt(squareErr / holdoutInds.size()) << std::endl;

    std::fstream ftr;
    ftr.open("train.txt", std::ios::out);
    for (auto i = 0u; i < targetRate.rows(); ++i) {
        for (auto j = 0; j < targetRate.cols(); ++j) {
            if (targetRate(i, j) != 0) {
                ftr << i << " " << j << " " <<  targetRate(i, j) << std::endl;
            }
        }
    }
    std::fstream ftest;
    ftr.open("test.txt", std::ios::out);
    for (auto i = 0u; i < holdoutInds.size(); ++i) {
        int x = holdoutInds[i].first;
        int y = holdoutInds[i].second;
        ftest << x << " " << y << targetRate(x, y) << " " << holdoutAnss[i] << std::endl;
        squareErr += SQUARE(targetRate(x, y) - holdoutAnss[i]);
    }
    return 0;
}



// lagacy
    // if (args.sourceModel != "") {
    //     Eigen::MatrixXd sourceP, sourceQ;
    //     loadModel(args.sourceModel, sourceP, sourceQ);
    //     Eigen::MatrixXd sourceFilled = sourceP * sourceQ.transpose();
    //     sourceRate.array() += ((1 - sourceMask.array()) * sourceFilled.array());
    //     std::cerr << "finish loading model " << sourceRate.rows() << " " << sourceRate.cols() <<  std::endl;
    // }
    
    // Eigen::MatrixXd s;
    // Eigen::MatrixXd f;
    // Eigen::MatrixXd g;
    // nmtf(args.nmtfRate, args.nmtfConv,
    //      args.nClustersU, args.nClustersI,
    //      sourceRate, sourceMask, f, s, g);
