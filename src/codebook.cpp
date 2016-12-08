#include "utils.hpp"
#include "nmtf.hpp"
#include <boost/program_options.hpp>
#include <iostream>

struct Arguments {
    std::string source, target;
    double nmtfConv, nmtfRate;
    int nClustersU, nClustersI, rD1, rD2;
};


Arguments getArgs(int argc, char**argv) {
    Arguments args;
    try {
        namespace po = boost::program_options;

        po::positional_options_description positional;
        positional.add("source", 1);
        positional.add("target", 1);

        po::options_description desc("===== Codebook =====");
        desc.add_options()
            ("source", po::value<std::string>(&args.source) -> required(), "source.txt")
            ("target", po::value<std::string>(&args.target) -> required(), "test.txt")
            ("help", "Print help message.")
            ("nmtfRate", po::value<double>(&args.nmtfRate) -> default_value(0.0001), "learning rate for nmtf")
            ("converge", po::value<double>(&args.nmtfConv) -> default_value(10), "converge criteria for nmtf")
            ("rD1", po::value<int>(&args.rD1) -> default_value(50000), "dimention of rating matrix")             
            ("rD2", po::value<int>(&args.rD2) -> default_value(5000), "dimention of rating matrix")             
            ("nClustersU", po::value<int>(&args.nClustersU) -> default_value(10), "number of user clusters")
            ("nClustersI", po::value<int>(&args.nClustersI) -> default_value(10), "number of item clusters");

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

int main(int argc, char**argv){
    Arguments args = getArgs(argc, argv);

    Eigen::MatrixXd sourceRate(args.rD1, args.rD2);
    Eigen::MatrixXd sourceMask(args.rD1, args.rD2);
    loadMatrix(args.source, sourceRate, sourceMask);

    Eigen::MatrixXd s;
    Eigen::MatrixXd f;
    Eigen::MatrixXd g;
    nmtf(args.nmtfRate, args.nmtfConv,
         args.nClustersU, args.nClustersI,
         sourceRate, sourceMask, f, s, g);
    
    return 0;
}
