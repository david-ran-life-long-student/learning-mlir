//
// Created by dran on 2/22/26.
//
#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "CountAffineFor.cpp"

int main(int argc, char **argv) {
    // this is the replacement for init all dialects
    mlir::DialectRegistry registry;
    registry.insert<mlir::affine::AffineDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::func::FuncDialect>();

    mlir::PassRegistration<CountAffineForPass>();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "CountAffineFor Tool", registry)
    );
}