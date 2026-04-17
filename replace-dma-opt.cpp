#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "ReplaceDMA.cpp"

int main(int argc, char **argv) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::affine::AffineDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::func::FuncDialect>();

    // Correctly registering your pass
    mlir::PassRegistration<InlinedDMAPass>();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "DMA Inlining Tool", registry)
    );
}