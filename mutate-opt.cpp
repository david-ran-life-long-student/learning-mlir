//
// Created by dran on 3/10/26.
//
#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "MutateAffineFor.cpp"

int main(int argc, char **argv) {
    // this is the replacement for init all dialects
    mlir::DialectRegistry registry;
    registry.insert<mlir::affine::AffineDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<mlir::arith::ArithDialect>();
	registry.insert<mlir::func::FuncDialect>();

    mlir::PassRegistration<MutateAffineForPass>();

	// Register a pipeline that finds all affine.fors automatically
    mlir::PassPipelineRegistration<>(
        "mutate-all-for",
        "Run MutateAffineFor on all loops regardless of nesting",
        [](mlir::OpPassManager &pm) {
            // This nested call tells MLIR to find all affine.fors
            // and schedule the pass on them.
            pm.addNestedPass<mlir::affine::AffineForOp>(
                std::make_unique<MutateAffineForPass>()
            );
        }
    );

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "MutateAffineFor Tool", registry)
    );
}