#ifndef REPLACE_DMA_CPP
#define REPLACE_DMA_CPP

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "InlinedDMARewritePattern.cpp"

struct InlinedDMAPass
    : public mlir::PassWrapper<InlinedDMAPass, mlir::OperationPass<mlir::func::FuncOp>> {

    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InlinedDMAPass)

    llvm::StringRef getArgument() const final { return "inline-hw-dma"; }
    llvm::StringRef getDescription() const final { return "Replaces DMA stub calls with polyhedral loop nests."; }

    void runOnOperation() override {
        mlir::func::FuncOp function = getOperation();
        mlir::MLIRContext *context = &getContext();

        mlir::RewritePatternSet patterns(context);
        patterns.add<InlinedDMARewritePattern>(context);

        // Updated to the modern API name
        if (mlir::failed(mlir::applyPatternsGreedily(function, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

std::unique_ptr<mlir::Pass> createInlinedDMAPass() {
    return std::make_unique<InlinedDMAPass>();
}

#endif // REPLACE_DMA_CPP