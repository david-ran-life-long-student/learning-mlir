//
// Created by dran on 2/22/26.
//
#ifndef COUNTAFFINEFOR_CPP
#define COUNTAFFINEFOR_CPP

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

// the strategy is to use at module op that is scheduled one at the top of the IR tree
// then use a lambda that only get's called on affine.for to count the number of for loops
struct CountAffineForPass : public PassWrapper<CountAffineForPass, OperationPass<ModuleOp>> {
    // This is required when not using TableGen
    StringRef getArgument() const final { return "my-affine-analysis"; }
    StringRef getDescription() const final { return "count affine for loops in a module"; }

    void runOnOperation() override {
        Operation* current = getOperation();
        int for_count = 0;

        current->walk([&](affine::AffineForOp op) {
            ++for_count;
        });

        llvm::outs() << "[dran] for loop count: " << for_count << '\n';
    }
};

#endif // COUNTAFFINEFOR_CPP
