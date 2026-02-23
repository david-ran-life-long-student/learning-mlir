//
// Created by dran on 2/22/26.
//
#ifndef COUNTAFFINEFOR_CPP
#define COUNTAFFINEFOR_CPP

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/include/mlir/Pass/Pass.h"

using namespace mlir;

// the strategy is to use at module op that is scheduled one at the top of the IR tree
// then use a lambda that only get's called on affine.for to count the number of for loops
struct CountAffinePass : public PassWrapper<CountAffinePass, OperationPass<ModuleOp>> {
    void runOnOperation() override {
        Operation* current = getOperation();
        int for_count = 0;

        current->walk([&](affine::AffineForOp* op) {
            ++for_count;
        })
    }
}

#endif // COUNTAFFINEFOR_CPP
