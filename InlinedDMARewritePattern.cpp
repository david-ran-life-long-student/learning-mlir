#ifndef INLINEDDMARWRITEPATTERN_CPP
#define INLINEDDMARWRITEPATTERN_CPP

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h" // Added for CallOp
#include "mlir/IR/PatternMatch.h"

// Target the standard func::CallOp instead of the non-existent custom op
struct InlinedDMARewritePattern : public mlir::OpRewritePattern<mlir::func::CallOp> {

  InlinedDMARewritePattern(mlir::MLIRContext *context)
      : OpRewritePattern<mlir::func::CallOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult matchAndRewrite(mlir::func::CallOp op,
                                      mlir::PatternRewriter &rewriter) const override {

    // ONLY rewrite calls that are targeting our specific stub
    if (op.getCallee() != "hardware_dma_stub") {
        return mlir::failure();
    }

    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *ctx = rewriter.getContext();

    // Extract the SSA values directly from the CallOp operands
    mlir::Value allocMemRef = op.getOperand(0);
    mlir::Value mainMemRef  = op.getOperand(1);
    mlir::Value dynRow      = op.getOperand(2);
    mlir::Value dynCol      = op.getOperand(3);

    // Build the Affine Expressions
    mlir::AffineExpr d0 = rewriter.getAffineDimExpr(0);
    mlir::AffineExpr d1 = rewriter.getAffineDimExpr(1);
    mlir::AffineExpr d2 = rewriter.getAffineDimExpr(2);
    mlir::AffineExpr d3 = rewriter.getAffineDimExpr(3);
    mlir::AffineExpr s0 = rewriter.getAffineSymbolExpr(0);
    mlir::AffineExpr s1 = rewriter.getAffineSymbolExpr(1);

    mlir::AffineExpr rowExpr = d0 * 128 + d2 + s0;
    mlir::AffineExpr colExpr = d1 * 64 + d3 + s1;

    auto writebackMap = mlir::AffineMap::get(4, 2, {rowExpr, colExpr}, ctx);

    llvm::SmallVector<int64_t, 4> lowerBounds = {0, 0, 0, 0};
    llvm::SmallVector<int64_t, 4> upperBounds = {4, 1, 128, 64};
    llvm::SmallVector<int64_t, 4> steps = {1, 1, 1, 1};

    // Note the mlir::affine namespace update here
    mlir::affine::buildAffineLoopNest(
        rewriter, loc, lowerBounds, upperBounds, steps,
        [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc, mlir::ValueRange ivs) {

          auto loadOp = nestedBuilder.create<mlir::affine::AffineLoadOp>(
              nestedLoc, allocMemRef, ivs);

          llvm::SmallVector<mlir::Value, 6> mapOperands;
          mapOperands.push_back(ivs[0]);
          mapOperands.push_back(ivs[1]);
          mapOperands.push_back(ivs[2]);
          mapOperands.push_back(ivs[3]);
          mapOperands.push_back(dynRow);
          mapOperands.push_back(dynCol);

          nestedBuilder.create<mlir::affine::AffineStoreOp>(
              nestedLoc, loadOp.getResult(), mainMemRef, writebackMap, mapOperands);
        });

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

#endif // INLINEDDMARWRITEPATTERN_CPP