#ifndef INLINEDDMARWRITEPATTERN_CPP
#define INLINEDDMARWRITEPATTERN_CPP

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h" 
#include "mlir/IR/PatternMatch.h"

// 1. THE CLASS DEFINITION
// We target the standard func::CallOp. Because we used the "demo stub" 
// workaround, the opaque custom hardware instruction is masquerading as a function call.
struct InlinedDMARewritePattern : public mlir::OpRewritePattern<mlir::func::CallOp> {

  InlinedDMARewritePattern(mlir::MLIRContext *context)
      : OpRewritePattern<mlir::func::CallOp>(context, /*benefit=*/1) {}

  // 2. THE CORE FUNCTION
  mlir::LogicalResult matchAndRewrite(mlir::func::CallOp op,
                                      mlir::PatternRewriter &rewriter) const override {

    // ==========================================
    // PHASE 1: MATCH
    // This pattern runs on EVERY function call in the module. We must 
    // filter aggressively so we don't accidentally rewrite a standard printf!
    // ==========================================
    
    // Check the symbol name of the target. If it is not our specific stub, bail out safely.
    if (op.getCallee() != "hardware_dma_stub") {
        return mlir::failure();
    }

    // ==========================================
    // PHASE 2: REWRITE
    // We found our target. Now we extract its state, define the polyhedral math, 
    // and generate the new loop structure.
    // ==========================================

    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *ctx = rewriter.getContext();

    // STEP A: Extract the State (The Operands)
    // We reach into the CallOp and pull out the raw SSA pointers and dynamic offsets.
    mlir::Value allocMemRef = op.getOperand(0); // The 4x1x128x64 local accelerator memory
    mlir::Value mainMemRef  = op.getOperand(1); // The 512x512 main system memory
    mlir::Value dynRow      = op.getOperand(2); // Dynamic row offset (unknown at compile time)
    mlir::Value dynCol      = op.getOperand(3); // Dynamic col offset (unknown at compile time)

    // STEP B: Construct the Polyhedral Math (The Routing Logic)
    // We are defining the AST of the affine map using C++ builder functions.
    // 'd' stands for dimension (the loop iterators). 's' stands for symbol (runtime constants).
    mlir::AffineExpr d0 = rewriter.getAffineDimExpr(0);
    mlir::AffineExpr d1 = rewriter.getAffineDimExpr(1);
    mlir::AffineExpr d2 = rewriter.getAffineDimExpr(2);
    mlir::AffineExpr d3 = rewriter.getAffineDimExpr(3);
    mlir::AffineExpr s0 = rewriter.getAffineSymbolExpr(0);
    mlir::AffineExpr s1 = rewriter.getAffineSymbolExpr(1);

    // This is the literal math defining where each byte physically lands in main memory.
    mlir::AffineExpr rowExpr = d0 * 128 + d2 + s0;
    mlir::AffineExpr colExpr = d1 * 64 + d3 + s1;

    // Combine the expressions into a single 2D map. 
    // Signature: 4 dimensions in, 2 symbols in -> 2 coordinates out.
    auto writebackMap = mlir::AffineMap::get(4, 2, {rowExpr, colExpr}, ctx);

    // STEP C: Define the Iteration Space
    // These vectors define the min, max, and step size of our 4 nested loops.
    // This perfectly matches the 32,768 elements of the local tile.
    llvm::SmallVector<int64_t, 4> lowerBounds = {0, 0, 0, 0};
    llvm::SmallVector<int64_t, 4> upperBounds = {4, 1, 128, 64};
    llvm::SmallVector<int64_t, 4> steps = {1, 1, 1, 1};

    // STEP D: Generate the Loop Nest
    // This utility automatically constructs the 4 nested affine.for operations.
    // The lambda function defines what happens inside the deepest loop block.
    mlir::affine::buildAffineLoopNest(
        rewriter, loc, lowerBounds, upperBounds, steps,
        [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc, mlir::ValueRange ivs) {

          // NOTE: We use 'nestedBuilder' here, not 'rewriter', because our insertion 
          // point is now deep inside the loop body, not out in the main function block.

          // Read a single byte from the local tile sequentially using the loop iterators (ivs).
          auto loadOp = mlir::affine::AffineLoadOp::create(
              nestedBuilder, nestedLoc, allocMemRef, ivs);

          // Prepare the inputs for our AffineMap. 
          // The polyhedral verifier strictly demands that all Dimensions (ivs) 
          // are pushed first, followed by all Symbols (dynRow, dynCol).
          llvm::SmallVector<mlir::Value, 6> mapOperands;
          mapOperands.push_back(ivs[0]);
          mapOperands.push_back(ivs[1]);
          mapOperands.push_back(ivs[2]);
          mapOperands.push_back(ivs[3]);
          mapOperands.push_back(dynRow);
          mapOperands.push_back(dynCol);

          // Write the byte to main memory, fusing the complex routing map directly to the store operation.
          mlir::affine::AffineStoreOp::create(
              nestedBuilder, nestedLoc, loadOp.getResult(), mainMemRef, writebackMap, mapOperands);
        });

    // STEP E: The Transactional Swap
    // Safely destroy the original func.call operation. The new loops have already 
    // been securely grafted into the AST right where the call used to be.
    rewriter.eraseOp(op);
    
    // Tell the GreedyPatternRewriteDriver that the mutation succeeded.
    return mlir::success();
  }
};

#endif // INLINEDDMARWRITEPATTERN_CPP
