//
// Created by dran on 3/10/26.
//
#ifndef MUTATE_AFFINE_FOR_OP
#define MUTATE_AFFINE_FOR_OP

#include <vector>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;


// this time we are taking all affine for an wrapping the loop body in a
// scf.if if the whatever loop bound is greater than 0, then we execute the loop
// this doesn't respect the semantics of the original program of course
struct MutateAffineForPass : public PassWrapper<MutateAffineForPass, OperationPass<ModuleOp>> {
    // This is required when not using TableGen
    StringRef getArgument() const final { return "mutate-affine-for"; }
    StringRef getDescription() const final { return "mutate affine for loops in a module"; }

	// note: this is necessary to tell mlir lazy dialect loading to load scf and arith
	// unless a dialect is used in the input file, it will not be loaded
	void getDependentDialects(DialectRegistry &registry) const override {
        // Tell the PassManager to explicitly load these dialects before running the pass
        registry.insert<scf::SCFDialect>();
        registry.insert<arith::ArithDialect>();
    }

    void runOnOperation() {

		// llvm::outs() << "[dran] insert started\n";

        Operation* current = getOperation();
        int for_count = 0;

        current->walk([&](affine::AffineForOp for_op) {
        	// get the loop body as a block
        	Block* loop_body = for_op.getBody();
        	// get builder
        	OpBuilder builder(loop_body, loop_body->begin());

        	// make a new scf.if
        	// we'll make a true condition to make things easy
        	Value always_true = arith::ConstantIntOp::create(  // note new API: builder.create has been deprecated
				builder,
            	for_op.getLoc(), // debug location, not insert location
            	1, // value
            	1  // one bit wide for bool
            	);
       		scf::IfOp created_if = scf::IfOp::create(
				builder,
        	    for_op.getLoc(),
            	always_true,
	            false // don't create else region
    	        );

        	// now we move the origial loop body into the new block inside if
        	std::vector<Operation*> to_move;
        	for (auto &each_op : *loop_body) {
            	if (&each_op != always_true.getDefiningOp() \
                && &each_op != created_if.getOperation() \
                && !each_op.hasTrait<mlir::OpTrait::IsTerminator>()) {
                	to_move.push_back(&each_op);
            	}
        	}
        	for (Operation* op : to_move) {
            	op->moveBefore(created_if.getBody()->getTerminator());
        	}

			// llvm::outs() << "[dran] did 1 insert\n" ;
        });

    }
};

#endif // MUTATE_AFFINE_FOR_OP