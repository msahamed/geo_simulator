pub mod solver;
pub mod direct;
pub mod iterative;
pub mod preconditioner;
pub mod picard;
pub mod jfnk;

pub use solver::{Solver, SolverStats, LinearOperator};
pub use direct::DirectSolver;
pub use iterative::{ConjugateGradient, BiCGSTAB};
pub use preconditioner::{Preconditioner, JacobiPreconditioner};
pub use picard::{picard_solve, PicardConfig, PicardStats};
pub use jfnk::{jfnk_solve, JFNKConfig, JFNKStats};
