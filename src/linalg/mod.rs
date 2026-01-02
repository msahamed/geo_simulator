pub mod solver;
pub mod direct;
pub mod iterative;
pub mod preconditioner;
pub mod picard;
pub mod jfnk;
pub mod scaling;
pub mod amg;

pub use solver::{Solver, SolverStats, LinearOperator};
pub use direct::DirectSolver;
pub use iterative::{ConjugateGradient, BiCGSTAB, GMRES};
pub use preconditioner::{Preconditioner, JacobiPreconditioner};
pub use picard::{picard_solve, PicardConfig, PicardStats};
pub use jfnk::{jfnk_solve, jfnk_solve_nondimensional, JFNKConfig, JFNKStats};
pub use scaling::CharacteristicScales;
pub use amg::AMGPreconditioner;
