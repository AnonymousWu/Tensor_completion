def add_general_arguments(parser):

    parser.add_argument(
        '--I',
        type=int,
        default=64,
        metavar='int',
        help='Input tensor size in first dimension (default: 64)')
    parser.add_argument(
        '--J',
        type=int,
        default=64,
        metavar='int',
        help='Input tensor size in second dimension (default: 64)')
    parser.add_argument(
        '--K',
        type=int,
        default=64,
        metavar='int',
        help='Input tensor size in third dimension (default: 64)')
    parser.add_argument(
        '--R',
        type=int,
        default=10,
        metavar='int',
        help='Input CP decomposition rank (default: 10)')
    parser.add_argument(
        '--num-iter-ALS-implicit',
        type=int,
        default=100,
        metavar='int',
        help='Number of iterations (sweeps) to run ALS with implicit CG (default: 10)')
    parser.add_argument(
        '--num-iter-ALS-explicit',
        type=int,
        default=100,
        metavar='int',
        help='Number of iterations (sweeps) to run ALS with explicit CG (default: 10)')
    parser.add_argument(
        '--num-iter-CCD',
        type=int,
        default=100,
        metavar='int',
        help='Number of iterations (updates to each column of each matrix) for which to run CCD (default: 10)')
    parser.add_argument(
        '--num-iter-SGD',
        type=int,
        default=1000,
        metavar='int',
        help='Number of iteration, each iteration computes subgradients from --sample-frac-SGD of the total number of nonzeros in tensor (default: 10)')
    parser.add_argument(
        '--time-limit',
        type=float,
        default=30,
        metavar='float',
        help='Number of seconds after which to terminate tests for either ALS, SGD, or CCD if number of iterations is not exceeded (default: 30)')
    parser.add_argument(
        '--obj-freq-CCD',
        type=int,
        default=1,
        metavar='int',
        help='Number of iterations after which to calculate objective (time for objective calculation not included in time limit) for CCD (default: 1)')
    parser.add_argument(
        '--obj-freq-SGD',
        type=int,
        default=1,
        metavar='int',
        help='Number of iterations after which to calculate objective (time for objective calculation not included in time limit) for SGD (default: 1)')
    parser.add_argument(
        '--err-thresh',
        type=float,
        default=1.E-5,
        metavar='float',
        help='Residual norm threshold at which to halt if number of iterations does not expire (default 1.E-5)')
    parser.add_argument(
        '--sp-fraction',
        type=float,
        default=.1,
        metavar='float',
        help='sparsity (default: .1)')
    parser.add_argument(
        '--use-sparse-rep',
        type=int,
        default=1,
        metavar='int',
        help='whether to store tensor as sparse (default: 1, i.e. True)')
    parser.add_argument(
        '--block-size-ALS-implicit',
        type=int,
        default=0,
        metavar='int',
        help='block-size for implicit ALS (default: 0, meaning to use a single block)')
    parser.add_argument(
        '--block-size-ALS-explicit',
        type=int,
        default=0,
        metavar='int',
        help='block-size for explicit ALS (default: 0, meaning to use a single block)')
    parser.add_argument(
        '--regularization-ALS',
        type=float,
        default=0.00001,
        metavar='float',
        help='regularization for ALS (default: 0.00001)')
    parser.add_argument(
        '--regularization-CCD',
        type=float,
        default=0.00001,
        metavar='float',
        help='regularization for CCD (default: 0.00001)')
    parser.add_argument(
        '--regularization-SGD',
        type=float,
        default=0.00001,
        metavar='float',
        help='regularization for SGD (default: 0.00001)')
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.01,
        metavar='float',
        help='learning rate for SGD (default: 0.01)')
    parser.add_argument(
        '--sample-frac-SGD',
        type=float,
        default=0.1,
        metavar='float',
        help='sample size as fraction of total number of nonzeros for SGD (default: 0.01)')
    parser.add_argument(
        '--function-tensor',
        type=int,
        default=0,
        metavar='int',
        help='whether to use function tensor as test problem (default: 0, i.e. False, use explicit low CP-rank sampled tensor)')
    parser.add_argument(
        '--use-CCD-TTTP',
        type=int,
        default=1,
        metavar='int',
        help='whether to use TTTP for CCD contractions (default: 1, i.e. Yes)')
    parser.add_argument(
        '--tensor-file',
        type=str,
        default='',
        metavar='str',
        help='Filename from which to read tensor (default: None, use model problem)')




