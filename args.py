import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cont_s1", "-cs1", help="Contour size parameter 1",
                        type=int, default=600)
    parser.add_argument("--cont_s2", "-cs2", help="Contour size parameter 2",
                        type=int, default=600)

    parser.add_argument("--cont_d1", "-cd1", help="Contour directionality parameter 1",
                        type=float, default=0.9)
    parser.add_argument("--cont_d2", "-cd2", help="Contour directionality parameter 2",
                        type=float, default=0.9)

    parser.add_argument("--filter_size", "-fs", help="Filter size for filtering the outliers",
                        type=int, default=100)

    parser.add_argument("--resolution", "-r", help="Manually set pixel resolution",
                        type=float, default=0.5)
    parser.add_argument("--measure", "-m", help="Weather to measure the pixel resolution in image",
                        default=False, action='store_true')
    parser.add_argument("--originx", "-ox", help="Set x origin",
                        type=int, default=-1)
    parser.add_argument("--originy", "-oy", help="Set y origin",
                        type=int, default=-1)
    parser.add_argument("--occupied", "-oc", help="Occupied pixel threshold",
                        type=float, default=0.5)
    parser.add_argument("--free", "-fr", help="Free pixel threshold",
                        type=float, default=0.1)
    parser.add_argument("--x_scale", "-xs", help="Scaling factor of X-axis",
                        type=float, default=1.0)
    parser.add_argument("--x_bias", "-xb", help="Bias value of X-axis in meters",
                        type=float, default=0.0)
    parser.add_argument("--y_scale", "-ys", help="Scaling factor of Y-axis",
                        type=float, default=1.0)
    parser.add_argument("--y_bias", "-yb", help="Bias value of Y-axis in meters",
                        type=float, default=0.0)

    return parser.parse_args()


d_args = get_arguments()
