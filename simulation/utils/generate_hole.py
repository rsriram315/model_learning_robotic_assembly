# flake8: noqa
# code from this github repo: https://github.com/deanpham98/learn-seq
import os
import numpy as np

RADIUS = 0.0102
LENGTH = 0.03
DEPTH = 0.05
NO_BLOCKS = 150


def generate_hole_xml(radius, length, depth, no_blocks):
    '''
    Approximate hole using primitive objects
    Input:
        - radius: of the hole
        - length: width of each block
        - depth: of the hole
        - no_blocks used to approximate the hole
    Output: the xml file
    '''
    # Create the file
    asset_file = os.path.join("simulation", "assets", "round-hole.xml")

    if not os.path.exists(asset_file):
        os.mknod(asset_file)


    # write to file
    # tab character
    t = "  "

    # mass of each block
    mass = 1. / (no_blocks + 1)
    alpha = 2 * np.pi / no_blocks
    block_edge = 2 * radius * np.tan(alpha / 2)
    base_height = 0.02

    size = [length / 2, block_edge / 2, depth / 2]
    # r = radius + block_edge/2
    r = radius + length / 2

    # calculate vertex
    l = np.sqrt(length**2 + block_edge**2) / 2
    phi = -np.arctan(block_edge / length)
    P0 = np.array([r, 0, size[2] + base_height])
    P1 = np.array(
        [r * np.cos(alpha), r * np.sin(alpha), size[2] + base_height])
    V1 = np.array(
        [P0[0] - l * np.cos(phi), P0[1] - l * np.sin(phi), base_height])
    V2 = np.array([
        P0[0] - l * np.cos(phi), P0[1] - l * np.sin(phi), base_height + depth
    ])
    V3 = np.array(
        [P0[0] + l * np.cos(-phi), P0[1] + l * np.sin(-phi), base_height])
    V4 = np.array([
        P0[0] + l * np.cos(-phi), P0[1] + l * np.sin(-phi), base_height + depth
    ])
    V5 = np.array([
        P1[0] + l * np.cos(alpha + phi), P1[1] + l * np.sin(alpha + phi),
        base_height
    ])
    V6 = np.array([
        P1[0] + l * np.cos(alpha + phi), P1[1] + l * np.sin(alpha + phi),
        base_height + depth
    ])

    with open(asset_file, "w+") as f:
        print("File Opened")
        f.write("<mujoco model=\"round-hole\">\n")

        # triangular prism mesh
        f.write(t + "<asset>\n")
        f.write(
            t + t +
            "<mesh name=\"prism\" vertex=\"{:1.4f} {:1.5f} {:1.4f}  {:1.4f} {:1.5f} {:1.4f}  {:1.5f} {:1.5f} {:1.5f}  {:1.5f} {:1.5f} {:1.5f}  {:1.5f} {:1.5f} {:1.5f}  {:1.5f} {:1.5f} {:1.5f}\"/>\n"
            .format(V1[0], V1[1], V1[2], V2[0], V2[1], V2[2], V3[0], V3[1],
                    V3[2], V4[0], V4[1], V4[2], V5[0], V5[1], V5[2], V6[0],
                    V6[1], V6[2]))
        f.write(t + "</asset>\n\n")

        f.write(t + "<worldbody>\n")
        f.write(t + t +"<body>\n")
        f.write(t + t + t + "<body name=\"object\" pos=\"0 0 0\">\n")

        # hole base
        f.write(
            t + t + t + t +
            "<geom name=\"base\" type=\"box\" mass=\"{:1.5f}\" pos=\"0 0 {}\" group=\"1\" size=\"{:1.5f} {:1.5f} {}\"/>\n"
            .format(mass, base_height / 2, radius + block_edge / 2 +
                    0.002, radius + block_edge / 2 + 0.002, base_height / 2))

        # hole blocks
        for i in range(no_blocks):
            pos = [
                r * np.cos(i * alpha), r * np.sin(i * alpha),
                size[2] + base_height
            ]
            rotate_angle = i * alpha

            f.write(
                t + t + t + t +
                "<geom name=\"hole{}\" type=\"box\" mass=\"{:1.4f}\" group=\"1\" size=\"{:1.5f} {:1.5f} {:1.5f}\" pos=\"{:1.5f} {:1.5f} {:1.5f}\" euler=\"0 0 {:1.3f}\"/>\n"
                .format(i + 1, mass, size[0], size[1], size[2], pos[0], pos[1], pos[2], rotate_angle))

            if i == 0:
                f.write(
                    t + t + t + t +
                    "<geom name=\"prism1\" type=\"mesh\" mesh=\"prism\" group=\"1\"/>\n")
            elif i > 0:
                f.write(
                    t + t + t + t +
                    "<geom name=\"prism{}\" type=\"mesh\" mesh=\"prism\" group=\"1\" euler=\"0 0 {:1.5f}\"/>\n"
                    .format(i + 1, rotate_angle))

        f.write(t + t + t + "</body>\n")

        # site information
        f.write(
            t + t + t +
            "<site rgba=\"0 0 0 0\" size=\"0.005\" pos=\"0 0 -0.05\" name=\"bottom_site\"/>\n"
        )
        f.write(
            t + t + t +
            "<site rgba=\"0 0 0 0\" size=\"0.005\" pos=\"0 0 0.05\" name=\"top_site\"/>\n"
        )
        f.write(
            t + t + t +
            "<site rgba=\"0 0 0 0\" size=\"0.005\" pos=\"0.0204 0.0204 0\" name=\"horizontal_radius_site\"/>\n"
        )

        f.write(t + t + "</body>\n")
        f.write(t + "</worldbody>\n")
        f.write(t + "</mujoco>\n")

    f.close()


if __name__ == '__main__':
    generate_hole_xml(RADIUS, LENGTH, DEPTH, NO_BLOCKS)
