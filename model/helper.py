import torch


def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))
    v_mag = torch.max(v_mag, torch.tensor([1e-8], requires_grad=True,
                                          dtype=torch.float32).cuda())
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    return v


def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)
    return out


def compute_rotation_matrix_from_ortho6d(raw_output):
    """
    This orthogonalization is different from the paper. see this issue:
        https://github.com/papagina/RotationContinuity/issues/2

    However, cross product and Gram-Schmidt is equivalent in R^3,but cross
    product only works in R^3 but the Gram-Schmidt can work in higher
    dimension. see this question:
        https://math.stackexchange.com/questions/1847465/why-to-use-gram-schmidt-process-to-orthonormalise-a-basis-instead-of-cross-produ
    """
    # first 3 elements are pos
    x_raw = raw_output[:, 6:9]
    y_raw = raw_output[:, 9:12]

    x = normalize_vector(x_raw)
    z = cross_product(x, y_raw)
    z = normalize_vector(z)
    y = cross_product(z, x)

    x = x.view(-1, 3)
    y = y.view(-1, 3)
    z = z.view(-1, 3)

    output = torch.cat((raw_output[:, :6], x, y, z), 1)
    return output


def compute_geodesic_distance_from_two_matrices(m1, m2):
    m1 = torch.reshape(m1, (-1, 3, 3))
    m2 = torch.reshape(m2, (-1, 3, 3))

    batch = m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1, 2))

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()))
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda())*-1)

    theta = torch.acos(cos)
    return theta


def compute_rotation_matrix_from_euler_sin_cos(euler_sin_cos):
    batch = euler_sin_cos.shape[0]

    s1 = euler_sin_cos[:, 3].view(batch, 1)
    c1 = euler_sin_cos[:, 0].view(batch, 1)
    s2 = euler_sin_cos[:, 4].view(batch, 1)
    c2 = euler_sin_cos[:, 1].view(batch, 1)
    s3 = euler_sin_cos[:, 5].view(batch, 1)
    c3 = euler_sin_cos[:, 2].view(batch, 1)

    row1 = torch.cat((c2*c3, -s2, c2*s3), 1).view(-1, 3)
    row2 = torch.cat((c1*s2*c3+s1*s3, c1*c2, c1*s2*s3-s1*c3), 1).view(-1, 3)
    row3 = torch.cat((s1*s2*c3-c1*s3, s1*c2, s1*s2*s3+c1*c3), 1).view(-1, 3)

    matrix = torch.cat((row1, row2, row3), 1)
    return matrix
