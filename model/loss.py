import torch


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
