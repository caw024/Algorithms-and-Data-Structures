from math import sqrt

# points are 2D coordinates (x,y)
# lines are of the form (a,b,c) or ax + by + c = 0
# should test these methods more

def dist(A, B):
    return sqrt( sum( (A[i] - B[i])**2 for i in range(len(A)) ) )

def coor_sum(A, B):
    return [A[i] + B[i] for i in range(len(A))]

# returns line of the form ax + by + c = 0 as [a,b,c]
# when c == 0, line is vertical. else c == 1 is nonvertical
def getline(A, B):
    Ax, Ay = A
    Bx, By = B
    # vertical line
    if Ax == Bx:
        return [-1, 0, Ax]
    M = (By - Ay)/(Bx - Ax)
    b = Ay - M * Ax
    return [M, -1, b]


# assumes lines aren't parallel; returns a point
def lineintersect(line1, line2):
    # originally: (a,b,c): ax + by + c = 0
    a1, b1, c1 = line1
    a2, b2, c2 = line2

    x = (b1 * c2 - b2 * c1)/(a1 * b2 - a2 * b1)
    y = (a2 * c1 - a1 * c2)/(a1 * b2 - a2 * b1)
    return [x,y]


# returns line
def perpbis(A, B):
    xmid, ymid = map(lambda x: x/2, coor_sum(A, B))
    M, c, _ = getline(A, B)
    # horizontal
    if M == 0:
        return [0, -1, xmid]
    # vertical
    if c == 0:
        return [-1, 0, ymid]
    M2 = -1/M
    b2 = ymid - xmid * M2
    return [M2, -1, b2]

# returns point
def reflect(A, line):
    B = projection(A, line)
    return [2 * B[0] - A[0], 2 *B[1] - A[1]]


# returns point
def projection(point, line):
    x1, y1 = point
    M, c, b = line
    if M == 0:
        return [x1, b]
    if c == 0:
        return [b, y1]
    M2 = -1/M
    b2 = y1 - M2 * x1
    return lineintersect(line, [M2, -1, b2])


def heron(A, B, C):
    a, b, c = dist(B,C), dist(A, C), dist(A, B)
    s = (a+b+c)/2
    return sqrt(s * (s-a) * (s-b) * (s-c))