from math import sqrt
from functools import partial

class Point:
    """ Every 2D point can be represented as tuple (x,y) """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.coor = (x,y)

    def __repr__(self):
        x, y = map(partial(round, ndigits=5), self.coor)
        return f"(x,y): ({x},{y})\t"

class Line:
    def __init__(self, p1: Point = None, p2: Point = None, eqn: tuple = None):
        """Equation of a 2D line is of the form:
        Ax + By + C = 0
        """
        if p1 and p2:
            self.eqn = self.get_line(p1, p2)
        elif eqn:
            self.eqn = eqn
        else:
            raise Exception("Not a valid line")

    def __repr__(self):
        a, b, c = map(partial(round, ndigits=5), self.eqn)
        return f"Eqn: {a}x + {b}y + {c}\t"

    def get_line(self, A, B) -> tuple:
        """ returns line of the form ax + by + c = 0 as (a,b,c)
        when c == 0, line is vertical. else c == 1 is nonvertical
        """
        Ax, Ay = A.coor
        Bx, By = B.coor
        # vertical line
        if Ax == Bx:
            return (-1, 0, Ax)
        M = (By - Ay)/(Bx - Ax)
        b = Ay - M * Ax
        return (M, -1, b)


def dist(A, B) -> float:
    return sqrt( (A.x - B.x)**2 + (A.y - B.y)**2 )

def coor_sum(A, B) -> Point:
    return Point(A.x + B.x, A.y + B.y)

# checks if line is parallel -> call this before line_intersect
def is_parallel(l1, l2) -> bool:
    a1, b1, _ = l1.eqn
    a2, b2, _ = l2.eqn
    if a1 == a2 == 0:
        return True
    if a1 == 0 or a2 == 0:
        return False
    return (b1/a1 == b2/a2)


# assumes lines aren't parallel; returns a point
def line_intersect(l1, l2) -> Point:
    a1, b1, c1 = l1.eqn
    a2, b2, c2 = l2.eqn

    x = (b1 * c2 - b2 * c1)/(a1 * b2 - a2 * b1)
    y = (a2 * c1 - a1 * c2)/(a1 * b2 - a2 * b1)
    return Point(x,y)


def perp_bis(A, B) -> Line:
    xmid, ymid = map(lambda x: x/2, coor_sum(A, B))
    line = Line(A,B)
    M, c, _ = line.eqn
    # current line is horizontal, perp bis has constant x
    if M == 0:
        return Line(Point(xmid, 0), Point(xmid, 1))
    # current line is vertical line, perp bis has constant y
    if c == 0:
        return Line(Point(0, ymid), Point(1, ymid))
    M2 = -1/M
    b2 = ymid - xmid * M2
    return Line(eqn=(M2, -1, b2))


def reflect(point, line) -> Point:
    proj = projection(point, line)
    return Point(2 * proj.x - point.x, 2 * proj.y - point.y)


def projection(point, line) -> Point:
    x1, y1 = point.coor
    M, c, b = line.eqn
    if M == 0:
        return Point(x1, b)
    if c == 0:
        return Point(b, y1)
    M2 = -1/M
    b2 = y1 - M2 * x1
    return line_intersect(line, Line(eqn=(M2, -1, b2)) )


def heron(A, B, C) -> float:
    a, b, c = dist(B,C), dist(A, C), dist(A, B)
    s = (a+b+c)/2
    return sqrt(s * (s-a) * (s-b) * (s-c))


# def distance_from_point_to_line(point, line):
#     return dist(point, projection(point, line))

def distance_from_point_to_line(point, line) -> float:
    a, b, c = line.eqn
    x,y = point.coor
    if abs(a) + abs(b) == 0:
        return 0
    return abs(a*x + b*y + c)/sqrt(a**2 + b**2)
    

if __name__ == "__main__":
    # methods have been tested on Three Triangle problem of 2020 ICPC GRNY
    pass